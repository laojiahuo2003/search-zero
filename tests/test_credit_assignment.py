"""
Tests for CW-GRPO credit assignment (scripts/credit_assignment.py) and its
integration points in the training loop (scripts/train_grpo_search.py).

Run:  .venv/bin/python -m pytest tests/test_credit_assignment.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from scripts.credit_assignment import (
    normalize_contributions,
    build_token_advantages,
    rule_judge_turn,
    build_trace_text,
    parse_judge_response,
    CreditConfig,
    get_credit_config,
)
from scripts.train_grpo_search_MI300X import (
    grpo_loss,
    format_reward,
    accuracy_reward,
)


def make_turns(queries, obs_list, answer_text):
    """Rollout-shaped turn dicts (same schema as generate_with_search)."""
    turns = []
    for q, obs in zip(queries, obs_list):
        turns.append({
            'text': f'THOUGHT: t\nACTION: SEARCH: {q}',
            'query': q,
            'observation': obs,
            'gen_ids': [1, 2, 3],  # dummy token ids
            'credit': None,       # filled by Phase 2.5
        })
    turns.append({
        'text': f'THOUGHT: final\nACTION: ANSWER: {answer_text}',
        'query': None,
        'observation': None,
        'gen_ids': [4, 5],
        'credit': None,
    })
    return turns


CFG = CreditConfig(mode='rule', gamma=1.0, fallback_uniform=True)


class TestNormalizeContributions:
    def test_softmax_sums_to_one(self):
        w = normalize_contributions([1.0, 0.0], 1.0)
        assert abs(sum(w) - 1.0) < 1e-6
        assert w[0] > w[1]

    def test_hard_renormalizes(self):
        w = normalize_contributions([1.0, 0.0], 10.0)
        assert abs(w[0] - 1.0) < 1e-6 and abs(w[1]) < 1e-6

    def test_equal_contributions_degenerate_to_uniform(self):
        # softmax(c - max(c)) of equal values -> uniform; x*N == 1 per turn.
        w = normalize_contributions([0.5, 0.5], 1.0)
        assert abs(w[0] - 0.5) < 1e-6 and abs(w[1] - 0.5) < 1e-6


class TestRuleJudge:
    """Negative-list design: only clearly useless turns score 0."""

    def test_bridge_entity_hop_scores(self):
        # Multi-hop question: the first hop retrieves the BRIDGE entity's
        # document, which contains no gold word — it must still score.
        turns = make_turns(
            ["Frank Herbert birthplace", "Dune author education"],
            ["Frank Herbert was born in Tacoma Washington.",
             "He attended the University of Washington."],
            "University of Washington",
        )
        ret, thk = rule_judge_turn("Where did the Dune author study?",
                                  "University of Washington", turns, 0)
        assert ret == 1 and thk == 1

    def test_no_new_information_scores_zero(self):
        # Second obs is a near-duplicate of the first: nothing novel.
        turns = make_turns(
            ["Alice", "Alice"],
            ["Alice Johnson was the district attorney.",
             "Alice Johnson was the district attorney of X."],
            "Alice Johnson",
        )
        ret, thk = rule_judge_turn("q", "Alice Johnson", turns, 1)
        assert ret == 0

    def test_no_result_marker_scores_zero(self):
        turns = [
            {'text': 't', 'query': 'zzzqqq', 'gen_ids': [1], 'credit': None,
             'observation': 'OBSERVATION: No results found for "zzzqqq".'},
            {'text': 't', 'query': '', 'gen_ids': [2], 'credit': None,
             'observation': None},
        ]
        ret, thk = rule_judge_turn("q", "gold", turns, 0)
        assert ret == 0

    def test_repeat_query_scores_thinking_zero(self):
        turns = make_turns(
            ["Alice Johnson", "Alice Johnson"],
            ["Alice was DA.", "Alice was DA."],
            "Alice",
        )
        ret, thk = rule_judge_turn("q", "Alice", turns, 1)
        assert thk == 0

    def test_near_duplicate_query_scores_thinking_zero(self):
        # 5/6 tokens identical -> Jaccard ~0.83 >= 0.8 default threshold.
        turns = make_turns(
            ["Alice Johnson attorney of X", "Alice Johnson attorney of X county"],
            ["obs one has several words.", "obs two has several words."],
            "Alice",
        )
        ret, thk = rule_judge_turn("q", "Alice", turns, 1)
        assert thk == 0

    def test_rephrased_query_not_penalized(self):
        # Upstream judge: varying a failed query is a USEFUL attempt.
        turns = make_turns(
            ["Alice Johnson", "district attorney Alice"],
            ["Alice Johnson was DA.", "She served 1990-1998."],
            "Alice Johnson",
        )
        ret, thk = rule_judge_turn("q", "Alice Johnson", turns, 1)
        assert thk == 1

    def test_empty_query_scores_thinking_zero(self):
        turns = [{'query': '', 'observation': 'x y z w', 'text': 't',
                  'gen_ids': [1], 'credit': None},
                 {'query': '', 'observation': '', 'text': 't',
                  'gen_ids': [2], 'credit': None}]
        ret, thk = rule_judge_turn("q", "gold", turns, 0)
        assert thk == 0

    def test_gold_word_exempts_novelty_threshold(self):
        # Only 4 novel words (< threshold 10), but one is a new gold word
        # -> the exemption keeps the turn alive.
        cfg = CreditConfig(mode='rule', gamma=1.0, fallback_uniform=True,
                           rule_min_new_words=10)
        turns = make_turns(
            ["Alice Johnson"],
            ["Alice was born here."],
            "Alice Johnson",
        )
        ret, thk = rule_judge_turn("q", "Alice Johnson", turns, 0, cfg)
        assert ret == 1

    def test_threshold_configurable(self):
        # Same turn, no gold word: fails the raised novelty threshold.
        cfg = CreditConfig(mode='rule', gamma=1.0, fallback_uniform=True,
                           rule_min_new_words=50)
        turns = make_turns(
            ["bridge entity"],
            ["This document has only a handful of words."],
            "some answer",
        )
        ret, thk = rule_judge_turn("q", "some answer", turns, 0, cfg)
        assert ret == 0


class TestBuildTokenAdvantages:
    def _turns(self, credits):
        turns = make_turns(
            ["q1", "q2"], ["obs1", "obs2"], "answer",
        )
        turns[0]['credit'] = credits[0]
        turns[1]['credit'] = credits[1]
        return turns

    def test_positive_adv_reallocates(self):
        turns = self._turns([1.0, 0.0])
        adv = build_token_advantages(turns, 2.0, CFG)
        assert len(adv) == 3 + 3 + 2
        assert adv[0] > 2.0          # turn0 boosted
        assert adv[3] < 2.0          # turn1 reduced
        assert adv[-1] == 2.0        # ANSWER turn unchanged

    def test_credit_conservation(self):
        turns = self._turns([1.0, 0.0])
        adv = build_token_advantages(turns, 2.0, CFG)
        # mean over non-final turns == group advantage
        assert abs((adv[0] + adv[3]) / 2.0 - 2.0) < 1e-4

    def test_negative_adv_broadcasts(self):
        turns = self._turns([1.0, 0.0])
        adv = build_token_advantages(turns, -1.0, CFG)
        assert torch.allclose(adv, torch.full_like(adv, -1.0))

    def test_zero_adv_broadcasts(self):
        turns = self._turns([1.0, 0.0])
        adv = build_token_advantages(turns, 0.0, CFG)
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_all_zero_contributions_fallback_uniform(self):
        turns = self._turns([0.0, 0.0])
        adv = build_token_advantages(turns, 2.0, CFG)
        assert torch.allclose(adv, torch.full_like(adv, 2.0))

    def test_all_zero_contributions_no_fallback_zeroes_turns(self):
        cfg = CreditConfig(mode='rule', gamma=1.0, fallback_uniform=False)
        turns = self._turns([0.0, 0.0])
        adv = build_token_advantages(turns, 2.0, cfg)
        assert torch.allclose(adv, torch.tensor([0, 0, 0, 0, 0, 0, 2.0, 2.0],
                                                dtype=torch.float32))

    def test_single_turn_broadcasts(self):
        turns = make_turns([], [], "answer")
        adv = build_token_advantages(turns, 2.0, CFG)
        assert torch.allclose(adv, torch.full_like(adv, 2.0))


class TestJudgeResponseParsing:
    def test_fenced_json(self):
        content = '```json\n{"analysis": "x", "thinking_reward": 1, "retrieval_reward": 0}\n```'
        assert parse_judge_response(content) == (0, 1)

    def test_bare_json(self):
        assert parse_judge_response(
            '{"thinking_reward": 0, "retrieval_reward": 1}') == (1, 0)

    def test_garbage_returns_none(self):
        assert parse_judge_response("not json") is None
        assert parse_judge_response("") is None


class TestTraceText:
    def test_trace_format(self):
        turns = make_turns(["q1"], ["obs1"], "ans")
        trace = build_trace_text("Q?", turns, 1)
        assert trace.startswith("Question: Q?")
        assert "Agent:" in trace and "Information:" in trace


class TestConfig:
    def test_env_config(self, monkeypatch):
        monkeypatch.setenv("CREDIT_MODE", "rule")
        monkeypatch.setenv("CREDIT_GAMMA", "2.5")
        cfg = get_credit_config()
        assert cfg.mode == "rule"
        assert cfg.gamma == 2.5

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.setenv("CREDIT_MODE", "bogus")
        with pytest.raises(ValueError):
            get_credit_config()


class TestTrainingLoopIntegration:
    """Phase 2 -> 2.5 -> 3 data flow without a GPU (dummy logprobs)."""

    GT = "Alice Johnson"
    QUESTION = "Who was the district attorney?"

    def test_end_to_end_reallocation_and_loss(self):
        turns_a = make_turns(
            ["district attorney Alice", "Alice Johnson career"],
            ["The DA was a woman named Alice.", "Johnson served 1990-1998."],
            "Alice Johnson",
        )
        turns_b = make_turns(
            ["population of France", "population of France"],
            ["France has 67M people.", "France has 67M people."],
            "Bob Smith",
        )
        group_turns = [turns_a, turns_b]
        group_texts = ["\n".join(t['text'] for t in turns_a),
                       "\n".join(t['text'] for t in turns_b)]

        # Phase 2: rewards + group-normalized advantages.
        rewards = [format_reward(t) + accuracy_reward(t, self.GT)
                   for t in group_texts]
        r = torch.tensor(rewards, dtype=torch.float32)
        advantages = (r - r.mean()) / (r.std() + 1e-4)
        assert advantages[0] > advantages[1]

        # Phase 2.5: rule judge, main-loop logic (skip adv <= 0 groups).
        for g in range(2):
            turns = group_turns[g]
            if len(turns) < 2:
                continue
            if advantages[g] <= 0 and CFG.judge_only_positive_adv:
                continue
            for idx in range(len(turns) - 1):
                if not turns[idx].get('query'):
                    turns[idx]['credit'] = 0.0
                    continue
                ret, thk = rule_judge_turn(self.QUESTION, self.GT, turns, idx)
                turns[idx]['credit'] = float(ret * thk)
        assert turns_a[0]['credit'] == 1.0
        assert turns_a[1]['credit'] == 1.0
        assert all(t['credit'] is None for t in turns_b[:-1])  # skipped

        # Phase 3: per-token advantage + grpo_loss, both trajectories.
        for g in range(2):
            turns = group_turns[g]
            adv = build_token_advantages(turns, float(advantages[g]), CFG)
            total = sum(len(t['gen_ids']) for t in turns)
            assert adv.shape == (total,)

            new_lps = torch.randn(total, dtype=torch.float32, requires_grad=True)
            old_lps = torch.randn(total, dtype=torch.float32)
            loss = grpo_loss(new_lps, old_lps, adv)
            assert loss.dim() == 0 and torch.isfinite(loss)
            loss.backward()
            assert new_lps.grad is not None

        # Unequal credits must change the advantage (turn0 boosted).
        turns_a[0]['credit'] = 1.0
        turns_a[1]['credit'] = 0.0
        adv = build_token_advantages(turns_a, float(advantages[0]), CFG)
        assert adv[0] > float(advantages[0])
        assert adv[-1] == float(advantages[0])

    def test_vanilla_scalar_advantage_still_works(self):
        loss = grpo_loss(torch.randn(11), torch.randn(11), 0.5)
        assert loss.dim() == 0 and torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
