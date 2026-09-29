"""
Tests for multi-prompt batched generation (train_grpo_search.py).

The generation functions are exercised WITHOUT a GPU using a fake model whose
generate() returns a deterministic per-row marker token. This pins down the
contracts the training loop depends on:

  1. row ordering: row = s * num_generations + g (sample-major)
  2. left padding + attention mask assembly for ragged prompt lengths
  3. per-row state machine: SEARCH appends an observation, ANSWER finishes
  4. single-prompt wrapper backward compatibility (P=1 => G results)

Run:  .venv/bin/python -m pytest tests/test_batched_generation.py -v
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.train_grpo_search import (
    batched_generate_with_search,
    batched_generate_multi_prompt,
)

SEARCH_MARK = 111   # decodes to "ACTION: SEARCH: fake query"
ANSWER_MARK = 222   # decodes to "ACTION: ANSWER: fake answer"
THOUGHT_MARK = 333  # decodes to a THOUGHT with no ACTION -> row stops

_MARKER_TEXT = {
    SEARCH_MARK: "THOUGHT: t\nACTION: SEARCH: fake query",
    ANSWER_MARK: "THOUGHT: t\nACTION: ANSWER: fake answer",
    THOUGHT_MARK: "THOUGHT: only thinking",
}


class FakeTokenizer:
    """Minimal tokenizer: deterministic ids, marker-aware decode."""

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        # Only used for observation/marker tokens; content does not matter.
        return [501]

    def decode(self, ids, skip_special_tokens=False):
        for mark, text in _MARKER_TEXT.items():
            if mark in ids:
                return text
        return "plain"


class FakeModel:
    """generate() returns one marker token per active row per call.

    schedule: list where entry i is either an int marker (same for every row)
    or a callable(row_idx) -> marker. Calls beyond the schedule repeat the
    last entry. Every call is recorded for attention-mask assertions.
    """

    def __init__(self, schedule):
        self.device = "cpu"
        self.config = SimpleNamespace(use_cache=False)
        self.schedule = schedule
        self.calls = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        spec = self.schedule[min(len(self.calls), len(self.schedule) - 1)]
        self.calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
        })
        B = input_ids.shape[0]
        if callable(spec):
            col = torch.tensor([[spec(r)] for r in range(B)], dtype=torch.long)
        else:
            col = torch.full((B, 1), spec, dtype=torch.long)
        return torch.cat([input_ids, col], dim=1)


class FakeWikiSearcher:
    def search(self, query, top_k=3, sentences=3):
        return "OBSERVATION: fake observation text"


def make_env(schedule=None):
    if schedule is None:
        schedule = [SEARCH_MARK, ANSWER_MARK]
    return FakeModel(schedule), FakeTokenizer(), FakeWikiSearcher()


class TestRowOrdering:
    def test_sample_major_rows(self):
        """row = s*G + g: all G completions of sample 0 first."""
        model, tok, wiki = make_env()
        prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
        all_turns, all_texts = batched_generate_multi_prompt(
            model, tok, prompts, wiki, num_generations=2, max_turns=3,
        )

        assert len(all_turns) == 4 and len(all_texts) == 4
        # Turn 0's input_ids are exactly the prompt of the owning sample.
        for g in range(2):
            assert all_turns[g][0]['input_ids'] == [1, 2, 3]
            assert all_turns[2 + g][0]['input_ids'] == [4, 5, 6, 7, 8]

    def test_turn_state_machine(self):
        """SEARCH turn gets query+observation; ANSWER turn ends the row."""
        model, tok, wiki = make_env()
        prompts = [[1, 2, 3]]
        all_turns, all_texts = batched_generate_multi_prompt(
            model, tok, prompts, wiki, num_generations=2, max_turns=3,
        )

        for turns in all_turns:
            assert len(turns) == 2
            assert turns[0]['query'] == "fake query"
            assert "OBSERVATION" in turns[0]['observation']
            assert turns[1]['query'] is None  # ANSWER turn: no search
        assert all_texts[0] == all_texts[1]  # deterministic fake decode


class TestLeftPadding:
    def test_padding_and_attention_mask(self):
        model, tok, wiki = make_env()
        prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]  # ragged lengths
        batched_generate_multi_prompt(
            model, tok, prompts, wiki, num_generations=2, max_turns=3,
        )

        first = model.calls[0]  # turn 0: rows are the raw prompts
        ids, mask = first["input_ids"], first["attention_mask"]
        # maxlen = len([4,5,6,7,8]) = 5: short rows pad on the LEFT.
        assert ids.shape == (4, 5) and mask.shape == (4, 5)
        # Short prompt rows: 2 pad tokens on the left, then [1,2,3].
        assert ids[0].tolist() == [0, 0, 1, 2, 3]
        assert mask[0].tolist() == [0, 0, 1, 1, 1]
        # Long prompt rows: no padding, full attention.
        assert ids[2].tolist() == [4, 5, 6, 7, 8]
        assert mask[2].tolist() == [1, 1, 1, 1, 1]
        # tokenizer padding_side is restored afterwards
        assert tok.padding_side == "right"

    def test_second_turn_batches_only_active_rows(self):
        """Rows that ANSWERed drop out; SEARCH rows keep generating."""
        model, tok, wiki = make_env(
            schedule=[lambda r: ANSWER_MARK if r % 2 else SEARCH_MARK,
                      ANSWER_MARK],
        )
        prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
        all_turns, _ = batched_generate_multi_prompt(
            model, tok, prompts, wiki, num_generations=2, max_turns=3,
        )

        # Odd rows finished on turn 0 (ANSWER), even rows on turn 1.
        for r, turns in enumerate(all_turns):
            assert len(turns) == (1 if r % 2 else 2)
        # Turn-1 call only contains the rows that searched.
        assert model.calls[1]["input_ids"].shape[0] == 2


class TestEdgeCases:
    def test_row_without_action_stops(self):
        model, tok, wiki = make_env(schedule=[THOUGHT_MARK])
        prompts = [[1, 2, 3]]
        all_turns, _ = batched_generate_multi_prompt(
            model, tok, prompts, wiki, num_generations=2, max_turns=3,
        )
        for turns in all_turns:
            assert len(turns) == 1
            assert turns[0]['query'] is None
            assert turns[0]['observation'] is None

    def test_empty_prompt_list(self):
        model, tok, wiki = make_env()
        all_turns, all_texts = batched_generate_multi_prompt(
            model, tok, [], wiki, num_generations=2,
        )
        assert all_turns == [] and all_texts == []

    def test_single_prompt_wrapper(self):
        """batched_generate_with_search stays a P=1 pass-through."""
        model, tok, wiki = make_env()
        all_turns, all_texts = batched_generate_with_search(
            model, tok, [9, 9, 9], wiki, num_generations=3, max_turns=3,
        )
        assert len(all_turns) == 3 and len(all_texts) == 3
        for turns in all_turns:
            assert turns[0]['input_ids'] == [9, 9, 9]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
