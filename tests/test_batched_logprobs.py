"""
Tests for packed batched logprob computation (train_grpo_search.py).

Exercised WITHOUT a GPU using fake models. Pins down:

  1. packed vs sequential equivalence (exact per-token logprobs)
  2. left padding + attention mask assembly
  3. batch splitting by max_batch_rows
  4. gradient flow through the packed forward (Phase 3 needs it)
  5. gradient equivalence packed vs sequential (same parameter values)
  6. Phase 1 integration: old_logprobs back-filled into turn_data
  7. Phase 3 integration: dedup + per-completion reconstruction

Run:  .venv/bin/python -m pytest tests/test_batched_logprobs.py -v
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from scripts.train_grpo_search_MI300X import (
    batched_compute_turn_logprobs,
    compute_turn_logprobs,
    compute_batch_logprobs,
    batched_generate_multi_prompt,
)


class PositionalLogitsModel:
    """Fake model whose logits depend ONLY on position (and vocab index),
    never on batch, row or total length.

    Positions are derived from the attention mask exactly like HF
    left-padded forward passes (cumsum - 1, pad positions zeroed). Because
    logits are a unique function of position, any column-index bug in the
    packed path shifts the values and breaks the equivalence assertions.
    """

    def __init__(self, vocab=16):
        self.vocab = vocab
        self.device = torch.device("cpu")
        self.calls = []

    def __call__(self, ids, attention_mask=None):
        self.calls.append(
            (ids.clone(),
             None if attention_mask is None else attention_mask.clone())
        )
        B, L = ids.shape
        if attention_mask is None:
            pos = torch.arange(L).expand(B, -1)
        else:
            pos = attention_mask.cumsum(1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)
        vs = torch.arange(self.vocab, dtype=torch.float32) + 1
        # logits[b, t, v] = pos * (v + 1) * 0.5  — unique per (pos, v)
        logits = pos.unsqueeze(-1).float() * vs.unsqueeze(0) * 0.5
        return SimpleNamespace(logits=logits)


class TinyModel(nn.Module):
    """Real parameterised model for gradient-flow assertions."""

    def __init__(self, vocab=8, hidden=8):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)
        self.device = torch.device("cpu")

    def forward(self, ids, attention_mask=None):
        return SimpleNamespace(logits=self.proj(self.embed(ids)))


class FakeTokenizer:
    """decode() returns a THOUGHT with no ACTION -> every row stops after
    one turn (no search needed)."""

    pad_token_id = 0
    eos_token_id = 99

    def __init__(self):
        self.padding_side = "right"

    def decode(self, ids, skip_special_tokens=False):
        return "THOUGHT: only thinking"


class CombinedFakeModel(PositionalLogitsModel):
    """PositionalLogitsModel + deterministic generate() for Phase 1 tests."""

    def __init__(self, vocab=16):
        super().__init__(vocab)
        self.config = SimpleNamespace(use_cache=False)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        out = torch.zeros((B, L + 3), dtype=torch.long)
        out[:, :L] = input_ids
        out[:, L:] = torch.tensor([11, 12, 13])
        return out


def _rand_spec(rng, vocab, max_in=8, max_gen=5):
    inp = [int(rng.integers(1, vocab)) for _ in range(int(rng.integers(1, max_in + 1)))]
    gen = [int(rng.integers(1, vocab)) for _ in range(int(rng.integers(1, max_gen + 1)))]
    return inp, gen


def test_equivalence_packed_vs_sequential():
    """Packed results must equal per-row compute_turn_logprobs() exactly."""
    rng = np.random.default_rng(0)
    specs = [_rand_spec(rng, vocab=16) for _ in range(6)]
    specs.append(([1, 2, 3], []))  # empty gen edge case

    packed_model = PositionalLogitsModel(vocab=16)
    packed = batched_compute_turn_logprobs(packed_model, specs, max_batch_rows=2)

    seq_model = PositionalLogitsModel(vocab=16)
    sequential = [compute_turn_logprobs(seq_model, inp, gen)
                  for inp, gen in specs]

    assert len(packed) == len(specs)
    for p, s in zip(packed, sequential):
        assert p.shape == s.shape, f"shape {p.shape} vs {s.shape}"
        assert torch.allclose(p, s, atol=1e-6), f"max diff {(p - s).abs().max()}"

    # 7 specs with max_batch_rows=2 => 4 packed forwards (2/2/2/1)
    assert len(packed_model.calls) == 4


def test_padding_and_mask_assembly():
    """Short rows are left-padded with 0 and masked 0 on the pad side."""
    specs = [([1, 2, 3], [5, 6]), ([4, 5, 6, 7, 8], [9])]
    model = PositionalLogitsModel(vocab=16)
    batched_compute_turn_logprobs(model, specs, max_batch_rows=4)

    assert len(model.calls) == 1
    ids, mask = model.calls[0]
    assert ids.shape == (2, 6)          # maxlen = 5 + 1
    assert mask.shape == (2, 6)
    # row 0: full = [1,2,3,5,6] (len 5) -> one leading pad column
    assert torch.equal(ids[0], torch.tensor([0, 1, 2, 3, 5, 6]))
    assert torch.equal(mask[0], torch.tensor([0, 1, 1, 1, 1, 1]))
    # row 1: full = [4,5,6,7,8,9] len 6 -> no padding
    assert torch.equal(ids[1], torch.tensor([4, 5, 6, 7, 8, 9]))
    assert torch.equal(mask[1], torch.tensor([1, 1, 1, 1, 1, 1]))


def test_batch_splitting():
    """Rows beyond max_batch_rows spill into additional forwards, and the
    returned list stays aligned with the input order."""
    specs = [([1], [2])] * 10
    model = PositionalLogitsModel(vocab=16)
    packed = batched_compute_turn_logprobs(model, specs, max_batch_rows=4)

    assert len(packed) == 10
    assert len(model.calls) == 3          # 4 / 4 / 2
    for ids, _ in model.calls[:2]:
        assert ids.shape[0] == 4
    assert model.calls[2][0].shape[0] == 2
    # identical specs -> identical results, all in order
    for p in packed:
        assert torch.allclose(p, packed[0], atol=1e-6)


def test_empty_specs():
    model = PositionalLogitsModel(vocab=16)
    assert batched_compute_turn_logprobs(model, []) == []
    assert len(model.calls) == 0


def test_gradient_flow():
    """Packed forward must keep the graph (Phase 3 backprops through it)."""
    torch.manual_seed(0)
    model = TinyModel()
    specs = [([1, 2], [3, 4]), ([1, 2, 3, 4], [5])]
    packed = batched_compute_turn_logprobs(model, specs, max_batch_rows=4)

    total = sum(t.sum() for t in packed)
    total.backward()
    assert model.proj.weight.grad is not None
    assert model.proj.weight.grad.abs().sum() > 0
    assert model.embed.weight.grad is not None


def test_gradient_equivalence_with_sequential():
    """Parameter gradients from the packed path match the sequential path."""
    torch.manual_seed(0)
    packed_model = TinyModel()
    torch.manual_seed(0)
    seq_model = TinyModel()

    specs = [([1, 2], [3, 4]), ([1, 2, 3, 4], [5]), ([2, 1], [4, 3, 2])]
    packed = batched_compute_turn_logprobs(packed_model, specs, max_batch_rows=2)
    sum(t.sum() for t in packed).backward()

    sequential = [compute_turn_logprobs(seq_model, inp, gen)
                  for inp, gen in specs]
    sum(t.sum() for t in sequential).backward()

    for name, p_param in packed_model.named_parameters():
        s_param = dict(seq_model.named_parameters())[name]
        assert torch.allclose(p_param.grad, s_param.grad, atol=1e-6, rtol=1e-6), \
            f"grad mismatch on {name}: {(p_param.grad - s_param.grad).abs().max()}"


def test_phase1_old_logprobs_backfill():
    """batched_generate_multi_prompt back-fills old_logprobs into every
    turn_data via ONE packed pass, matching the sequential maths."""
    model = CombinedFakeModel(vocab=16)
    tokenizer = FakeTokenizer()
    prompts = [[1, 2, 3], [4, 5, 6, 7]]

    turns, texts = batched_generate_multi_prompt(
        model, tokenizer, prompts, None,
        num_generations=2, max_turns=3, max_tokens_per_turn=8,
        temperature=0.9, compute_logprobs=True,
    )

    # Exactly ONE forward went through the model: a single packed 4-row
    # pass (the generate() call does not hit __call__).
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[0] == 4

    assert len(turns) == 4
    expected_model = CombinedFakeModel(vocab=16)
    for i in range(4):
        assert len(turns[i]) == 1          # no ACTION -> one turn only
        turn = turns[i][0]
        assert turn['gen_ids'] == [11, 12, 13]
        assert turn['old_logprobs'] is not None
        expected = compute_turn_logprobs(
            expected_model, prompts[i // 2], [11, 12, 13])
        assert torch.allclose(turn['old_logprobs'], expected, atol=1e-6)


class MultiTurnTokenizer(FakeTokenizer):
    """turn 1 gen contains 11 -> SEARCH; turn 2 gen contains 14 -> ANSWER."""

    def __init__(self):
        super().__init__()

    def encode(self, text, add_special_tokens=False):
        return [501]

    def decode(self, ids, skip_special_tokens=False):
        if 11 in ids:
            return "THOUGHT: t\nACTION: SEARCH: fake query"
        if 14 in ids:
            return "THOUGHT: t\nACTION: ANSWER: fake answer"
        return "THOUGHT: only"


class MultiTurnModel(CombinedFakeModel):
    """First generate() call emits [11,12,13] (SEARCH), second [13,14,15]
    (ANSWER) — drives a two-turn trajectory per row."""

    def __init__(self, vocab=16):
        super().__init__(vocab)
        self._gen_call = 0

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self._gen_call += 1
        toks = [11, 12, 13] if self._gen_call == 1 else [13, 14, 15]
        B, L = input_ids.shape
        out = torch.zeros((B, L + 3), dtype=torch.long)
        out[:, :L] = input_ids
        out[:, L:] = torch.tensor(toks)
        return out


class FakeWikiSearcher:
    def search(self, query, top_k=3, sentences=3):
        return "fake observation text"


def test_phase1_multiturn_old_logprobs():
    """Two-turn trajectories (SEARCH then ANSWER): every turn of every row
    gets old_logprobs back-filled from packed passes that match the
    sequential maths, including the observation-augmented second turn."""
    model = MultiTurnModel(vocab=16)
    tokenizer = MultiTurnTokenizer()
    prompts = [[1, 2, 3], [4, 5, 6, 7]]

    turns, texts = batched_generate_multi_prompt(
        model, tokenizer, prompts, FakeWikiSearcher(),
        num_generations=2, max_turns=3, max_tokens_per_turn=8,
        temperature=0.9, compute_logprobs=True,
    )

    # Two packed logprob passes: one per turn (4 rows each).
    assert len(model.calls) == 2
    assert model.calls[0][0].shape[0] == 4
    assert model.calls[1][0].shape[0] == 4

    assert len(turns) == 4
    expected_model = MultiTurnModel(vocab=16)
    for i in range(4):
        assert len(turns[i]) == 2          # SEARCH -> ANSWER
        turn1, turn2 = turns[i][0], turns[i][1]

        assert turn1['gen_ids'] == [11, 12, 13]
        assert turn1['query'] == 'fake query'
        assert turn1['observation'] == 'fake observation text'
        exp1 = compute_turn_logprobs(
            expected_model, prompts[i // 2], [11, 12, 13])
        assert torch.allclose(turn1['old_logprobs'], exp1, atol=1e-6)

        assert turn2['gen_ids'] == [13, 14, 15]
        # second turn context = prompt + turn1 gen + observation tokens
        assert turn2['input_ids'] == prompts[i // 2] + [11, 12, 13] + [501] * 5
        exp2 = compute_turn_logprobs(
            expected_model, turn2['input_ids'], [13, 14, 15])
        assert torch.allclose(turn2['old_logprobs'], exp2, atol=1e-6)


def test_phase3_batch_splitting_many_unique():
    """More unique turns than max_batch_rows: packed forwards split into
    ceil(n/4) calls and per-completion reconstruction stays aligned."""
    model = PositionalLogitsModel(vocab=16)
    # 10 unique turns across 3 completions
    turns_a = [{'input_ids': [1], 'gen_ids': [2, 3],
                'old_logprobs': torch.tensor([-0.1, -0.2])}
               for _ in range(4)]
    turns_b = [{'input_ids': [1], 'gen_ids': [2, 3],
                'old_logprobs': torch.tensor([-0.1, -0.2])}
               for _ in range(3)]
    turns_c = [{'input_ids': [1], 'gen_ids': [2, 3],
                'old_logprobs': torch.tensor([-0.1, -0.2])}
               for _ in range(3)]
    # make every (input_ids, gen_ids) pair distinct (input_ids alone suffice)
    for k, t in enumerate(turns_a + turns_b + turns_c):
        t['input_ids'] = [k + 1]
        t['gen_ids'] = [2, 3]
        t['old_logprobs'] = torch.tensor([-float(k + 1), -float(k + 2)])
    all_turns_list = [turns_a, turns_b, turns_c]

    results = compute_batch_logprobs(model, all_turns_list)

    assert len(results) == 3
    # 10 unique turns -> ceil(10/4) = 3 packed forwards (4/4/2)
    assert len(model.calls) == 3
    assert model.calls[0][0].shape[0] == 4
    assert model.calls[1][0].shape[0] == 4
    assert model.calls[2][0].shape[0] == 2

    exp = [compute_turn_logprobs(model, t['input_ids'], t['gen_ids'])
           for t in turns_a + turns_b + turns_c]
    assert torch.allclose(
        results[0][0], torch.cat(exp[0:4]), atol=1e-6)
    assert torch.allclose(
        results[1][0], torch.cat(exp[4:7]), atol=1e-6)
    assert torch.allclose(
        results[2][0], torch.cat(exp[7:10]), atol=1e-6)


def test_phase3_dedup_and_reconstruction():
    """compute_batch_logprobs dedups identical turns, packs the unique ones
    into one forward, and reconstructs per-completion (new, old) pairs."""
    model = PositionalLogitsModel(vocab=16)
    t1 = {'input_ids': [1, 2, 3], 'gen_ids': [5, 6],
          'old_logprobs': torch.tensor([-0.1, -0.2])}
    t2 = {'input_ids': [1, 2, 3, 4, 5], 'gen_ids': [7],
          'old_logprobs': torch.tensor([-0.3])}
    t3 = {'input_ids': [9], 'gen_ids': [10, 11, 12],
          'old_logprobs': torch.tensor([-0.4, -0.5, -0.6])}
    all_turns_list = [[t1, t2], [t1, t3]]   # t1 duplicated across completions

    results = compute_batch_logprobs(model, all_turns_list)

    assert len(results) == 2
    # 3 unique turns -> ONE packed forward
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[0] == 3

    new0, old0 = results[0]
    new1, old1 = results[1]
    assert new0.shape == (3,)          # cat(len5->2, len7->1)
    assert new1.shape == (5,)          # cat(len5->2, len9->3)
    assert torch.allclose(old0, torch.tensor([-0.1, -0.2, -0.3]), atol=1e-6)
    assert torch.allclose(old1, torch.tensor([-0.1, -0.2, -0.4, -0.5, -0.6]),
                          atol=1e-6)

    # numeric equivalence of the new logprobs vs sequential computation
    exp_t1 = compute_turn_logprobs(model, t1['input_ids'], t1['gen_ids'])
    exp_t2 = compute_turn_logprobs(model, t2['input_ids'], t2['gen_ids'])
    exp_t3 = compute_turn_logprobs(model, t3['input_ids'], t3['gen_ids'])
    assert torch.allclose(new0, torch.cat([exp_t1, exp_t2]), atol=1e-6)
    assert torch.allclose(new1, torch.cat([exp_t1, exp_t3]), atol=1e-6)
