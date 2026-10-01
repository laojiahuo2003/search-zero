"""
Tests for PPO-style inner-loop updates (run_phase3_updates) in
train_grpo_search_MI300X.py.

Exercised WITHOUT a GPU using a tiny real parameterised model. Pins down:

  1. mu=1 reproduces the historical single-pass Phase 3 EXACTLY
     (same loss value, same parameter update as the old inline loop)
  2. mu=N performs exactly N optimizer steps per data-step
  3. old_logprobs stay frozen across inner passes (the clip's reference)
  4. ratio deviates from 1 and the clip engages on inner pass >= 2
     (the whole point of the feature; mu=1 reports zero stats)
  5. loss scaling is invariant to how completions are split into micros
  6. empty completions are skipped without NaN/crash
  7. CW-GRPO rule mode runs through the inner loop and keeps gradients sane

Run:  .venv/bin/python -m pytest tests/test_inner_updates.py -v
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from scripts.train_grpo_search_MI300X import (
    run_phase3_updates,
    compute_batch_logprobs,
    compute_turn_logprobs,
    grpo_loss,
)
from scripts.credit_assignment import CreditConfig, build_token_advantages


class TinyModel(nn.Module):
    """Real parameterised model (embed + proj) with the model.device /
    model(ids, attention_mask) interface the logprob path expects."""

    def __init__(self, vocab=12, hidden=8):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)
        self.device = torch.device("cpu")

    def forward(self, ids, attention_mask=None):
        return SimpleNamespace(logits=self.proj(self.embed(ids)))


def _make_turn(input_ids, gen_ids, old_logprobs=None):
    if old_logprobs is None:
        old_logprobs = torch.full((len(gen_ids),), -0.5)
    return {
        'input_ids': list(input_ids),
        'gen_ids': list(gen_ids),
        'old_logprobs': old_logprobs,
        'query': None,
        'observation': None,
        'credit': None,
    }


def _make_group():
    """One sample's group: G=2 completions with distinct (input, gen)."""
    return [
        [_make_turn([1, 2, 3], [4, 5, 6])],
        [_make_turn([1, 2, 3], [5, 6, 7, 8]),
         _make_turn([1, 2, 3, 5, 6, 7, 8, 9], [10, 11])],
    ]


def _make_step_data():
    """2 samples x G=2 completions, arranged as ONE micro. Deterministic."""
    return [[
        {'turns': _make_group(),
         'advantages': torch.tensor([1.0, -1.0])},
        {'turns': _make_group(),
         'advantages': torch.tensor([0.5, -0.5])},
    ]]


def _make_model_and_optimizer(lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = TinyModel()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    return model, params, optimizer


def _reference_phase3(model, optimizer, trainable_params, step_data,
                      total_completions, credit_cfg, beta, eps_low, eps_high,
                      lr):
    """Reference semantics for ONE Phase 3 pass, computed with a separate
    forward graph per completion (the train_grpo_search.py baseline style:
    compute_all_logprobs per completion, then per-completion backward).
    run_phase3_updates(mu=1) must match this exactly, which simultaneously
    validates the sum-then-backward-once fix: a group's completions share
    ONE packed-forward graph, so per-completion sequential backward() would
    re-traverse freed saved tensors and crash."""
    optimizer.zero_grad()
    batch_loss = 0.0
    for micro_samples in step_data:
        for sample in micro_samples:
            group_turns = sample['turns']
            advantages = sample['advantages']
            for g in range(len(group_turns)):
                # one completion per call -> independent autograd graph
                new_lps, old_lps = compute_batch_logprobs(
                    model, [group_turns[g]])[0]
                if len(new_lps) == 0 or len(old_lps) == 0:
                    continue
                if credit_cfg.mode != "none":
                    adv = build_token_advantages(
                        group_turns[g], float(advantages[g]), credit_cfg
                    ).to(model.device)
                else:
                    adv = advantages[g]
                loss = grpo_loss(new_lps, old_lps, adv, beta=beta,
                                 epsilon_low=eps_low, epsilon_high=eps_high)
                loss = loss / total_completions
                loss.backward()
                batch_loss += loss.item() * total_completions
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    optimizer.step()
    return batch_loss


def test_mu1_matches_reference_single_pass():
    """mu=1 must reproduce the old inline Phase 3 exactly: identical loss
    value and identical parameter update."""
    step_data_a, step_data_b = _make_step_data(), _make_step_data()
    total = 4
    cfg = CreditConfig(mode="none")

    model_a, params_a, opt_a = _make_model_and_optimizer()
    loss_a, stats_a = run_phase3_updates(
        model_a, opt_a, params_a, step_data_a, total,
        num_inner_updates=1, credit_cfg=cfg,
        beta=0.04, epsilon_low=0.2, epsilon_high=0.28, lr=1e-3)

    model_b, params_b, opt_b = _make_model_and_optimizer()
    loss_b = _reference_phase3(model_b, opt_b, params_b, step_data_b, total,
                               cfg, 0.04, 0.2, 0.28, 1e-3)

    assert abs(loss_a - loss_b) < 1e-6, f"loss {loss_a} vs {loss_b}"
    assert stats_a == {'ratio_dev': 0.0, 'clip_frac': 0.0}
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-7), \
            f"param drift {(pa - pb).abs().max()}"


def test_mu_optimizer_step_count():
    """mu=N performs exactly N optimizer.step() and N zero_grad() calls."""
    model, params, optimizer = _make_model_and_optimizer()
    calls = {'step': 0, 'zero': 0}
    orig_step, orig_zero = optimizer.step, optimizer.zero_grad

    def spy_step(*a, **k):
        calls['step'] += 1
        return orig_step(*a, **k)

    def spy_zero(*a, **k):
        calls['zero'] += 1
        return orig_zero(*a, **k)

    optimizer.step, optimizer.zero_grad = spy_step, spy_zero

    init = [p.detach().clone() for p in model.parameters()]
    run_phase3_updates(model, optimizer, params, _make_step_data(), 4,
                       num_inner_updates=3, credit_cfg=CreditConfig(),
                       beta=0.04, epsilon_low=0.2, epsilon_high=0.28, lr=1e-3)

    assert calls == {'step': 3, 'zero': 3}
    # three real updates must have moved the parameters
    assert any(not torch.allclose(p, q) for p, q
               in zip(model.parameters(), init))


def test_old_logprobs_frozen_across_inner_passes():
    """old_logprobs are the clip's frozen reference: the inner loop must
    never touch them (values AND device unchanged after mu=2)."""
    step_data = _make_step_data()
    before = [[t['old_logprobs'].clone() for t in sample['turns'][0]]
              for micro in step_data for sample in micro]

    model, params, optimizer = _make_model_and_optimizer()
    run_phase3_updates(model, optimizer, params, step_data, 4,
                       num_inner_updates=2, credit_cfg=CreditConfig(),
                       beta=0.04, epsilon_low=0.2, epsilon_high=0.28, lr=1e-3)

    after = [[t['old_logprobs'] for t in sample['turns'][0]]
             for micro in step_data for sample in micro]
    for row_b, row_a in zip(before, after):
        for tb, ta in zip(row_b, row_a):
            assert torch.equal(tb, ta)
            assert ta.device.type == "cpu"


def test_clip_engages_on_later_inner_passes():
    """The feature's raison d'etre: on inner pass >= 2 the policy has moved
    away from the frozen old_logprobs, so ratio != 1 (ratio_dev > 0) and —
    with tight epsilons — the clip actually fires (clip_frac > 0)."""
    model, params, optimizer = _make_model_and_optimizer()
    _, stats = run_phase3_updates(
        model, optimizer, params, _make_step_data(), 4,
        num_inner_updates=2, credit_cfg=CreditConfig(),
        beta=0.0, epsilon_low=0.01, epsilon_high=0.01, lr=1.0)

    assert stats['ratio_dev'] > 0.0
    assert stats['clip_frac'] > 0.0


def test_micro_split_invariance():
    """Same completions, arranged as 1 micro vs 2 micros, with the same
    total_completions: identical loss and identical gradients (the loss
    scaling is invariant to the micro split)."""
    one_micro = _make_step_data()              # [[s1, s2]]
    two_micros = [[one_micro[0][0]], [one_micro[0][1]]]  # [[s1], [s2]]
    cfg = CreditConfig(mode="none")

    model_a, params_a, opt_a = _make_model_and_optimizer()
    loss_a, _ = run_phase3_updates(model_a, opt_a, params_a, one_micro, 4,
                                   num_inner_updates=1, credit_cfg=cfg,
                                   beta=0.04, epsilon_low=0.2,
                                   epsilon_high=0.28, lr=1e-3)

    model_b, params_b, opt_b = _make_model_and_optimizer()
    loss_b, _ = run_phase3_updates(model_b, opt_b, params_b, two_micros, 4,
                                   num_inner_updates=1, credit_cfg=cfg,
                                   beta=0.04, epsilon_low=0.2,
                                   epsilon_high=0.28, lr=1e-3)

    assert abs(loss_a - loss_b) < 1e-6
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-7)


def test_empty_completions_skipped():
    """A completion that produced no tokens (empty gen everywhere) is
    skipped; the remaining completions still produce finite loss and grads."""
    group = _make_group()
    group[0] = [_make_turn([1, 2, 3], [], torch.tensor([]))]  # empty
    step_data = [[{'turns': group, 'advantages': torch.tensor([1.0, -1.0])}]]

    model, params, optimizer = _make_model_and_optimizer()
    loss, _ = run_phase3_updates(model, optimizer, params, step_data, 2,
                                 num_inner_updates=2, credit_cfg=CreditConfig(),
                                 beta=0.04, epsilon_low=0.2,
                                 epsilon_high=0.28, lr=1e-3)

    assert loss == loss  # not NaN
    assert model.proj.weight.grad is not None
    assert torch.isfinite(model.proj.weight.grad).all()


def test_credit_rule_mode_inner_loop():
    """CW-GRPO rule mode flows through the inner loop: per-token advantages
    are rebuilt every pass, loss stays finite, gradients flow, and the
    credit-conserving property (mean of non-final weights == 1) holds."""
    cfg = CreditConfig(mode="rule")
    step_data = _make_step_data()
    # Simulate Phase 2.5: fill credits on non-final turns of the 2-turn
    # completion (index 1 of each group).
    for sample in step_data[0]:
        sample['turns'][1][0]['credit'] = 1.0
        sample['turns'][1][1]['credit'] = 1.0

    model, params, optimizer = _make_model_and_optimizer()
    loss, stats = run_phase3_updates(
        model, optimizer, params, step_data, 4,
        num_inner_updates=2, credit_cfg=cfg,
        beta=0.04, epsilon_low=0.2, epsilon_high=0.28, lr=1e-2)

    assert loss == loss
    assert stats['ratio_dev'] >= 0.0
    assert model.proj.weight.grad is not None
    assert torch.isfinite(model.proj.weight.grad).all()

    # Credit conservation sanity: positive-advantage 2-turn trajectory with
    # credit 1.0 on the non-final turn -> non-final tokens get adv * 1
    # (single non-final turn, mean-1 normalised), final turn keeps adv.
    turns = _make_group()[1]
    turns[0]['credit'] = 1.0
    tok_adv = build_token_advantages(turns, 0.5, cfg)
    n0, n1 = len(turns[0]['gen_ids']), len(turns[1]['gen_ids'])
    assert tok_adv.shape == (n0 + n1,)
    assert torch.allclose(tok_adv[:n0], torch.full((n0,), 0.5))
    assert torch.allclose(tok_adv[n0:], torch.full((n1,), 0.5))
