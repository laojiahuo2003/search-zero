"""
GRPO Training with Real Wikipedia Search — Search-R1
=====================================================
Custom training loop: multi-turn ReAct generation + real Wikipedia retrieval + GRPO loss.

Key design decisions:
  - Raw text concatenation with Qwen2.5 chat markers (not apply_chat_template for every turn)
    This keeps token IDs deterministic between generation and loss computation.
  - Incremental forward passes for both old and new logprobs → perfect alignment.
  - Wikipedia search with disk cache to minimize API latency.
  - NUM_INNER_UPDATES (mu) PPO-style inner updates per rollout batch with
    frozen old_logprobs: mu=1 reproduces the original REINFORCE-equivalent
    behaviour (ratio == 1, clip never engages); mu>1 makes the clip real.

Usage (on A100):
  /data/miniconda/envs/torch/bin/python scripts/train_grpo_search.py
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

# CRITICAL: Prevent OOM from memory fragmentation during multi-turn forward passes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiki_search import CachedWikiSearcher, LocalWikiSearcher
from credit_assignment import (
    CreditConfig, get_credit_config, build_token_advantages,
    rule_judge_turn, LLMTurnJudge,
)

# SwanLab tracking (optional — degrades to a no-op when unconfigured)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.config import get_config
from app.utils.tracking import init_tracking, log_metrics, finish_tracking

_cfg = get_config()

# ============================================
# Configuration
# ============================================
# ============================================
# Credit Assignment (CW-GRPO)
# ============================================
# Per-turn contribution weights reallocate the trajectory-level GRPO
# advantage across turns (see credit_assignment.py). Configured via env
# vars, also settable in .env:
#   CREDIT_MODE=none|rule|llm         (default none -> vanilla GRPO)
#   CREDIT_GAMMA=1.0                  softmax inverse temperature (>=10 hard)
#   CREDIT_JUDGE_WORKERS=16           parallel LLM judge calls
#   CREDIT_RULE_MIN_NEW_WORDS=3       rule judge novelty threshold
#   CREDIT_RULE_QUERY_SIM=0.8         rule judge repeat-query Jaccard
#   CREDIT_FALLBACK_UNIFORM=1         uniform weights when all credits are 0
#   CREDIT_JUDGE_ONLY_POSITIVE_ADV=1  skip judging adv <= 0 trajectories
CREDIT_CFG = get_credit_config()

# All paths derive from SEARCH_ZERO_ROOT (see app/utils/config.py). Unset,
# they resolve inside the repo, which is the historical layout.
MODEL_PATH = _cfg.base_model
SFT_CHECKPOINT = _cfg.sft_checkpoint
OUTPUT_DIR = _cfg.grpo_output_dir
WIKI_CACHE_DIR = _cfg.wiki_cache_dir
HOTPOTQA_PATH = _cfg.hotpotqa_train_path

NUM_EPOCHS = 2
# Samples per micro-batch. With BATCHED_GENERATION on, the P samples x G
# completions are left-padded into ONE generate() call (P x G sequences);
# the group-normalized advantage is still computed per sample.
#
# Sizing on this MI300X (launch-bound: GFX-Uti 100% / Mem-Uti 8%, ~84us per
# kernel, ~700 launches per forward):
#   * P drives speed. An epoch issues (MAX_TURNS x NUM_SAMPLES / P) generate()
#     calls and a call's cost is nearly row-independent while launch-bound, so
#     wall-clock ~ 1/P. P=16 with G=4 packs 64 rows (~11GB of KV cache) — 4x
#     the launch amortisation of the previous P=8 x G=2.
#   * GA trades update count for gradient noise and cancels out of the
#     generation cost (fewer steps, more micros per step). GA=1 keeps
#     M = P = 16 -> 31 steps/epoch; GA=4 would halve that to 15.
#   * Phase 3 peak VRAM does NOT grow with GA or P: backward() runs per
#     sample, so only one group's autograd graph is alive at a time. It does
#     grow with G — one group is G x MAX_TURNS turn rows, packed at
#     PACKED_MAX_ROWS and all alive until that sample's backward.
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5.0e-7
WARMUP_RATIO = 0.1
NUM_GENERATIONS = 4  # G=2 collapses the group-normalized advantage to a
# sign-only +/-0.707; G>=4 restores magnitude information.
TEMPERATURE = 0.9
BETA = 0.04
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
MAX_TURNS = 3                    # Reduced from 3: SEARCH → ANSWER (still allows multi-hop)
MAX_TOKENS_PER_TURN = 256        # Slightly reduced from 300 for speed
MAX_COMPLETION_TOKENS = 3072  # 3 turns × 256 + obs overhead
SAVE_STEPS = 500
LOG_STEPS = 1
NUM_SAMPLES = 500

# Inner policy updates per rollout batch (PPO-style epochs over the collected
# data, old_logprobs frozen). mu=1 is pure on-policy REINFORCE: old and new
# logprobs come from the same parameters, ratio == 1, so the clip never
# engages and the KL term is exactly zero. With mu>1 the rollout data
# (turns / old_logprobs / advantages / credits) is collected ONCE per step
# and Phase 3 runs mu times over it; from the 2nd inner pass the parameters
# have moved, ratio deviates from 1 and the asymmetric clip
# (epsilon_low/high) plus the KL penalty actually constrain the update.
# Generation — the expensive phase — still happens only once; each extra pass
# costs one packed-logprob + backward sweep over the step's completions.
# mu does NOT raise peak VRAM: the inner passes are sequential, so pass u's
# autograd graphs are freed before pass u+1 starts.
# Env-overridable for A/B runs: GRPO_INNER_UPDATES=1 uv run python ...
NUM_INNER_UPDATES = int(os.environ.get("GRPO_INNER_UPDATES", "2"))

# Per-completion backward: process one completion at a time in Phase 3 instead
# of batching all completions of a group. Trades deduplication for memory:
# peak drops from G×MAX_TURNS (12 turn rows, ~128GB activations) to MAX_TURNS
# (3 turn rows, ~32GB activations) — a 75% reduction. Slightly slower because
# it loses cross-completion turn deduplication (e.g. if 2 completions share
# the same SEARCH turn, they now forward it twice). Set to "1" to enable when
# Phase 3 OOMs with the default batched path.
# Env-overridable: PER_COMPLETION_BACKWARD=1 python ...
PER_COMPLETION_BACKWARD = os.environ.get("PER_COMPLETION_BACKWARD", "0") == "1"

# Pack ALL sequences of one micro-batch (PER_DEVICE_BATCH_SIZE samples x
# NUM_GENERATIONS completions) into a single batched generate() call.
# On a launch-bound GPU (measured: GFX-Uti 100% but Mem-Uti 8%, ~84us/kernel)
# this amortises fixed kernel-launch overhead across P x G sequences. Output
# is mathematically identical to per-sample batched calls: each row is an
# independent trajectory, group normalization is per sample. Set False to
# A/B test against the sequential path.
BATCHED_GENERATION = True

# Attention backend. On ROCm, flash_attention_2's Composable Kernel kernel is
# frequently SLOWER than PyTorch's native SDPA for single-stream decode
# (observed ~10x). "sdpa" is the recommended default; try "flash_attention_2"
# only if you benchmark it faster on your stack.
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")

# Rows per packed forward in the Phase 1 / Phase 3 logprob passes.
#
# The binding constraint is NOT the logits tensor ((rows, maxlen, vocab) bf16,
# ~700MB per row at 2.3k tokens) — it is the per-row autograd graph that
# Phase 3's backward needs, roughly ~11GB per turn row at 2.3k tokens for a
# 7B (28 layers x ~170KB of saved activations per token, SDPA so no
# materialised attention matrix). Measured on the MI300X: the whole run sits
# at 182GB of 196GB with this at 4, i.e. ~14GB of headroom, so each extra row
# costs about as much as the headroom left. Raise it only together with
# gradient checkpointing or shorter turns.
PACKED_MAX_ROWS = 4

# Qwen2.5 chat template markers (hardcoded for deterministic tokenization)
CHAT_MARKERS = {
    "end": "<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "asst_start": "<|im_start|>assistant\n",
    "sys_start": "<|im_start|>system\n",
    "eos": "<|endoftext|>",
}

GRPO_SYSTEM_PROMPT = """You are a research AI agent that uses the ReAct framework to answer questions.
You have access to a Wikipedia search tool.

For each step, follow this format EXACTLY:

THOUGHT: <your reasoning about what to do next>
ACTION: SEARCH: <search query>

When you search, you will receive OBSERVATION with real Wikipedia results.
You may search multiple times to gather enough information.

When you are ready to answer, output:

THOUGHT: <final reasoning>
ACTION: ANSWER: <your answer with source citations like [1], [2]>

Important: Always cite your sources. Always end with ACTION: ANSWER:"""


# ============================================================
# Reward Functions
# ============================================================
# 提取答案
def extract_answer(text: str) -> str:
    """Extract ANSWER content from generated text."""
    m = re.search(r'ACTION\s*:\s*ANSWER\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT)|$)',
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'ANSWER\s*:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

# 格式分数
def format_reward(text: str) -> float:
    """ReAct format compliance: THOUGHT +0.3, ACTION +0.3, ANSWER +0.5."""
    score = 0.0
    if re.search(r'THOUGHT\s*:', text, re.IGNORECASE):
        score += 0.3
    if re.search(r'ACTION\s*:', text, re.IGNORECASE):
        score += 0.3
    if re.search(r'ANSWER\s*:', text, re.IGNORECASE):
        score += 0.5
    return min(1.0, score)

# 答案奖励
def accuracy_reward(text: str, ground_truth: str) -> float:
    """Continuous QA reward: contains=0.7, EM=1.0, partial word overlap=0.1-0.5.

    Avoids F1's short-answer penalty. Reward is still continuous enough
    for good advantage signal in GRPO.
    """
    answer = extract_answer(text)
    if not answer or not ground_truth:
        return 0.0

    ans_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)

    # 1.0 = exact match EM 精确匹配
    if ans_norm == gt_norm:
        return 1.0

    # 0.7 = gold answer contained in prediction (longer prediction) GT 包含于答案（答案更长更全）
    if gt_norm and gt_norm in ans_norm:
        return 0.7

    # 0.5 = prediction contained in gold (partial but right direction 答案包含于 GT（方向对但不全）
    if ans_norm and ans_norm in gt_norm:
        return 0.5

    # 0.1-0.4 = token overlap ratio (continuous fallback)词重叠
    gt_words = set(ground_truth.lower().split())
    ans_words = set(answer.lower().split())
    if gt_words and ans_words:
        overlap = len(gt_words & ans_words)
        score = min(0.4, overlap / max(len(gt_words), 1) * 2)
        return round(score, 2)

    return 0.0


def normalize_answer(text: str) -> str:
    """Normalize text for comparison."""
    import re as _re
    text = text.lower().strip()
    text = _re.sub(r'\s+', ' ', text)
    text = _re.sub(r'[^\w\s]', '', text)
    return text


# ============================================================
# Token Utilities
# ============================================================

def encode_marker(tokenizer, marker_key: str) -> list:
    """Encode a chat template marker."""
    return tokenizer.encode(CHAT_MARKERS[marker_key], add_special_tokens=False)


def tokenize_observation(tokenizer, obs_text: str) -> list:
    """Tokenize an observation message with Qwen2.5 chat markers."""
    ids = []
    ids += encode_marker(tokenizer, "end")         # end previous assistant
    ids += encode_marker(tokenizer, "user_start")   # start user message
    ids += tokenizer.encode(obs_text, add_special_tokens=False)
    ids += encode_marker(tokenizer, "end")          # end user message
    ids += encode_marker(tokenizer, "asst_start")   # start next assistant
    return ids


def make_prompt_ids(tokenizer, system_prompt: str, question: str) -> list:
    """Build initial prompt with Qwen2.5 chat template."""
    # <|im_start|>system\n{text}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    ids = []
    ids += encode_marker(tokenizer, "sys_start")
    ids += tokenizer.encode(system_prompt, add_special_tokens=False)
    ids += encode_marker(tokenizer, "end")
    ids += encode_marker(tokenizer, "user_start")
    ids += tokenizer.encode(question, add_special_tokens=False)
    ids += encode_marker(tokenizer, "end")
    ids += encode_marker(tokenizer, "asst_start")
    return ids


# ============================================================
# Generation with Wikipedia Search
# ============================================================

def generate_with_search(model, tokenizer, prompt_ids, wiki_searcher,
                         max_turns=3, max_tokens_per_turn=300, temperature=0.9,
                         compute_logprobs=False):
    """Multi-turn generation with real Wikipedia search.

    Args:
        prompt_ids: list of token IDs for the initial prompt (includes asst_start)
        wiki_searcher: CachedWikiSearcher instance
        compute_logprobs: if True, compute logprobs during generation (old_logprobs)

    Returns:
        turns: list of dicts, each with:
            'input_ids': list of token IDs (model input for this turn)
            'gen_ids': list of token IDs (model output for this turn)
            'text': str (decoded model output)
            'old_logprobs': tensor (if compute_logprobs=True)
        all_gen_text: concatenated model-generated text (for reward)
    """
    turns = []
    current_ids = list(prompt_ids)  # accumulates across turns
    model_gen_texts = []

    # Timing stats
    time_gen = 0.0
    time_search = 0.0
    time_logprob = 0.0

    for turn_idx in range(max_turns):
        input_tensor = torch.tensor([current_ids], device=model.device)

        # Truncate if too long (keep last MAX_PROMPT_LENGTH tokens)
        # MAX_PROMPT_LENGTH = max prompt context
        if input_tensor.shape[1] > 2048:
            # Keep system prompt + recent context
            input_tensor = input_tensor[:, -2048:]
            current_ids = input_tensor[0].tolist()

        # Generate
        t0 = time.time()
        with torch.no_grad():
            # Rollout MUST use the KV cache. main() disables use_cache for the
            # training backward pass, but leaving it off here makes every decode
            # step recompute attention over the whole prefix (O(n^2) per turn).
            # Toggle it on for generation, then restore it.
            prev_cache = model.config.use_cache
            model.config.use_cache = True
            try:
                output = model.generate(
                    input_tensor,
                    max_new_tokens=max_tokens_per_turn,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            finally:
                model.config.use_cache = prev_cache
        time_gen += time.time() - t0

        # Extract generated tokens
        gen_ids = output[0, input_tensor.shape[1]:].tolist()
        if not gen_ids:
            break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        turn_data = {
            'input_ids': list(current_ids),
            'gen_ids': gen_ids,
            'text': gen_text,
            'query': None,          # SEARCH query of this turn (if any)
            'observation': None,    # retrieval result of this turn (if any)
            'credit': None,         # CW-GRPO contribution weight (filled in Phase 2.5)
        }

        # Compute logprobs immediately after generation (for old_logprobs)
        if compute_logprobs:
            t0 = time.time()
            with torch.no_grad():
                old_lps = compute_turn_logprobs(model, current_ids, gen_ids)
                turn_data['old_logprobs'] = old_lps.cpu()  # Move to CPU to save VRAM
            time_logprob += time.time() - t0

        turns.append(turn_data)
        model_gen_texts.append(gen_text)

        # Check for ANSWER
        if re.search(r'ACTION\s*:\s*ANSWER\s*:', gen_text, re.IGNORECASE):
            break

        # Extract SEARCH query
        search_match = re.search(
            r'ACTION\s*:\s*SEARCH\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT|$)|$)',
            gen_text, re.IGNORECASE | re.DOTALL
        )
        if search_match:
            query = search_match.group(1).strip()
            if query:
                t0 = time.time()
                obs_text = wiki_searcher.search(query, top_k=3, sentences=3)
                time_search += time.time() - t0
                # Keep query + observation for CW-GRPO credit assignment
                turn_data['query'] = query
                turn_data['observation'] = obs_text
                obs_ids = tokenize_observation(tokenizer, obs_text)
                current_ids = current_ids + gen_ids + obs_ids
                continue

        # No SEARCH, no ANSWER — stop
        break

    all_gen_text = "\n".join(model_gen_texts)

    # Store timing info for debugging
    if hasattr(turns, '__timing__'):
        turns.__timing__ = {'gen': time_gen, 'search': time_search, 'logprob': time_logprob}

    return turns, all_gen_text


def batched_generate_multi_prompt(model, tokenizer, prompt_ids_list,
                                  wiki_searcher, num_generations=2,
                                  max_turns=3, max_tokens_per_turn=256,
                                  temperature=0.9, compute_logprobs=False):
    """Batched multi-turn generation for P prompts x G completions.

    Generalises the single-prompt batched path to MULTIPLE prompts: the
    P x G sequences are left-padded into ONE batched generate() call per
    turn. On a bandwidth-underutilised GPU (at batch=1 every one of the
    ~700 kernels in a forward pass processes one token, so the fixed
    launch overhead dominates — measured ~84us/kernel on ROCm), this
    amortises that overhead across P x G rows instead of G without
    changing the maths: each row is an independent trajectory, and the
    per-sequence output is identical to calling the single-prompt
    batched_generate_with_search() once per prompt.

    Left-padding is used so sequences of different lengths share a batch.
    HF generate() handles cache positions correctly for left-padded inputs
    as long as an explicit attention_mask is passed.

    Row ordering contract (sample-major, CRITICAL for regrouping):
        row = s * num_generations + g   for sample s in [0, P), completion g
    i.e. all G completions of sample 0 occupy the first G rows, then
    sample 1, and so on.

    Args:
        prompt_ids_list: list of P prompt token-id lists (one per sample)
        num_generations: G completions to sample per prompt

    Returns:
        all_turns: list of length P x G; all_turns[s*G + g] is the turn-dict
            list for the g-th completion of the s-th prompt (same schema as
            generate_with_search())
        all_texts: list of length P x G of joined generation text
    """
    P = len(prompt_ids_list)
    if P == 0:
        return [], []
    B = P * num_generations
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Sample-major row order: rows s*G .. s*G+G-1 hold the G completions of
    # prompt s. Different prompts never share a row.
    seq_ids = [list(prompt_ids_list[s])
               for s in range(P) for _ in range(num_generations)]
    finished = [False] * B
    all_turns = [[] for _ in range(B)]
    all_texts = [[] for _ in range(B)]

    # Left padding so batched generation is correct for ragged lengths.
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for _turn_idx in range(max_turns):
            active = [i for i in range(B) if not finished[i]]
            if not active:
                break
            # (turn_data, input_ids, gen_ids) collected by the row loop for
            # one packed old-logprob pass after it.
            old_lp_specs = []

            seqs = [seq_ids[i] for i in active]

            # Same guard as the sequential path: cap context so a long
            # observation cannot push the sequence past the truncation limit.
            MAX_PROMPT_LENGTH = 2048
            for i, s in zip(active, seqs):
                if len(s) > MAX_PROMPT_LENGTH:
                    seq_ids[i] = s[-MAX_PROMPT_LENGTH:]
            seqs = [seq_ids[i] for i in active]

            maxlen = max(len(s) for s in seqs)

            input_ids = torch.full((len(active), maxlen), pad_id,
                                   dtype=torch.long, device=model.device)
            attn_mask = torch.zeros((len(active), maxlen),
                                    dtype=torch.long, device=model.device)
            for r, s in enumerate(seqs):
                input_ids[r, maxlen - len(s):] = torch.tensor(s, device=model.device)
                attn_mask[r, maxlen - len(s):] = 1

            prev_cache = model.config.use_cache
            model.config.use_cache = True
            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_tokens_per_turn,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=pad_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            finally:
                model.config.use_cache = prev_cache

            for r, i in enumerate(active):
                gen_ids = output[r, maxlen:].tolist()

                # A row that stopped early is padded out to the longest row in
                # the batch. Truncate at the first EOS rather than stripping
                # trailing pads: Qwen's pad_token_id == eos_token_id, so
                # stripping pads would also discard a real EOS token and make
                # this path disagree with the sequential one. Rows that ran to
                # max_new_tokens contain no EOS and are left untouched.
                eos_id = tokenizer.eos_token_id
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]

                if not gen_ids:
                    finished[i] = True
                    continue

                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                turn_data = {
                    'input_ids': list(seq_ids[i]),
                    'gen_ids': gen_ids,
                    'text': gen_text,
                    'query': None,          # SEARCH query of this turn (if any)
                    'observation': None,    # retrieval result of this turn (if any)
                    'credit': None,         # CW-GRPO contribution weight (filled in Phase 2.5)
                }

                if compute_logprobs:
                    # Collected here; computed in ONE packed pass after the
                    # row loop to amortise per-forward launch overhead.
                    old_lp_specs.append((turn_data, list(seq_ids[i]), gen_ids))

                all_turns[i].append(turn_data)
                all_texts[i].append(gen_text)

                # ANSWER ends the trajectory.
                if re.search(r'ACTION\s*:\s*ANSWER\s*:', gen_text, re.IGNORECASE):
                    finished[i] = True
                    continue

                # SEARCH appends an observation and continues.
                search_match = re.search(
                    r'ACTION\s*:\s*SEARCH\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT|$)|$)',
                    gen_text, re.IGNORECASE | re.DOTALL
                )
                if search_match:
                    query = search_match.group(1).strip()
                    if query:
                        obs_text = wiki_searcher.search(query, top_k=3, sentences=3)
                        # Keep query + observation for CW-GRPO credit assignment
                        turn_data['query'] = query
                        turn_data['observation'] = obs_text
                        obs_ids = tokenize_observation(tokenizer, obs_text)
                        seq_ids[i] = seq_ids[i] + gen_ids + obs_ids
                        continue

                # Neither SEARCH nor ANSWER — stop this sequence.
                finished[i] = True

            # One packed pass computes old_logprobs for every row of this
            # turn (identical maths to the per-row path, fewer forward calls).
            if compute_logprobs and old_lp_specs:
                with torch.no_grad():
                    packed_old = batched_compute_turn_logprobs(
                        model,
                        [(inp, gen) for _, inp, gen in old_lp_specs],
                    )
                for (turn_data, _, _), old_lps in zip(old_lp_specs, packed_old):
                    turn_data['old_logprobs'] = old_lps.cpu()
    finally:
        tokenizer.padding_side = prev_padding_side

    return all_turns, ["\n".join(t) for t in all_texts]


def batched_generate_with_search(model, tokenizer, prompt_ids, wiki_searcher,
                                 num_generations=2, max_turns=3,
                                 max_tokens_per_turn=256, temperature=0.9,
                                 compute_logprobs=False):
    """Batched multi-turn generation for G completions of the SAME prompt.

    Thin wrapper over batched_generate_multi_prompt() with a single prompt:
    identical behaviour and return shape (lists of length num_generations).
    """
    all_turns, all_texts = batched_generate_multi_prompt(
        model, tokenizer, [prompt_ids], wiki_searcher,
        num_generations=num_generations,
        max_turns=max_turns,
        max_tokens_per_turn=max_tokens_per_turn,
        temperature=temperature,
        compute_logprobs=compute_logprobs,
    )
    return all_turns, all_texts


# ============================================================
# Logprob Computation (aligned with generation)
# ============================================================

def compute_turn_logprobs(model, input_ids, gen_ids):
    """Compute per-token logprobs for generated tokens, using given input context.

    Forward pass: model(input + gen) → logits
    Logprobs for gen tokens at positions [L_in, L_in+L_gen-1] come from logits at [L_in-1, L_in+L_gen-2]
    """
    full_ids = torch.tensor([input_ids + gen_ids], device=model.device)
    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    L_in = len(input_ids)
    L_gen = len(gen_ids)

    gen_logits = logits[L_in - 1 : L_in + L_gen - 1]  # (L_gen, V)
    gen_targets = torch.tensor(gen_ids, device=model.device).unsqueeze(1)
    # Only the target token's logprob is needed, so skip the full-vocab
    # log_softmax: log_softmax(x)[i] == x[i] - logsumexp(x). This drops two
    # full-vocab kernels per turn row and a (L_gen, vocab) float32 tensor that
    # backward would otherwise have to keep alive for log_softmax's own
    # gradient (~40MB at the observed L_gen=68, up to ~156MB at the 256 cap —
    # small next to the ~11GB activation graph per row, but free). Verified
    # numerically identical forward and backward (rel diff ~1e-6, four orders
    # below bf16's own eps).
    token_logprobs = (
        gen_logits.gather(1, gen_targets).squeeze(1).float()
        - torch.logsumexp(gen_logits.float(), dim=-1)
    )  # (L_gen,)

    return token_logprobs


def batched_compute_turn_logprobs(model, turn_specs, max_batch_rows=PACKED_MAX_ROWS):
    """Compute per-token logprobs for many (input_ids, gen_ids) pairs via
    packed left-padded forward passes.

    Mathematically identical to calling compute_turn_logprobs() once per
    pair — the causal shift uses row-local indices (row r's generated
    tokens live in columns [maxlen-L_gen[r]-1, maxlen-1)) and the
    attention mask excludes padding — but up to max_batch_rows rows share
    ONE forward pass, amortising the per-forward kernel launch overhead
    (~700 kernels) across more rows. The row cap bounds the
    (rows, seq, vocab) logits tensor (~700MB/row at 2.3k tokens for a
    152k vocab).

    Gradients flow through the returned logprobs when the model requires
    them (used for new_logprobs in Phase 3).

    Args:
        turn_specs: list of (input_ids, gen_ids) lists
        max_batch_rows: max rows per packed forward (logits memory cap)

    Returns:
        list of (L_gen,) logprob tensors aligned with turn_specs order
    """
    results = []
    for start in range(0, len(turn_specs), max_batch_rows):
        results.extend(_packed_turn_logprob_forward(
            model, turn_specs[start:start + max_batch_rows]))
    return results


def _packed_turn_logprob_forward(model, batch):
    """One packed forward pass over a list of (input_ids, gen_ids) pairs.

    Left-padding + attention mask; per-row extraction mirrors
    compute_turn_logprobs() with row-local causal shift.
    """
    rows = len(batch)
    if rows == 0:
        return []
    full_ids_list = [list(inp) + list(gen) for inp, gen in batch]
    lens_gen = [len(gen) for _, gen in batch]
    maxlen = max(len(f) for f in full_ids_list)

    input_ids = torch.zeros((rows, maxlen), dtype=torch.long,
                            device=model.device)
    attn_mask = torch.zeros((rows, maxlen), dtype=torch.long,
                            device=model.device)
    for r, f in enumerate(full_ids_list):
        input_ids[r, maxlen - len(f):] = torch.tensor(f, device=model.device)
        attn_mask[r, maxlen - len(f):] = 1

    outputs = model(input_ids, attention_mask=attn_mask)
    logits = outputs.logits  # (rows, maxlen, V)

    row_results = []
    for r in range(rows):
        L_gen = lens_gen[r]
        # The token generated at position p of row r lives in column
        # maxlen-L_gen+p; its probability comes from the logits of the
        # PREVIOUS column, i.e. columns [maxlen-L_gen-1, maxlen-1). For an
        # unpadded row maxlen == L_in+L_gen, so this is exactly the
        # sequential path's logits[L_in-1 : L_in+L_gen-1].
        gen_logits = logits[r, maxlen - L_gen - 1 : maxlen - 1]  # (L_gen, V)
        gen_targets = torch.tensor(batch[r][1],
                                   device=model.device).unsqueeze(1)
        # Same substitution as compute_turn_logprobs: target logit minus
        # logsumexp instead of a full-vocab log_softmax.
        row_results.append(
            gen_logits.gather(1, gen_targets).squeeze(1).float()
            - torch.logsumexp(gen_logits.float(), dim=-1)
        )

    return row_results


def compute_batch_logprobs(model, all_turns_list):
    """Batch compute logprobs for multiple completions to reduce forward passes.

    Args:
        all_turns_list: list of turns (each element is a list of turn dicts)

    Returns:
        list of (new_lps, old_lps) tuples, one per completion
    """
    results = []

    # Collect all unique (input_ids, gen_ids) pairs to avoid redundant computation
    unique_turns = []
    turn_to_idx = {}

    for turns in all_turns_list:
        completion_indices = []
        for turn in turns:
            # Create hashable key
            key = (tuple(turn['input_ids']), tuple(turn['gen_ids']))
            if key not in turn_to_idx:
                turn_to_idx[key] = len(unique_turns)
                unique_turns.append(turn)
            completion_indices.append(turn_to_idx[key])
        results.append(completion_indices)

    # Batch compute all unique turns: the deduplicated (input_ids, gen_ids)
    # pairs are left-padded into packed forward passes (gradients preserved
    # for the Phase 3 backward pass).
    all_new_lps = batched_compute_turn_logprobs(
        model,
        [(turn['input_ids'], turn['gen_ids']) for turn in unique_turns],
    )

    # Reconstruct per-completion logprobs
    final_results = []
    for completion_idx, turns in enumerate(all_turns_list):
        indices = results[completion_idx]
        new_lps = torch.cat([all_new_lps[idx] for idx in indices]) if indices else torch.tensor([], device=model.device)
        old_lps = torch.cat([turns[i]['old_logprobs'].to(model.device) for i in range(len(turns)) if 'old_logprobs' in turns[i]]) if turns else torch.tensor([], device=model.device)
        final_results.append((new_lps, old_lps))

    return final_results


# ============================================================
# GRPO Loss
# ============================================================

def grpo_loss(per_token_logps, old_per_token_logps, advantages,
              beta=0.04, epsilon_low=0.2, epsilon_high=0.28):
    """Compute GRPO policy gradient loss (token-level, no mask needed).

    Args:
        per_token_logps: (T,) current model logprobs
        old_per_token_logps: (T,) old logprobs (detached)
        advantages: scalar advantage for this completion, or a (T,) tensor of
            per-token advantages (CW-GRPO credit reallocation broadcasts it
            elementwise, matching the per-token logprobs)
        beta: KL penalty
        epsilon_low/high: clipping
    """
    log_ratio = per_token_logps - old_per_token_logps
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    if beta > 0:
        per_token_kl = torch.exp(old_per_token_logps - per_token_logps) - \
                       (old_per_token_logps - per_token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl

    return per_token_loss.mean()


def run_phase3_updates(model, optimizer, trainable_params, step_data,
                       total_completions, num_inner_updates, credit_cfg,
                       beta, epsilon_low, epsilon_high, lr):
    """Phase 3: consume one step's collected rollout data with
    `num_inner_updates` optimizer steps (PPO-style inner loop).

    The rollout data (turns / frozen old_logprobs / advantages / credits) is
    collected ONCE per step by Phases 1-2.5; this function replays it
    `num_inner_updates` times. Inner pass 1 is on-policy w.r.t. the frozen
    old_logprobs (ratio == 1, mu=1 reproduces the historical behaviour
    exactly); from pass 2 on, the parameters have moved, ratio deviates
    from 1 and the asymmetric clip actually constrains the update.

    Each inner pass accumulates gradients over ALL micro-batches (each
    completion's loss is divided by total_completions, so a pass's gradient
    is the average over the step's completions), then clips, applies the
    shared scheduled LR and steps the optimizer.

    Args:
        step_data: list over micro-batches; each element is a list over
            samples of {'turns': group_turns, 'advantages': (G,) tensor}
        total_completions: completions generated in this step (all micros);
            the per-completion loss divisor (see main loop comment)
        num_inner_updates: mu, number of optimizer steps over the same data
        lr: scheduled learning rate applied to every inner pass of this step

    Returns:
        (mean_batch_loss, stats): mean_batch_loss averages the per-pass
            batch loss over inner passes (identical to the old value when
            mu=1); stats holds off-policy diagnostics averaged over inner
            passes 2..mu ('ratio_dev' = mean |ratio - 1|, 'clip_frac' =
            fraction of clipped tokens); both 0.0 when mu == 1.
    """
    total_loss = 0.0
    ratio_devs, clip_fracs = [], []

    for u in range(num_inner_updates):
        optimizer.zero_grad()
        for micro_samples in step_data:
            for sample in micro_samples:
                group_turns = sample['turns']
                advantages = sample['advantages']

                if PER_COMPLETION_BACKWARD:
                    # Per-completion backward: process one completion at a time
                    # to keep only MAX_TURNS turn rows alive (peak ~32GB
                    # activations instead of G×MAX_TURNS ~128GB). Loses
                    # cross-completion turn deduplication, so slightly slower.
                    for g in range(len(group_turns)):
                        batch_logprobs = compute_batch_logprobs(model, [group_turns[g]])
                        new_lps, old_lps = batch_logprobs[0]

                        if len(new_lps) == 0 or len(old_lps) == 0:
                            continue

                        if credit_cfg.mode != "none":
                            adv = build_token_advantages(
                                group_turns[g], float(advantages[g]), credit_cfg
                            ).to(model.device)
                        else:
                            adv = advantages[g]

                        loss = grpo_loss(new_lps, old_lps, adv,
                                         beta=beta, epsilon_low=epsilon_low,
                                         epsilon_high=epsilon_high)
                        loss = loss / total_completions

                        # Backward immediately to free this completion's activations
                        loss.backward()

                        total_loss += loss.item() * total_completions

                        if u > 0:
                            with torch.no_grad():
                                ratio = torch.exp(new_lps - old_lps)
                                ratio_devs.append((ratio - 1).abs().mean().item())
                                clip_fracs.append(
                                    ((ratio < 1 - epsilon_low) |
                                     (ratio > 1 + epsilon_high)
                                     ).float().mean().item())
                else:
                    # Batch compute all logprobs at once (reduces forward passes)
                    batch_logprobs = compute_batch_logprobs(model, group_turns)

                    # Sum the group's per-completion losses and backward ONCE.
                    # The packed forward shares ONE autograd graph across all
                    # completions of the group, so per-completion sequential
                    # backward() would re-traverse freed saved tensors
                    # (RuntimeError: backward through the graph a second time).
                    # Summing first is mathematically identical (gradients add)
                    # and the peak memory is the same: the shared activations
                    # are alive until the last backward either way.
                    sample_loss = None
                    for g in range(len(group_turns)):
                        new_lps, old_lps = batch_logprobs[g]

                        if len(new_lps) == 0 or len(old_lps) == 0:
                            continue

                        if credit_cfg.mode != "none":
                            # CW-GRPO: reallocate the trajectory advantage across
                            # turns. Token order matches the concatenated
                            # per-turn logprobs in new_lps.
                            adv = build_token_advantages(
                                group_turns[g], float(advantages[g]), credit_cfg
                            ).to(model.device)
                        else:
                            adv = advantages[g]
                        loss = grpo_loss(new_lps, old_lps, adv,
                                         beta=beta, epsilon_low=epsilon_low,
                                         epsilon_high=epsilon_high)
                        # Divide by the completions in the WHOLE step (all
                        # micros) so a pass's gradient is the average over
                        # its completions, invariant to PER_DEVICE_BATCH_SIZE
                        # and GRADIENT_ACCUMULATION_STEPS.
                        loss = loss / total_completions
                        sample_loss = loss if sample_loss is None else sample_loss + loss

                        total_loss += loss.item() * total_completions

                        if u > 0:
                            # Off-policy diagnostics: only inner passes 2..mu can
                            # deviate from the frozen old_logprobs (pass 1 is
                            # on-policy by construction).
                            with torch.no_grad():
                                ratio = torch.exp(new_lps - old_lps)
                                ratio_devs.append((ratio - 1).abs().mean().item())
                                clip_fracs.append(
                                    ((ratio < 1 - epsilon_low) |
                                     (ratio > 1 + epsilon_high)
                                     ).float().mean().item())

                    if sample_loss is not None:
                        sample_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

    stats = {
        'ratio_dev': sum(ratio_devs) / len(ratio_devs) if ratio_devs else 0.0,
        'clip_frac': sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0,
    }
    return total_loss / num_inner_updates, stats


# ============================================================
# Data Loading
# ============================================================

def load_hotpotqa_data(data_path: str, num_samples: int = 500):
    """Load HotpotQA and return HuggingFace Dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    gts = []
    for item in data[:num_samples]:
        q, a = item.get("question", ""), item.get("answer", "")
        if q and a:
            prompts.append({"system": GRPO_SYSTEM_PROMPT, "question": q})
            gts.append(a)

    dataset = Dataset.from_dict({"prompt": prompts, "ground_truth": gts})
    print(f"Loaded {len(dataset)} HotpotQA samples")
    return dataset


def preprocess_dataset(dataset, tokenizer):
    """Pre-tokenize all prompts to avoid repeated tokenization."""
    def tokenize_fn(example):
        system_prompt = example['prompt']['system']
        question = example['prompt']['question']
        prompt_ids = make_prompt_ids(tokenizer, system_prompt, question)
        return {
            'prompt': example['prompt'],
            'prompt_ids': prompt_ids,
            'ground_truth': example['ground_truth'],
        }

    print("  Pre-tokenizing prompts...")
    dataset = dataset.map(tokenize_fn, batched=False, desc="Tokenizing")
    return dataset


# ============================================================
# Training
# ============================================================

def main():
    print("=" * 60)
    print("  Search-R1 GRPO — Real Wikipedia Search")
    print(f"  Model: {MODEL_PATH}")
    print(f"  SFT LoRA: {SFT_CHECKPOINT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU: {gpu_name} ({vram:.1f} GB)")

    # ---- 1. Tokenizer ----
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Model ----
    print("[2/6] Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        print(f"  ✓ Attention backend: {ATTN_IMPLEMENTATION}")
    except Exception as e:
        print(f"  ✗ {ATTN_IMPLEMENTATION} not available: {e}")
        print("  → Falling back to eager attention")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            attn_implementation="eager",
        )

    # ---- 3. LoRA ----
    print("[3/6] Loading SFT LoRA...")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    for n, p in model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
    # enable_input_require_grads is REQUIRED for GRPO — without it,
    # per-token logprobs don't require grad and loss.backward() fails.
    # It makes embedding output require gradients, which cascades through
    # the model. This uses significant VRAM (~30+ GB for 7B model), so we
    # must keep sequences short and turns minimal.
    model.enable_input_require_grads()
    # Disable gradient checkpointing — it conflicts with the hook above
    if model.config.use_cache:
        model.config.use_cache = False
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.1f}M")

    # ---- 4. Wikipedia Search ----
    print("[4/6] Initializing Wikipedia searcher...")
    # Try local index first (no network needed, works behind GFW)
    WIKI_INDEX_PATH = _cfg.wiki_index_path
    if os.path.exists(WIKI_INDEX_PATH):
        wiki = LocalWikiSearcher(WIKI_INDEX_PATH)
        print(f"  Using LOCAL search: {wiki.get_stats()['articles']} articles")
    else:
        print("  Local index not found, trying live Wikipedia...")
        wiki = CachedWikiSearcher(cache_dir=WIKI_CACHE_DIR)
        print(f"  Using LIVE Wikipedia (may be blocked in China)")

    # Quick test
    print("  Testing search...")
    try:
        r = wiki.search("Python programming", top_k=1, sentences=1)
        ok = "OBSERVATION" in r
        print(f"  Search: {'OK' if ok else 'FAIL'}")
        if ok:
            print(f"  {r[:150]}...")
    except Exception as e:
        print(f"  Search test error: {e}")

    # ---- 5. Data ----
    print("[5/6] Loading HotpotQA data...")
    dataset = load_hotpotqa_data(HOTPOTQA_PATH, num_samples=NUM_SAMPLES)
    dataset = preprocess_dataset(dataset, tokenizer)

    # ---- 6. Optimizer ----
    print("[6/6] Setting up optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    total_steps = len(dataset) * NUM_EPOCHS // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def get_lr(step):
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup; start at a small positive value so the first
            # optimizer step is not wasted at lr=0.
            return LEARNING_RATE * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LEARNING_RATE * 0.5 * (1 + math.cos(progress * math.pi))

    print(f"  Steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"  Batch: {PER_DEVICE_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Inner updates per step (mu): {NUM_INNER_UPDATES}")
    if PER_COMPLETION_BACKWARD:
        print(f"  Per-completion backward: ENABLED (peak {MAX_TURNS} turn rows, -75% memory)")
    else:
        print(f"  Batched backward: ENABLED (peak {NUM_GENERATIONS}×{MAX_TURNS} turn rows)")

    # ============================================================
    # Training Loop
    # ============================================================
    swanlab_run = init_tracking(
        name=os.path.basename(OUTPUT_DIR.rstrip("/")) or "grpo",
        config={
            "model": MODEL_PATH,
            "sft_checkpoint": SFT_CHECKPOINT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": PER_DEVICE_BATCH_SIZE,
            "grad_accum": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "num_generations": NUM_GENERATIONS,
            "temperature": TEMPERATURE,
            "beta": BETA,
            "epsilon_low": EPSILON_LOW,
            "epsilon_high": EPSILON_HIGH,
            "num_inner_updates": NUM_INNER_UPDATES,
            "per_completion_backward": PER_COMPLETION_BACKWARD,
            "max_turns": MAX_TURNS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "num_samples": NUM_SAMPLES,
            "total_steps": total_steps,
            "credit_mode": CREDIT_CFG.mode,
            "credit_gamma": CREDIT_CFG.gamma,
        },
        tags=["grpo", "search-r1", "hotpotqa"],
    )

    # ---- Credit judge (CW-GRPO) ----
    credit_judge = LLMTurnJudge() if CREDIT_CFG.mode == "llm" else None
    if CREDIT_CFG.mode == "llm":
        print(f"  Credit assignment: LLM judge ({credit_judge.model})")
    elif CREDIT_CFG.mode == "rule":
        print("  Credit assignment: deterministic rule-based judge")

    print("\n" + "=" * 60)
    print("  Starting training")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        dataset = dataset.shuffle(seed=42 + epoch)

        # One optimizer step processes GRADIENT_ACCUMULATION_STEPS micro-batches
        # of PER_DEVICE_BATCH_SIZE samples each.
        step_size = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

        for batch_start in range(0, len(dataset), step_size):
            all_rewards, all_fmt, all_acc, all_lengths = [], [], [], []
            all_credit_ret, all_credit_thk = [], []  # judge stats for CW-GRPO
            n_micro = 0
            # Completions actually generated in this step (P x G per micro).
            # Each completion's loss is divided by this so a step's gradient is
            # the average over all its completions, invariant to
            # PER_DEVICE_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS and to a
            # ragged tail micro-batch.
            total_completions = 0
            # Rollout data collected by Phases 1/2/2.5, consumed by Phase 3
            # (NUM_INNER_UPDATES times, with frozen old_logprobs).
            # Layout: step_data[micro][sample] = {'turns', 'advantages'}
            step_data = []

            for micro in range(GRADIENT_ACCUMULATION_STEPS):
                sample_start = batch_start + micro * PER_DEVICE_BATCH_SIZE
                if sample_start >= len(dataset):
                    break
                n_micro += 1
                batch = dataset[sample_start:sample_start + PER_DEVICE_BATCH_SIZE]

                # ---- Gather this micro's prompts/questions/golds ----
                micro_prompt_ids, micro_questions, micro_gts = [], [], []
                for sample_idx in range(len(batch['prompt'])):
                    item = batch['prompt'][sample_idx]
                    # question is needed by Phase 2.5 (credit judge), keep it
                    # defined regardless of which prompt path is taken.
                    question = item['question']
                    micro_questions.append(question)
                    micro_gts.append(batch['ground_truth'][sample_idx])

                    # Use pre-tokenized prompt_ids if available
                    if 'prompt_ids' in batch and batch['prompt_ids'][sample_idx]:
                        micro_prompt_ids.append(batch['prompt_ids'][sample_idx])
                    else:
                        micro_prompt_ids.append(
                            make_prompt_ids(tokenizer, item['system'], question)
                        )
                n_samples = len(micro_prompt_ids)

                # ============================================================
                # Phase 1: GENERATION — n_samples x G completions in ONE batch
                # ============================================================
                # All n_samples x G sequences of this micro are left-padded into
                # one generate() call per turn; row = s*G + g, regrouped below.
                # Each row is an independent trajectory, so outputs equal
                # per-sample batched calls. A launch-bound GPU amortises its
                # ~84us/kernel overhead across n_samples x G rows here.
                t0_gen = time.time()
                micro_turns = []  # [sample][completion] = turn-dict list
                micro_texts = []  # [sample][completion] = joined text

                if BATCHED_GENERATION:
                    flat_turns, flat_texts = batched_generate_multi_prompt(
                        model, tokenizer, micro_prompt_ids, wiki,
                        num_generations=NUM_GENERATIONS,
                        max_turns=MAX_TURNS,
                        max_tokens_per_turn=MAX_TOKENS_PER_TURN,
                        temperature=TEMPERATURE,
                        compute_logprobs=True,
                    )
                    for s in range(n_samples):
                        base = s * NUM_GENERATIONS
                        micro_turns.append([flat_turns[base + g]
                                            for g in range(NUM_GENERATIONS)])
                        micro_texts.append([flat_texts[base + g]
                                            for g in range(NUM_GENERATIONS)])
                else:
                    # A/B baseline: per-sample, per-completion sequential gen.
                    for s in range(n_samples):
                        gturns, gtexts = [], []
                        for g in range(NUM_GENERATIONS):
                            turns, full_text = generate_with_search(
                                model, tokenizer, micro_prompt_ids[s], wiki,
                                max_turns=MAX_TURNS,
                                max_tokens_per_turn=MAX_TOKENS_PER_TURN,
                                temperature=TEMPERATURE,
                                compute_logprobs=True,  # old_logprobs during generation
                            )
                            gturns.append(turns)
                            gtexts.append(full_text)
                        micro_turns.append(gturns)
                        micro_texts.append(gtexts)

                total_completions += n_samples * NUM_GENERATIONS
                if global_step <= 2:
                    print(f"  Batched gen ({n_samples}x{NUM_GENERATIONS} seqs): "
                          f"{time.time() - t0_gen:.2f}s "
                          f"turns={[len(t) for ts in micro_turns for t in ts]}")

                # Phases 2/2.5 run per sample during collection (group
                # normalization is per-sample by definition); Phase 3 runs
                # after ALL micros are collected, NUM_INNER_UPDATES times.
                # Timers accumulate across samples, printed once per micro.
                t_reward_acc = t_credit_acc = 0.0
                micro_data = []
                for s in range(n_samples):
                    group_turns = micro_turns[s]
                    group_texts = micro_texts[s]
                    gt = micro_gts[s]
                    question = micro_questions[s]

                    for g in range(NUM_GENERATIONS):
                        total_tokens = sum(len(t['gen_ids']) for t in group_turns[g])
                        all_lengths.append(total_tokens)

                    # ============================================================
                    # Phase 2: REWARD — Compute rewards and advantages
                    # ============================================================
                    t0_reward = time.time()
                    group_rewards = []
                    group_fmt = []
                    group_acc = []
                    for g in range(NUM_GENERATIONS):
                        fmt_r = format_reward(group_texts[g])
                        acc_r = accuracy_reward(group_texts[g], gt)
                        group_rewards.append(fmt_r + acc_r)
                        group_fmt.append(fmt_r)
                        group_acc.append(acc_r)

                    # Group-normalized advantages
                    # Note: upstream CW-GRPO normalizes over answer_reward
                    # (binary EM) only; we keep search-zero's (format +
                    # accuracy) advantage — format is a first-class training
                    # signal here and the reallocation in Phase 2.5/3 does
                    # not depend on which signal produced the advantage.
                    rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std()
                    advantages = (rewards_t - mean_r) / (std_r + 1e-4)

                    all_rewards.extend(group_rewards)
                    all_fmt.extend(group_fmt)
                    all_acc.extend(group_acc)

                    t_reward_acc += time.time() - t0_reward

                    # ============================================================
                    # Phase 2.5: CREDIT — per-turn contribution weights (CW-GRPO)
                    # ============================================================
                    # Official CW-GRPO: judge every non-final search turn with a
                    # binary (retrieval x thinking) score, normalize within the
                    # trajectory, then reallocate the trajectory advantage across
                    # turns. Negative-advantage trajectories keep the broadcast
                    # advantage unchanged, so they are not judged at all.
                    if CREDIT_CFG.mode != "none":
                        t0_credit = time.time()
                        credit_jobs = []  # (g, turn_idx) awaiting the LLM judge
                        for g in range(NUM_GENERATIONS):
                            turns = group_turns[g]
                            if len(turns) < 2:
                                continue  # single turn: nothing to reallocate
                            if advantages[g] <= 0 and CREDIT_CFG.judge_only_positive_adv:
                                continue  # never reallocated -> skip the judge
                            for idx in range(len(turns) - 1):
                                if not turns[idx].get('query'):
                                    # No SEARCH in this turn: nothing to credit.
                                    turns[idx]['credit'] = 0.0
                                    continue
                                if CREDIT_CFG.mode == "rule":
                                    ret, thk = rule_judge_turn(
                                        question, gt, turns, idx, CREDIT_CFG
                                    )
                                    turns[idx]['credit'] = float(ret * thk)
                                    all_credit_ret.append(ret)
                                    all_credit_thk.append(thk)
                                else:
                                    credit_jobs.append((g, idx))
                        if credit_jobs:
                            results = credit_judge.judge_many(
                                [(question, group_turns[g], idx) for g, idx in credit_jobs],
                                workers=CREDIT_CFG.judge_workers,
                            )
                            for (g, idx), (ret, thk) in zip(credit_jobs, results):
                                group_turns[g][idx]['credit'] = float(ret * thk)
                                all_credit_ret.append(ret)
                                all_credit_thk.append(thk)
                        t_credit_acc += time.time() - t0_credit

                    # Stash for Phase 3 (runs after all micros, mu times)
                    micro_data.append({'turns': group_turns,
                                       'advantages': advantages})

                step_data.append(micro_data)
                if global_step <= 2:
                    print(f"  Reward computation: {t_reward_acc:.2f}s")
                    if CREDIT_CFG.mode != "none":
                        print(f"  Credit assignment: {t_credit_acc:.2f}s")

            if n_micro == 0:
                break

            # ============================================================
            # Phase 3: LEARNING — mu inner updates over the frozen rollout
            # ============================================================
            # old_logprobs stay frozen from Phase 1, so inner pass 2+ sees
            # ratio != 1 and the clip does real work. All inner passes of a
            # data-step share the same scheduled LR, so mu=1 reproduces the
            # historical behaviour exactly (same backward order, same loss
            # scaling, same LR schedule).
            t0_phase3 = time.time()
            current_lr = get_lr(global_step)
            batch_loss, update_stats = run_phase3_updates(
                model, optimizer, trainable_params, step_data,
                total_completions, NUM_INNER_UPDATES, CREDIT_CFG,
                BETA, EPSILON_LOW, EPSILON_HIGH, current_lr)
            if global_step <= 2:
                print(f"  Phase 3 (mu={NUM_INNER_UPDATES}): "
                      f"{time.time() - t0_phase3:.2f}s")

            global_step += 1

            # Logging
            if global_step % LOG_STEPS == 0:
                n = max(len(all_rewards), 1)
                metrics = {
                    "loss": batch_loss,
                    "reward": sum(all_rewards) / n,
                    "reward_format": sum(all_fmt) / n,
                    "reward_accuracy": sum(all_acc) / n,
                    "completion_len": sum(all_lengths) / n,
                    "lr": current_lr,
                    "epoch": epoch + 1,
                }
                if all_credit_ret:
                    metrics["credit_retrieval_mean"] = sum(all_credit_ret) / len(all_credit_ret)
                    metrics["credit_thinking_mean"] = sum(all_credit_thk) / len(all_credit_thk)
                # Off-policy diagnostics of the inner loop (mu>1 only):
                # ratio_dev grows as inner passes drift from the frozen
                # old_logprobs; clip_frac > 0 is direct evidence the
                # asymmetric clip is now constraining updates.
                if NUM_INNER_UPDATES > 1:
                    metrics["ratio_dev"] = update_stats['ratio_dev']
                    metrics["clip_frac"] = update_stats['clip_frac']
                inner_info = (f" rdev={update_stats['ratio_dev']:.4f}"
                              f" clip={update_stats['clip_frac']:.3f}"
                              if NUM_INNER_UPDATES > 1 else "")
                print(f"[Step {global_step}/{total_steps}] "
                      f"loss={batch_loss:.4f} "
                      f"rew={sum(all_rewards)/n:.3f} "
                      f"fmt={sum(all_fmt)/n:.3f} "
                      f"acc={sum(all_acc)/n:.3f} "
                      f"len={sum(all_lengths)/n:.0f} "
                      f"lr={current_lr:.2e}" + inner_info)
                log_metrics(swanlab_run, metrics, step=global_step)

            # Checkpoint
            if global_step % SAVE_STEPS == 0:
                ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  Saved: {ckpt}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # Save
    print("\n" + "=" * 60)
    print("  Saving final model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  {OUTPUT_DIR}")
    print(f"  Wiki cache: {wiki.get_stats()}")
    print("=" * 60)

    finish_tracking(swanlab_run)


if __name__ == "__main__":
    main()
