"""
GRPO Training with Real Wikipedia Search — Search-R1
=====================================================
Custom training loop: multi-turn ReAct generation + real Wikipedia retrieval + GRPO loss.

Key design decisions:
  - Raw text concatenation with Qwen2.5 chat markers (not apply_chat_template for every turn)
    This keeps token IDs deterministic between generation and loss computation.
  - Incremental forward passes for both old and new logprobs → perfect alignment.
  - Wikipedia search with disk cache to minimize API latency.

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

# SwanLab tracking (optional — degrades to a no-op when unconfigured)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.config import get_config
from app.utils.tracking import init_tracking, log_metrics, finish_tracking

_cfg = get_config()

# ============================================
# Configuration
# ============================================
# All paths derive from SEARCH_ZERO_ROOT (see app/utils/config.py). Unset,
# they resolve inside the repo, which is the historical layout.
MODEL_PATH = _cfg.base_model
SFT_CHECKPOINT = _cfg.sft_checkpoint
OUTPUT_DIR = _cfg.grpo_output_dir
WIKI_CACHE_DIR = _cfg.wiki_cache_dir
HOTPOTQA_PATH = _cfg.hotpotqa_train_path

NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5.0e-7
WARMUP_RATIO = 0.1
NUM_GENERATIONS = 4  # More completions for better advantage signal
TEMPERATURE = 0.9
BETA = 0.04
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
MAX_TURNS = 3                    # 3 turns: SEARCH → SEARCH → ANSWER (multi-hop)
MAX_TOKENS_PER_TURN = 300        # 300 tokens/turn, 80G VRAM plenty
MAX_COMPLETION_TOKENS = 3072  # 3 turns × 256 + obs overhead
SAVE_STEPS = 500
LOG_STEPS = 1
NUM_SAMPLES = 500

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

def extract_answer(text: str) -> str:
    """Extract ANSWER content from generated text."""
    m = re.search(r'ACTION\s*:\s*ANSWER\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT)|$)',
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'ANSWER\s*:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


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

    # 1.0 = exact match
    if ans_norm == gt_norm:
        return 1.0

    # 0.7 = gold answer contained in prediction (longer prediction)
    if gt_norm and gt_norm in ans_norm:
        return 0.7

    # 0.5 = prediction contained in gold (partial but right direction)
    if ans_norm and ans_norm in gt_norm:
        return 0.5

    # 0.1-0.4 = token overlap ratio (continuous fallback)
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
                         max_turns=3, max_tokens_per_turn=300, temperature=0.9):
    """Multi-turn generation with real Wikipedia search.

    Args:
        prompt_ids: list of token IDs for the initial prompt (includes asst_start)
        wiki_searcher: CachedWikiSearcher instance

    Returns:
        turns: list of dicts, each with:
            'input_ids': list of token IDs (model input for this turn)
            'gen_ids': list of token IDs (model output for this turn)
            'text': str (decoded model output)
        all_gen_text: concatenated model-generated text (for reward)
    """
    turns = []
    current_ids = list(prompt_ids)  # accumulates across turns
    model_gen_texts = []

    for turn_idx in range(max_turns):
        input_tensor = torch.tensor([current_ids], device=model.device)

        # Truncate if too long (keep last MAX_PROMPT_LENGTH tokens)
        # MAX_PROMPT_LENGTH = max prompt context
        if input_tensor.shape[1] > 2048:
            # Keep system prompt + recent context
            input_tensor = input_tensor[:, -2048:]
            current_ids = input_tensor[0].tolist()

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_tokens_per_turn,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract generated tokens
        gen_ids = output[0, input_tensor.shape[1]:].tolist()
        if not gen_ids:
            break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        turns.append({
            'input_ids': list(current_ids),
            'gen_ids': gen_ids,
            'text': gen_text,
        })
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
                obs_text = wiki_searcher.search(query, top_k=3, sentences=3)
                obs_ids = tokenize_observation(tokenizer, obs_text)
                current_ids = current_ids + gen_ids + obs_ids
                continue

        # No SEARCH, no ANSWER — stop
        break

    all_gen_text = "\n".join(model_gen_texts)
    return turns, all_gen_text


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
    gen_logprobs = torch.log_softmax(gen_logits.float(), dim=-1)
    gen_targets = torch.tensor(gen_ids, device=model.device).unsqueeze(1)
    token_logprobs = gen_logprobs.gather(1, gen_targets).squeeze(1)  # (L_gen,)

    return token_logprobs


def compute_all_logprobs(model, turns):
    """Compute per-token logprobs for all turns (with grad if model is trainable).

    Uses the exact same input_ids and gen_ids as generation.
    Returns concatenated logprobs for all model-generated tokens.
    """
    all_lps = []
    for turn in turns:
        lps = compute_turn_logprobs(model, turn['input_ids'], turn['gen_ids'])
        all_lps.append(lps)
    return torch.cat(all_lps) if all_lps else torch.tensor([], device=model.device)


# ============================================================
# GRPO Loss
# ============================================================

def grpo_loss(per_token_logps, old_per_token_logps, advantages,
              beta=0.04, epsilon_low=0.2, epsilon_high=0.28):
    """Compute GRPO policy gradient loss (token-level, no mask needed).

    Args:
        per_token_logps: (T,) current model logprobs
        old_per_token_logps: (T,) old logprobs (detached)
        advantages: scalar advantage for this completion
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
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
            "max_turns": MAX_TURNS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "num_samples": NUM_SAMPLES,
            "total_steps": total_steps,
        },
        tags=["grpo", "search-r1", "hotpotqa"],
    )

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
            optimizer.zero_grad()
            batch_loss = 0.0
            all_rewards, all_fmt, all_acc, all_lengths = [], [], [], []
            n_micro = 0

            for micro in range(GRADIENT_ACCUMULATION_STEPS):
                sample_start = batch_start + micro * PER_DEVICE_BATCH_SIZE
                if sample_start >= len(dataset):
                    break
                n_micro += 1
                batch = dataset[sample_start:sample_start + PER_DEVICE_BATCH_SIZE]

                for sample_idx in range(len(batch['prompt'])):
                    item = batch['prompt'][sample_idx]
                    gt = batch['ground_truth'][sample_idx]
                    system_prompt = item['system']
                    question = item['question']

                    # Build prompt token IDs
                    prompt_ids = make_prompt_ids(tokenizer, system_prompt, question)

                    # Generate G completions for this prompt
                    group_turns = []
                    group_texts = []
                    group_old_lps = []

                    for g in range(NUM_GENERATIONS):
                        turns, full_text = generate_with_search(
                            model, tokenizer, prompt_ids, wiki,
                            max_turns=MAX_TURNS,
                            max_tokens_per_turn=MAX_TOKENS_PER_TURN,
                            temperature=TEMPERATURE,
                        )
                        group_turns.append(turns)
                        group_texts.append(full_text)

                        # Old logprobs (detached)
                        with torch.no_grad():
                            old_lps = compute_all_logprobs(model, turns)
                            group_old_lps.append(old_lps)

                        total_tokens = sum(len(t['gen_ids']) for t in turns)
                        all_lengths.append(total_tokens)

                    # Compute rewards
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
                    rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std()
                    advantages = (rewards_t - mean_r) / (std_r + 1e-4)

                    all_rewards.extend(group_rewards)
                    all_fmt.extend(group_fmt)
                    all_acc.extend(group_acc)

                    # Free GPU memory from generation before computing loss
                    torch.cuda.empty_cache()

                    # Compute GRPO loss for each completion
                    for g in range(NUM_GENERATIONS):
                        turns = group_turns[g]
                        old_lps = group_old_lps[g]

                        if len(old_lps) == 0:
                            continue

                        # New logprobs (with grad) — same forward passes as generation
                        new_lps = compute_all_logprobs(model, turns)

                        if len(new_lps) == 0:
                            continue

                        # Ensure alignment
                        min_len = min(len(new_lps), len(old_lps))
                        new_lps = new_lps[:min_len]
                        old_lps = old_lps[:min_len].to(model.device)

                        adv = advantages[g]
                        loss = grpo_loss(new_lps, old_lps, adv,
                                         beta=BETA, epsilon_low=EPSILON_LOW,
                                         epsilon_high=EPSILON_HIGH)
                        loss = loss / (NUM_GENERATIONS * GRADIENT_ACCUMULATION_STEPS)
                        loss.backward()

                        batch_loss += loss.item() * NUM_GENERATIONS * GRADIENT_ACCUMULATION_STEPS

            if n_micro == 0:
                break

            # Gradient step — set LR before stepping so the schedule does not lag one step
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            for pg in optimizer.param_groups:
                pg['lr'] = get_lr(global_step)
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']

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
                print(f"[Step {global_step}/{total_steps}] "
                      f"loss={batch_loss:.4f} "
                      f"rew={sum(all_rewards)/n:.3f} "
                      f"fmt={sum(all_fmt)/n:.3f} "
                      f"acc={sum(all_acc)/n:.3f} "
                      f"len={sum(all_lengths)/n:.0f} "
                      f"lr={current_lr:.2e}")
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
