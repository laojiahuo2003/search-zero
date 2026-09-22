"""
GRPO Training Script — Search-R1 RL Stage
==========================================
Uses TRL GRPOTrainer directly (LLaMA-Factory v0.9.3 lacks GRPO support).

Reward function:
  1. Format reward (0-1): ReAct format compliance (THOUGHT → ACTION → ANSWER)
  2. Accuracy reward (0-1): Answer similarity to ground truth

Usage (on A100):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data/miniconda/envs/torch/bin/python scripts/train_grpo.py
"""

import json
import re
import os
import sys
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/data/models/qwen/Qwen2___5-7B-Instruct"
SFT_CHECKPOINT = "/data/outputs/search_r1_sft"
DATA_PATH = "/data/sft/sft_trajectories_filtered.jsonl"
OUTPUT_DIR = "/data/outputs/search_r1_grpo"

# Training config
GRPO_CONFIG = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,  # disabled: conflicts with PeftModel.enable_input_require_grads hook during recomputation
    bf16=True,
    learning_rate=1.0e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,  # log every step to verify gradients working
    logging_first_step=True,
    log_level="info",
    save_steps=500,
    save_total_limit=2,
    max_completion_length=1024,
    num_generations=2,  # generate 2 completions per prompt for GRPO
    temperature=0.9,  # enough exploration, system prompt now guides format
    report_to="none",
    # Memory optimizations
    use_vllm=False,  # set True if vLLM installed
)

# ============================================================
# Reward Functions
# ============================================================

def extract_answer_section(text: str) -> tuple:
    """Extract the ANSWER section from generated text.
    Returns (answer_text, has_proper_format)"""
    # Check for ReAct structure
    has_thought = bool(re.search(r'THOUGHT\s*:', text, re.IGNORECASE))
    has_action = bool(re.search(r'ACTION\s*:', text, re.IGNORECASE))

    # Extract ANSWER content
    answer_match = re.search(
        r'ACTION\s*:\s*ANSWER\s*:\s*(.+?)(?:\n|$|ACTION|THOUGHT)',
        text, re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: look for ANSWER anywhere
        answer_match = re.search(
            r'ANSWER\s*:\s*(.+?)$',
            text, re.IGNORECASE | re.DOTALL
        )
        answer = answer_match.group(1).strip() if answer_match else ""

    return answer, (has_thought, has_action)


def format_reward(completions, **kwargs):
    """Reward ReAct format compliance: 0-1 scale.
    - THOUGHT: +0.3
    - ACTION: +0.3
    - ANSWER: +0.5 (strong incentive to complete trajectory)
    No penalty for missing ANSWER — the absence of the bonus is the penalty.
    Ensures within-group variance for GRPO to learn.
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0
        if re.search(r'THOUGHT\s*:', text, re.IGNORECASE):
            score += 0.3
        if re.search(r'ACTION\s*:', text, re.IGNORECASE):
            score += 0.3
        if re.search(r'ANSWER\s*:', text, re.IGNORECASE):
            score += 0.5  # brings full trajectory to 1.0+
        rewards.append(min(1.0, score))  # cap at 1.0 for stability
    return rewards


def accuracy_reward(completions, prompts=None, ground_truth=None, **kwargs):
    """Reward answer accuracy by comparing to ground truth.
    Gets ground_truth from dataset columns passed via kwargs.
    """
    rewards = []
    # ground_truth should come from dataset columns via kwargs
    if ground_truth is None:
        # Try to get from kwargs
        ground_truth = kwargs.get("ground_truth", [""] * len(completions))

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else (
            completion[0]["content"] if isinstance(completion, list) else str(completion)
        )

        # Extract answer
        answer, _ = extract_answer_section(text)
        if not answer:
            rewards.append(0.0)
            continue

        # Get ground truth for this sample
        gt = ground_truth[i] if isinstance(ground_truth, list) and i < len(ground_truth) else ""

        if not gt:
            # Without ground truth, give modest reward for having a substantial answer
            rewards.append(0.3 if len(answer) > 20 else 0.1)
            continue

        # Simple word overlap scoring
        gt_words = set(gt.lower().split())
        ans_words = set(answer.lower().split())
        if not gt_words:
            rewards.append(0.3)
            continue

        overlap = len(gt_words & ans_words)
        score = min(1.0, overlap / max(len(gt_words), 1) * 2)
        rewards.append(score)

    return rewards


# ============================================================
# Data Loading
# ============================================================

def load_grpo_dataset(data_path: str) -> Dataset:
    """Load SFT data and convert to GRPO format.
    Each sample gets:
      - prompt: system + user question (the input to the model)
      - ground_truth: the final answer from the assistant (for reward)
    """
    prompts = []
    ground_truths = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            messages = sample["messages"]

            # Find system message and first user question
            system_msg = ""
            user_question = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user" and not user_question:
                    user_question = msg["content"]

            # Build prompt: system + user question (the model will generate the trajectory)
            # CRITICAL: Rewrite system prompt for single-pass GRPO generation.
            # The SFT system message says "Current step: 1/5" which makes the model
            # generate ONE turn then stop (waiting for OBSERVATION). We need the
            # model to generate the FULL trajectory including the final ANSWER.
            grpo_system = (
                "You are a deep research AI agent. Answer the user's question "
                "using the ReAct framework. Provide your complete reasoning "
                "and final answer in one response.\n\n"
                "Always output in this exact format:\n\n"
                "THOUGHT: <your reasoning>\n"
                "ACTION: SEARCH: <search query>\n"
                "THOUGHT: <analyze results>\n"
                "ACTION: ANSWER: <final answer with citations>\n\n"
                "Important: You MUST end with ACTION: ANSWER: to provide the final answer."
            )
            prompt = [{"role": "system", "content": grpo_system}]
            prompt.append({"role": "user", "content": user_question})

            # Extract ground truth answer (last assistant message)
            ground_truth = ""
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    answer, _ = extract_answer_section(msg["content"])
                    if answer:
                        ground_truth = answer
                        break

            prompts.append(prompt)
            ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
    })

    print(f"Loaded {len(dataset)} samples for GRPO")
    return dataset


# ============================================================
# Main Training
# ============================================================

def main():
    print("=" * 60)
    print("  Search-R1 GRPO Training")
    print(f"  Model: {MODEL_PATH}")
    print(f"  SFT LoRA: {SFT_CHECKPOINT}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("[2/5] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load SFT LoRA — keep as LoRA, do NOT merge (would OOM on 40GB)
    print("[3/5] Loading SFT LoRA adapter (keeping LoRA structure)...")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    # Enable training for LoRA params
    for n, p in model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
    # CRITICAL: enable_input_require_grads ensures gradient checkpointing
    # produces gradients through LoRA layers. Without this, PyTorch checkpoint
    # sees no requires_grad inputs and skips gradient computation entirely.
    model.enable_input_require_grads()
    model.train()  # set training mode
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / Total: {total/1e9:.2f}B")
    # Verify gradient setup
    emb = model.get_input_embeddings()
    print(f"  Embedding requires_grad: {emb.weight.requires_grad}")

    # Load dataset
    print("[4/5] Loading GRPO dataset...")
    dataset = load_grpo_dataset(DATA_PATH)
    print(f"  Train samples: {len(dataset)}")

    # Create GRPO trainer
    print("[5/5] Starting GRPO training...")
    print(f"  Config: {GRPO_CONFIG}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=GRPO_CONFIG,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Add logging callback to verify training metrics
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                import pprint
                print(f"[Step {state.global_step}] Metrics: {pprint.pformat(logs, compact=True)}")
    trainer.add_callback(MetricsCallback())

    trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("  Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
