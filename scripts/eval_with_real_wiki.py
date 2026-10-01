"""
Evaluation script — loads the GRPO LoRA checkpoint and evaluates on HotpotQA.

Training-consistency contract (IMPORTANT):
  The rollout reuses the EXACT context construction of the GRPO training
  loop (train_grpo_search_MI300X.py): GRPO_SYSTEM_PROMPT, make_prompt_ids()
  for the initial prompt, tokenize_observation() to wrap each observation
  as a user turn, and extract_answer() to read the final answer. Evaluating
  with a different prompt/format would measure a different policy than the
  one that was trained.

  Decoding is greedy (do_sample=False) so results are reproducible.

Default search backend is the LOCAL wiki index (data/wiki_index.json) — the
same deterministic backend used during training, no network needed.

Usage:
    python eval_with_real_wiki.py \
        --checkpoint outputs/search_r1_grpo_search \
        --eval_data data/hotpotqa_eval_100.json \
        --output eval_results.json

For live Wikipedia instead (requires the HTTP proxy):
    python eval_with_real_wiki.py --wiki_mode url \
        --checkpoint /data/outputs/search_r1_grpo_search/checkpoint-XXX \
        --eval_data /data/hotpotqa/eval.json \
        --wiki_url http://127.0.0.1:18080/search \
        --output eval_results.json

On small GPUs (e.g. RTX 3080 10GB) add --load_in_4bit.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiki_search import LocalWikiSearcher
# Training-side context construction — single source of truth for the
# prompt/observation format. Do NOT re-implement these locally.
from train_grpo_search_MI300X import (
    GRPO_SYSTEM_PROMPT,
    make_prompt_ids,
    tokenize_observation,
    extract_answer,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.config import get_config

_cfg = get_config()

# ============================================================
# Config
# ============================================================
MODEL_PATH = _cfg.base_model
MAX_TURNS = 3            # must match training (train_grpo_search_MI300X)
MAX_TOKENS_PER_TURN = 256


def load_model(checkpoint_path: Optional[str] = None, load_in_4bit: bool = False):
    """Load base model + optional LoRA checkpoint.

    load_in_4bit quantises the base via bitsandbytes so a 7B model fits a
    ~10GB GPU; the adapter is then kept UNMERGED (merging is unreliable on
    4-bit weights).
    """
    print(f"Loading base model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if load_in_4bit:
            print("  LoRA loaded (unmerged — 4-bit base).")
        else:
            model = model.merge_and_unload()
            print("  LoRA loaded and merged.")

    model.eval()
    return model, tokenizer


def wiki_search(query: str, wiki_url: str, top_k: int = 3, sentences: int = 3,
                local_searcher=None) -> str:
    """Search via local index (training-consistent) or Wikipedia HTTP proxy.

    Args:
        query: search query string
        wiki_url: HTTP proxy URL (used only when local_searcher is None)
        local_searcher: LocalWikiSearcher instance; if given, search the local
            index with the exact same backend used during training
    """
    if local_searcher is not None:
        return local_searcher.search(query, top_k=top_k, sentences=sentences)

    try:
        r = requests.get(wiki_url, params={"q": query, "top_k": top_k, "sentences": sentences}, timeout=15)
        data = r.json()
        results = data.get("results", [])
        if not results:
            return f'OBSERVATION: No results found for "{query}".'

        parts = []
        for item in results:
            parts.append(f"[{item['rank']}] {item['title']}\n    {item['summary']}\n    URL: {item['url']}")
        return "OBSERVATION:\n" + "\n\n".join(parts)
    except Exception as e:
        return f"OBSERVATION: Search failed: {e}"


def generate_answer(model, tokenizer, question: str, wiki_url: str,
                    local_searcher=None) -> tuple:
    """Multi-turn ReAct generation with search (local index or live Wikipedia).

    Mirrors the training rollout (generate_with_search) one-to-one:
    same prompt, same observation wrapping, same turn structure — only the
    decoding differs (greedy here, sampled during training).
    """
    current_ids = make_prompt_ids(tokenizer, GRPO_SYSTEM_PROMPT, question)
    all_turns = []

    for _ in range(MAX_TURNS):
        # Same guard as training: cap the context so a long observation
        # cannot push the sequence past the truncation limit.
        if len(current_ids) > 2048:
            current_ids = current_ids[-2048:]
        input_tensor = torch.tensor([current_ids], device=model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_tensor,
                max_new_tokens=MAX_TOKENS_PER_TURN,
                do_sample=False,  # greedy: reproducible eval
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0, input_tensor.shape[1]:].tolist()
        if not gen_ids:
            break
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        all_turns.append(gen_text)

        # ANSWER ends the trajectory (same regex family as training).
        if re.search(r'ACTION\s*:\s*ANSWER\s*:', gen_text, re.IGNORECASE):
            answer = extract_answer(gen_text)
            return (answer if answer else gen_text.strip()), all_turns

        # SEARCH appends the observation wrapped as a user turn — the exact
        # token structure the model saw during training.
        search_match = re.search(
            r'ACTION\s*:\s*SEARCH\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT|$)|$)',
            gen_text, re.IGNORECASE | re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            if query:
                obs = wiki_search(query, wiki_url, local_searcher=local_searcher)
                obs_ids = tokenize_observation(tokenizer, obs)
                current_ids = current_ids + gen_ids + obs_ids
                continue

        break  # No action found, stop

    # No ANSWER found - extract last line as answer fallback
    return all_turns[-1].strip() if all_turns else "(no output)", all_turns


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def exact_match(pred: str, gold: str) -> bool:
    n_pred, n_gold = normalize_answer(pred), normalize_answer(gold)
    if not n_gold:
        return False
    return n_pred == n_gold


def contains_match(pred: str, gold: str) -> bool:
    """Check if gold answer is contained in prediction."""
    n_pred, n_gold = normalize_answer(pred), normalize_answer(gold)
    if not n_gold or not n_pred:
        return False
    return n_gold in n_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--eval_data", type=str, default=_cfg.hotpotqa_eval_path)
    parser.add_argument("--wiki_url", type=str, default="http://127.0.0.1:18080/search")
    parser.add_argument("--wiki_mode", type=str, choices=["local", "url"], default="local",
                        help="local = local index (same backend as training), url = HTTP wiki proxy")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="quantise base model to 4-bit (small GPUs); adapter stays unmerged")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    # Load data
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    if args.max_samples > 0:
        eval_data = eval_data[:args.max_samples]
    print(f"Evaluating {len(eval_data)} samples...")

    # Initialize search backend
    local_searcher = None
    if args.wiki_mode == "local":
        index_path = _cfg.wiki_index_path
        if not os.path.exists(index_path):
            print(f"  FATAL: local wiki index not found: {index_path}")
            print("  Build it first: python scripts/build_wiki_index.py")
            sys.exit(1)
        local_searcher = LocalWikiSearcher(index_path)
        print(f"  Using LOCAL search: {local_searcher.get_stats()['articles']} articles")
    else:
        # Test wiki connection
        print(f"Testing wiki proxy: {args.wiki_url}?q=test")
        try:
            r = requests.get(args.wiki_url, params={"q": "test", "top_k": 1, "sentences": 1}, timeout=10)
            if r.status_code == 200:
                print("  Wiki proxy OK")
            else:
                print(f"  WARNING: HTTP {r.status_code}")
                print("  Start local server: python scripts/wiki_search_server.py")
                print("  Then SSH tunnel: ssh -R 18080:127.0.0.1:18080 root@IP -p PORT")
        except Exception as e:
            print(f"  WARNING: Cannot reach wiki proxy: {e}")
            print("  Make sure SSH tunnel is active.")

    # Load model
    model, tokenizer = load_model(args.checkpoint, load_in_4bit=args.load_in_4bit)

    # Evaluate
    results = []
    em_correct = 0
    contains_correct = 0
    total = 0

    for i, item in enumerate(eval_data):
        question = item["question"]
        gold_answer = item["answer"]

        print(f"\n[{i+1}/{len(eval_data)}] Q: {question[:100]}...")
        pred_answer, turns = generate_answer(model, tokenizer, question, args.wiki_url,
                                             local_searcher)

        em = exact_match(pred_answer, gold_answer)
        cm = contains_match(pred_answer, gold_answer)
        if em:
            em_correct += 1
        if cm:
            contains_correct += 1
        total += 1

        print(f"  Pred: {pred_answer[:150]}")
        print(f"  Gold: {gold_answer[:150]}")
        print(f"  EM={em}, Contains={cm} | EM acc={em_correct/total:.3f}, Contains acc={contains_correct/total:.3f}")

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "turns": turns,
            "exact_match": em,
            "contains_match": cm,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {total} samples")
    print(f"  Exact Match: {em_correct}/{total} = {em_correct/total:.3f}")
    print(f"  Contains:    {contains_correct}/{total} = {contains_correct/total:.3f}")
    print(f"{'='*60}")

    # Save
    summary = {
        "total": total,
        "exact_match": em_correct,
        "contains_match": contains_correct,
        "em_accuracy": em_correct / total if total > 0 else 0,
        "contains_accuracy": contains_correct / total if total > 0 else 0,
        "checkpoint": args.checkpoint,
        "decoding": "greedy",
        "prompt_format": "training-consistent (GRPO_SYSTEM_PROMPT + tokenize_observation)",
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
