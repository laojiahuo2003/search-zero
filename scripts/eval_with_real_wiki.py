"""
Evaluation script — runs on cloud A100, connects to local Wikipedia proxy via SSH tunnel.

Usage on cloud:
    python eval_with_real_wiki.py \
        --checkpoint /data/outputs/search_r1_grpo_search/checkpoint-XXX \
        --eval_data /data/hotpotqa/eval.json \
        --wiki_url http://127.0.0.1:18080/search \
        --output eval_results.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.config import get_config

_cfg = get_config()

# ============================================================
# Config
# ============================================================
MODEL_PATH = _cfg.base_model
MAX_TURNS = 3
MAX_TOKENS_PER_TURN = 256
TEMPERATURE = 0.7  # Lower temp for eval (greedy-ish)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions by searching Wikipedia.

Always follow this format:
THOUGHT: <your reasoning>
ACTION: SEARCH: <search query>
or
ACTION: ANSWER: <final answer>

When you have enough information, output ACTION: ANSWER: with the answer."""


def load_model(checkpoint_path: Optional[str] = None):
    """Load base model + optional LoRA checkpoint."""
    print(f"Loading base model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        print("  LoRA loaded and merged.")

    model.eval()
    return model, tokenizer


def wiki_search(query: str, wiki_url: str, top_k: int = 3, sentences: int = 3) -> str:
    """Call the local Wikipedia HTTP proxy."""
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


def generate_answer(model, tokenizer, question: str, wiki_url: str) -> tuple:
    """Multi-turn ReAct generation with real Wikipedia search."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_turns = []
    for turn in range(MAX_TURNS):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        if inputs["input_ids"].shape[1] > 2048:
            inputs["input_ids"] = inputs["input_ids"][:, -2048:]

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_TOKENS_PER_TURN,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        all_turns.append(gen_text)

        # Check for ANSWER
        answer_match = re.search(r'ACTION\s*:\s*ANSWER\s*:\s*(.+?)$', gen_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if answer_match:
            return answer_match.group(1).strip(), all_turns

        # Check for SEARCH
        search_match = re.search(r'ACTION\s*:\s*SEARCH\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT|$)|$)',
                                 gen_text, re.IGNORECASE | re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            if query:
                obs = wiki_search(query, wiki_url)
                prompt_text += gen_text + "\n" + obs + "\n"
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
    return normalize_answer(pred) == normalize_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    """Check if gold answer is contained in prediction."""
    return normalize_answer(gold) in normalize_answer(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--eval_data", type=str, default=_cfg.hotpotqa_eval_path)
    parser.add_argument("--wiki_url", type=str, default="http://127.0.0.1:18080/search")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    # Load data
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    if args.max_samples > 0:
        eval_data = eval_data[:args.max_samples]
    print(f"Evaluating {len(eval_data)} samples...")

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
    model, tokenizer = load_model(args.checkpoint)

    # Evaluate
    results = []
    em_correct = 0
    contains_correct = 0
    total = 0

    for i, item in enumerate(eval_data):
        question = item["question"]
        gold_answer = item["answer"]

        print(f"\n[{i+1}/{len(eval_data)}] Q: {question[:100]}...")
        pred_answer, turns = generate_answer(model, tokenizer, question, args.wiki_url)

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
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
