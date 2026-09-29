"""
Test a GRPO LoRA adapter on real questions (qualitative + EM/Contains).

ReAct loop mirrors scripts/eval_with_real_wiki.py, but:
  - loads the 7B base in 4-bit so it fits the RTX 3080 10GB
  - keeps the LoRA adapter unmerged (merging is unreliable on 4-bit)
  - search backend: real Wikipedia (wikipedia pkg) with DuckDuckGo fallback

Usage:
    uv run python scripts/test_adapter.py \
        --base ~/huggingface/Qwen2.5-7B-Instruct \
        --adapter outputs/search_r1_grpo_search_0929 \
        --max-samples 5
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_TURNS = 3
MAX_TOKENS_PER_TURN = 256
TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are a helpful assistant that answers questions by searching Wikipedia.

Always follow this format:
THOUGHT: <your reasoning>
ACTION: SEARCH: <search query>
or
ACTION: ANSWER: <final answer>

When you have enough information, output ACTION: ANSWER: with the answer."""


# ---------------------------------------------------------------- search ----
def wiki_search_wikipedia(query: str, top_k: int = 3, sentences: int = 3) -> str:
    """Real Wikipedia via the `wikipedia` package (same as wiki_search_server)."""
    import wikipedia

    try:
        titles = wikipedia.search(query, results=top_k)
        if not titles:
            return f'OBSERVATION: No results found for "{query}".'
        parts = []
        for i, title in enumerate(titles[:top_k], 1):
            try:
                summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
            except Exception:
                continue
            parts.append(f"[{i}] {title}\n    {summary}\n    URL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        if not parts:
            return f'OBSERVATION: No results found for "{query}".'
        return "OBSERVATION:\n" + "\n\n".join(parts)
    except Exception as e:
        return f"OBSERVATION: Search failed: {e}"


def wiki_search_ddgs(query: str, top_k: int = 3) -> str:
    """DuckDuckGo fallback when Wikipedia is unreachable."""
    from ddgs import DDGS

    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=top_k))
        if not results:
            return f'OBSERVATION: No results found for "{query}".'
        parts = []
        for i, r in enumerate(results[:top_k], 1):
            parts.append(f"[{i}] {r.get('title', '')}\n    {r.get('body', '')}\n    URL: {r.get('href', '')}")
        return "OBSERVATION:\n" + "\n\n".join(parts)
    except Exception as e:
        return f"OBSERVATION: Search failed: {e}"


def make_searcher(backend: str):
    if backend == "wikipedia":
        return wiki_search_wikipedia
    if backend == "ddgs":
        return wiki_search_ddgs

    def auto(query, top_k=3, sentences=3):
        obs = wiki_search_wikipedia(query, top_k, sentences)
        if "Search failed" in obs:
            obs = wiki_search_ddgs(query, top_k)
        return obs

    return auto


# ------------------------------------------------------------------ model ----
def load_model(base: str, adapter: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading base {base} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if adapter and os.path.exists(adapter):
        print(f"Loading LoRA adapter {adapter}...")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)  # keep unmerged on 4-bit
        print("  Adapter loaded (unmerged).")

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, search) -> tuple:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_turns = []
    for _ in range(MAX_TURNS):
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

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

        answer_match = re.search(
            r"ACTION\s*:\s*ANSWER\s*:\s*(.+?)$", gen_text, re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        if answer_match:
            return answer_match.group(1).strip(), all_turns

        search_match = re.search(
            r"ACTION\s*:\s*SEARCH\s*:\s*(.+?)(?:\n\s*(?:ACTION|THOUGHT|$)|$)",
            gen_text, re.IGNORECASE | re.DOTALL,
        )
        if search_match:
            query = search_match.group(1).strip()
            if query:
                obs = search(query)
                prompt_text += gen_text + "\n" + obs + "\n"
                continue
        break

    return (all_turns[-1].strip() if all_turns else "(no output)"), all_turns


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="/home/uos/huggingface/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/search_r1_grpo_search_0929")
    parser.add_argument("--data", type=str, default="data/hotpotqa_dev.json")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--search", type=str, default="auto", choices=["auto", "wikipedia", "ddgs"])
    parser.add_argument("--out", type=str, default="test_results.json")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_data = eval_data[: args.max_samples]
    print(f"Testing {len(eval_data)} samples, search backend={args.search}\n")

    search = make_searcher(args.search)
    model, tokenizer = load_model(args.base, args.adapter)

    em_correct = contains_correct = 0
    results = []
    for i, item in enumerate(eval_data):
        question, gold = item["question"], item["answer"]
        print(f"\n[{i + 1}/{len(eval_data)}] Q: {question}")
        pred, turns = generate_answer(model, tokenizer, question, search)

        print("  --- ReAct trajectory ---")
        for t in turns:
            print("  " + t.replace("\n", "\n  ")[:600])
            print("  " + "-" * 60)

        em = normalize_answer(pred) == normalize_answer(gold)
        cm = normalize_answer(gold) in normalize_answer(pred)
        em_correct += em
        contains_correct += cm
        print(f"  Pred : {pred[:200]}")
        print(f"  Gold : {gold[:200]}")
        print(f"  EM={em}, Contains={cm} | running: EM={em_correct}/{i + 1}, Contains={contains_correct}/{i + 1}")

        results.append({"question": question, "gold": gold, "pred": pred,
                        "turns": turns, "exact_match": em, "contains_match": cm})

    total = len(eval_data)
    print(f"\n{'=' * 60}")
    print(f"Results: {total} samples")
    print(f"  Exact Match: {em_correct}/{total} = {em_correct / total:.3f}")
    print(f"  Contains:    {contains_correct}/{total} = {contains_correct / total:.3f}")
    print(f"{'=' * 60}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"total": total, "em_accuracy": em_correct / total,
                   "contains_accuracy": contains_correct / total,
                   "adapter": args.adapter, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
