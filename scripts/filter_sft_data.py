"""
Filter SFT trajectories by comparing extracted final answers against HotpotQA ground truth.
Uses an LLM judge (cheap + fast) for answer extraction and correctness comparison.

Usage:
    python scripts/filter_sft_data.py                     # all 975 samples
    python scripts/filter_sft_data.py -w 32               # 32 concurrent workers
    python scripts/filter_sft_data.py --dry-run           # inspect first 5 without API calls
"""
import sys
import os
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.config import get_config

_print_lock = threading.Lock()

# Patterns that indicate the model gave up / couldn't answer
_GIVEUP_PATTERNS = [
    "no information",
    "no relevant information",
    "cannot answer",
    "can't answer",
    "not possible to answer",
    "impossible to answer",
    "not possible to identify",
    "not possible to determine",
    "unable to determine",
    "unable to find",
    "unable to answer",
    "no research results",
    "no results found",
    "no supporting information",
    "does not provide",
    "no data",
    "no evidence",
    "no records",
    "without additional information",
    "insufficient information",
    "lack of information",
    "cannot be determined",
    "cannot determine",
    "could not be determined",
    "not enough information",
    "no mention",
    "no reference",
    "no sources",
]


def is_giveup(text: str) -> bool:
    """Check if the model's answer is a 'give up' / 'I don't know' response."""
    # Only check the ANSWER: section if present, fall back to last 500 chars
    text_lower = text.lower()
    # Check for ANSWER section
    answer_start = text_lower.rfind("answer:")
    if answer_start >= 0:
        check_text = text_lower[answer_start:]
    else:
        check_text = text_lower[-500:]
    return any(p in check_text for p in _GIVEUP_PATTERNS)


def has_answer_action(text: str) -> bool:
    """Check if the last assistant message contains an ANSWER action."""
    return "ANSWER:" in text.upper()


def log(msg: str):
    with _print_lock:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode(), flush=True)


def extract_final_answer_and_judge(
    client: OpenAI,
    model: str,
    question: str,
    sft_answer: str,
    ground_truth: str,
) -> bool:
    """
    Ask an LLM to determine whether the SFT trajectory's final answer
    matches the HotpotQA ground truth answer.
    Returns True if the answer is correct, False otherwise.
    """
    prompt = f"""You are a strict but fair answer judge. Determine if the model's answer to the question is factually correct according to the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Response (last assistant message): {sft_answer[:2000]}

Rules:
- CORRECT = model gives the SAME factual answer as ground truth (wording can differ but core facts must match).
- INCORRECT = model gives a DIFFERENT answer, CONTRADICTS ground truth, or says "I don't know" / "no information" / "cannot answer" / any variation of giving up or being evasive.
- Be STRICT: if the model fails to provide a clear matching answer, mark INCORRECT.

Reply with ONLY one word: "CORRECT" or "INCORRECT"."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "CORRECT" in verdict
    except Exception as e:
        log(f"  Judge API error: {e}")
        return None  # unknown


def process_one(
    idx: int,
    total: int,
    sample: dict,
    ground_truth: str,
    client: OpenAI,
    model: str,
) -> dict | None:
    """Judge one trajectory. Hard-rule pre-filter + LLM judge."""
    question = sample["metadata"]["question"]
    msgs = sample["messages"]
    last_asst = [m for m in msgs if m["role"] == "assistant"][-1]["content"]

    # --- Hard rule 1: must have an ANSWER action ---
    if not has_answer_action(last_asst):
        log(f"[{idx}/{total}] RULE-SKIP (no ANSWER): {question[:60]}")
        sample["metadata"]["correct"] = False
        sample["metadata"]["judge_reason"] = "no ANSWER action"
        return sample

    # --- Hard rule 2: must not be a "give up" response ---
    if is_giveup(last_asst):
        log(f"[{idx}/{total}] RULE-SKIP (giveup): {question[:60]}")
        sample["metadata"]["correct"] = False
        sample["metadata"]["judge_reason"] = "model gave up / no information"
        return sample

    # --- LLM judge for remaining ---
    is_correct = extract_final_answer_and_judge(
        client, model, question, last_asst, ground_truth
    )

    if is_correct is None:
        log(f"[{idx}/{total}] ERROR (judge failed): {question[:60]}")
        return None

    status = "CORRECT" if is_correct else "WRONG"
    log(f"[{idx}/{total}] {status}: {question[:60]}")
    sample["metadata"]["correct"] = is_correct
    return sample


def main():
    parser = argparse.ArgumentParser(description="Filter SFT trajectories by answer correctness")
    parser.add_argument("-w", "--workers", type=int, default=16,
                        help="Concurrent judge workers (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 5 answer extractions without API calls")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples to judge (0 = all)")
    args = parser.parse_args()

    config = get_config()
    client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)
    # Judge with the same model as the agent (LLM_MODEL from .env).
    model = config.llm_model

    # Load SFT data
    sft_path = config.sft_trajectories_path
    if not os.path.exists(sft_path):
        print(f"ERROR: {sft_path} not found")
        sys.exit(1)

    with open(sft_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Load HotpotQA ground truth
    gt_path = config.hotpotqa_dev_path
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    # Build lookup by question
    gt_map = {d["question"]: d["answer"] for d in gt_data}

    # Match samples to ground truth
    matched = []
    unmatched = []
    for s in samples:
        q = s["metadata"]["question"]
        if q in gt_map:
            matched.append((s, gt_map[q]))
        else:
            unmatched.append(s)
            s["metadata"]["correct"] = None  # no ground truth available

    print(f"Loaded {len(samples)} trajectories")
    print(f"  Matched to ground truth: {len(matched)}")
    print(f"  No ground truth match:  {len(unmatched)}")

    if args.dry_run:
        print("\n--- Dry run: inspecting first 5 matched samples ---")
        for i, (s, gt) in enumerate(matched[:5]):
            question = s["metadata"]["question"]
            msgs = s["messages"]
            last_asst = [m for m in msgs if m["role"] == "assistant"][-1]["content"]
            # Extract ANSWER: line if present
            answer_line = ""
            for line in last_asst.split("\n"):
                if "ANSWER:" in line.upper():
                    answer_line = line.strip()
            print(f"\n[{i+1}] Q: {question[:100]}")
            print(f"    GT: {gt}")
            print(f"    Extracted ANSWER line: {answer_line[:200]}")
            print(f"    Last 200 chars: ...{last_asst[-200:]}")
        return

    # Limit if requested
    if args.limit > 0:
        matched = matched[:args.limit]

    total = len(matched)
    print(f"\nJudging {total} samples with {args.workers} workers...\n")

    start = time.time()

    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (s, gt) in enumerate(matched):
            f = pool.submit(process_one, i + 1, total, s, gt, client, model)
            futures[f] = i

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                idx = futures[future] + 1
                log(f"[{idx}/{total}] CRASH: {e}")
                continue
            if result is not None:
                judged.append(result)

    elapsed = time.time() - start

    # Statistics with breakdown
    correct = [s for s in judged if s["metadata"].get("correct")]
    wrong = [s for s in judged if s["metadata"].get("correct") is False]
    unknown = [s for s in judged if s["metadata"].get("correct") is None]

    # Breakdown by reason
    rule_no_answer = [s for s in wrong if s["metadata"].get("judge_reason") == "no ANSWER action"]
    rule_giveup = [s for s in wrong if s["metadata"].get("judge_reason") == "model gave up / no information"]
    judge_wrong = [s for s in wrong if s["metadata"].get("judge_reason") is None]
    judge_correct = correct  # passed both rules + LLM judge

    print(f"\n{'='*50}")
    print(f"Filtering complete: {elapsed:.1f}s")
    print(f"  Hard-rule: no ANSWER action:     {len(rule_no_answer)}")
    print(f"  Hard-rule: model gave up:        {len(rule_giveup)}")
    print(f"  LLM judge: WRONG answer:         {len(judge_wrong)}")
    print(f"  LLM judge: CORRECT answer:       {len(judge_correct)}")
    print(f"  ERROR/unknown:                   {len(unknown)}")
    print(f"  ---")
    print(f"  Total kept for SFT:              {len(judge_correct)}")

    # Save filtered (only LLM-judged correct; unmatched kept as unverified)
    keep = judge_correct + unmatched

    clean_path = config.sft_filtered_path
    with open(clean_path, "w", encoding="utf-8") as f:
        for s in keep:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Also save all with correctness flags (for analysis)
    all_judged = judged + unmatched
    judged_path = os.path.join(config.sft_data_dir, "sft_trajectories_judged.jsonl")
    with open(judged_path, "w", encoding="utf-8") as f:
        for s in all_judged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSaved:")
    print(f"  Filtered (correct+unmatched): {clean_path} ({len(keep)} samples)")
    print(f"  All judged:                   {judged_path} ({len(all_judged)} samples)")

    # Recommend next steps
    if len(correct) >= 500:
        print(f"\nReady for SFT training with {len(keep)} clean samples!")
        print(f"  python scripts/train_sft.py")


if __name__ == "__main__":
    main()
