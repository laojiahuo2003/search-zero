"""
Generate SFT training data by running Search-R1 agent on QA datasets.
Each trajectory is saved as a multi-turn conversation in OpenAI messages format.

Usage:
    python scripts/generate_sft_data.py                              # built-in 8 samples
    python scripts/generate_sft_data.py -w 8                         # 8 workers
    python scripts/generate_sft_data.py data/hotpotqa_dev.json 200   # HotpotQA
"""
import sys
import os
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.react_agent import ReactAgent
from app.agent.prompts import REACT_SYSTEM_PROMPT

BUILT_IN = [
    {"question": "What government position was held by the woman who portrayed Jane Roe in the 1997 film 'Roe vs. Wade'?", "answer": "district attorney"},
    {"question": "Are both the director of 'Inception' and the director of 'Interstellar' from the same country?", "answer": "yes"},
    {"question": "What year did the team that won the first Super Bowl change their name to their current name?", "answer": "1964"},
    {"question": "Who lived longer, the composer of 'The Magic Flute' or the author of 'The Trial'?", "answer": "the author of The Trial"},
    {"question": "Which film has a higher IMDb rating: The Shawshank Redemption or The Godfather?", "answer": "The Shawshank Redemption"},
    {"question": "What is the capital of the country that produced the inventor of the telephone?", "answer": "Ottawa"},
    {"question": "Who was born first, Albert Einstein or Marie Curie?", "answer": "Marie Curie"},
    {"question": "What major sporting event was held in the same year that Instagram was founded?", "answer": "2010 FIFA World Cup"},
]

_print_lock = threading.Lock()


def log(idx: int, total: int, question: str, status: str):
    """Thread-safe progress print."""
    with _print_lock:
        try:
            print(f"[{idx}/{total}] {question[:80]}... {status}", flush=True)
        except UnicodeEncodeError:
            print(f"[{idx}/{total}] (unicode question)... {status}", flush=True)


def build_sft_messages(question: str, agent: ReactAgent) -> dict | None:
    """Run the agent and capture the full ReAct trajectory as SFT training messages."""
    raw = agent.run(question)

    steps = raw.get("steps", []) if isinstance(raw, dict) else raw.steps
    final_answer = raw.get("final_answer", "") if isinstance(raw, dict) else raw.final_answer
    sub_queries = raw.get("sub_queries", []) if isinstance(raw, dict) else raw.sub_queries

    if not steps:
        return None

    messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT.format(
        step_num=1, max_steps=agent.max_steps
    )}]
    messages.append({"role": "user", "content": f"Question: {question}"})

    for step in steps:
        if isinstance(step, dict):
            thought = step.get("thought", "")
            action = step.get("action", "")
            action_query = step.get("action_query", "")
            observation = step.get("observation", "")
        else:
            thought = step.thought
            action = step.action
            action_query = step.action_query
            observation = step.observation

        assistant_msg = f"THOUGHT: {thought}\nACTION: {action}"
        if action_query:
            assistant_msg += f": {action_query}"
        messages.append({"role": "assistant", "content": assistant_msg})

        if observation:
            messages.append({"role": "user", "content": f"OBSERVATION: {observation[:1500]}"})

    # Only add final_answer from _node_answer if the last step didn't already answer.
    # The last step's THOUGHT + ACTION: ANSWER is the canonical model output for SFT.
    last_action = ""
    if steps:
        last = steps[-1]
        last_action = last.get("action", "") if isinstance(last, dict) else last.action
    if final_answer and not last_action.upper().startswith("ANSWER"):
        messages.append({"role": "assistant", "content": f"THOUGHT: I now have sufficient information to answer.\nACTION: ANSWER: {final_answer}"})

    return {
        "messages": messages,
        "metadata": {
            "question": question,
            "num_steps": len(steps),
            "sub_queries": sub_queries,
        },
    }


def process_one(idx: int, total: int, question: str) -> dict | None:
    """Worker: create its own agent, run one question, return result or None."""
    agent = ReactAgent(use_retrieval=False)
    try:
        sample = build_sft_messages(question, agent)
        if sample and len(sample["messages"]) >= 4:
            log(idx, total, question, f"OK ({sample['metadata']['num_steps']} steps)")
            return sample
        else:
            log(idx, total, question, "SKIP (empty trajectory)")
            return None
    except Exception as e:
        log(idx, total, question, f"ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories from QA pairs")
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to HotpotQA-style JSON file")
    parser.add_argument("limit", nargs="?", type=int, default=8,
                        help="Max questions to process (default: 8)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    # Load QA pairs
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        qa_pairs = [{"question": d["question"], "answer": d["answer"]} for d in raw[:args.limit]]
    else:
        qa_pairs = BUILT_IN[:args.limit]

    output_dir = Path("data/sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    success = 0
    total = len(qa_pairs)
    output_path = output_dir / "sft_trajectories.jsonl"

    print(f"Generating SFT data for {total} questions ({args.workers} workers)...\n")

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, i + 1, total, qa["question"]): i
            for i, qa in enumerate(qa_pairs)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                # Worker crashed; log and continue
                idx = futures[future] + 1
                log(idx, total, qa_pairs[futures[future]]["question"], f"CRASH: {e}")
                continue
            if result is not None:
                all_samples.append(result)
                success += 1
            # Incremental save every 100 samples
            if success % 100 == 0 and all_samples:
                with open(output_path, "w", encoding="utf-8") as f:
                    for sample in all_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    elapsed = time.time() - start

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    inspect_path = output_dir / "sft_trajectories_readable.json"
    with open(inspect_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Done: {success}/{total} success  |  {elapsed:.1f}s  |  {success/max(1,elapsed)*60:.0f} samples/min")
    print(f"Saved: {output_path}")
    print(f"       {inspect_path}")

    if all_samples:
        steps = [s["metadata"]["num_steps"] for s in all_samples]
        msg_counts = [len(s["messages"]) for s in all_samples]
        print(f"Avg steps: {sum(steps)/len(steps):.1f}  |  Avg messages/sample: {sum(msg_counts)/len(msg_counts):.1f}")

        # Preview first sample
        print(f"\n--- Sample 0 preview ---")
        for msg in all_samples[0]["messages"][:3]:
            print(f"  [{msg['role']}]: {msg['content'][:150]}...")


if __name__ == "__main__":
    main()
