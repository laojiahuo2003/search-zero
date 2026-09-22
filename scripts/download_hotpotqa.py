"""Download HotpotQA dev set and save as simple QA JSON."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try HF mirror first for users in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

print("Loading HotpotQA (distractor, dev split)...")
data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
qa = [{"question": d["question"], "answer": d["answer"]} for d in data]

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "hotpotqa_dev.json")
out_path = os.path.abspath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(qa, f, ensure_ascii=False, indent=2)

print(f"Saved {len(qa)} QA pairs to {out_path}")
