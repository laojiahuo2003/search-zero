"""
Build local Wikipedia search index from HotpotQA distractor contexts.
Creates:
  1. wiki_index.json — article title → sentences, for local search
  2. hotpotqa_train_500.json — training data with questions + answers + contexts
  3. hotpotqa_eval_100.json — evaluation data
"""
import json
import os
import sys

# Try without HF mirror first (it had SSL issues)
os.environ.pop("HF_ENDPOINT", None)


def build_index(data_dir: str, num_train: int = 500, num_eval: int = 100):
    """Download HotpotQA distractor dev, extract contexts, build search index."""
    print("Loading HotpotQA distractor validation set (with contexts)...")
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation",
        trust_remote_code=True, streaming=True
    )

    # Collect data
    all_articles = {}  # title → list of sentences (deduplicated)
    train_samples = []
    eval_samples = []
    idx = 0

    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")
        contexts = item.get("context", {})

        # Context format: {"title": [t1, t2, ...], "sentences": [[s1, s2, ...], ...]}
        titles = contexts.get("title", [])
        sentences = contexts.get("sentences", [])

        # Add to search index
        for title, sents in zip(titles, sentences):
            title = title.strip()
            if title not in all_articles:
                all_articles[title] = set()
            for s in sents:
                s = s.strip()
                if s:
                    all_articles[title].add(s)

        # Build sample data
        sample = {
            "question": question,
            "answer": answer,
            "context_titles": titles,
            "context_sentences": [[s.strip() for s in sents] for sents in sentences],
        }

        if idx < num_train:
            train_samples.append(sample)
        elif idx < num_train + num_eval:
            eval_samples.append(sample)

        idx += 1
        if idx % 1000 == 0:
            print(f"  Processed {idx} questions, {len(all_articles)} unique articles...")

        if idx >= num_train + num_eval:
            break

    print(f"\nDone: {idx} questions, {len(all_articles)} unique articles")

    # Convert articles to searchable list
    article_list = []
    total_sents = 0
    for title, sents in all_articles.items():
        sent_list = list(sents)
        article_list.append({
            "title": title,
            "sentences": sent_list,
            "full_text": " ".join(sent_list),
        })
        total_sents += len(sent_list)

    print(f"Total: {len(article_list)} articles, {total_sents} sentences")

    # Save search index
    os.makedirs(data_dir, exist_ok=True)
    index_path = os.path.join(data_dir, "wiki_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(article_list, f, ensure_ascii=False)
    size_mb = os.path.getsize(index_path) / 1024 / 1024
    print(f"Saved index: {index_path} ({size_mb:.1f} MB)")

    # Save training data (with contexts)
    train_path = os.path.join(data_dir, "hotpotqa_train_500.json")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved train: {train_path} ({len(train_samples)} samples)")

    # Save eval data
    eval_path = os.path.join(data_dir, "hotpotqa_eval_100.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved eval: {eval_path} ({len(eval_samples)} samples)")

    # Print sample
    if article_list:
        print(f"\nExample: {article_list[0]['title']}")
        print(f"  {article_list[0]['sentences'][0][:150]}...")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.utils.config import get_config

    out_dir = get_config().data_dir
    build_index(out_dir, num_train=500, num_eval=100)
