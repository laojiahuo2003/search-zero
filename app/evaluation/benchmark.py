"""
Benchmark runner — compares baseline RAG vs Search-R1 Mini on QA datasets.
"""
import json
import time
import os
from dataclasses import dataclass, field
from app.evaluation.metrics import evaluate
from app.agent.react_agent import ReactAgent
from app.utils.llm import get_llm


@dataclass
class BenchmarkResult:
    name: str
    predictions: list[str] = field(default_factory=list)
    ground_truths: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    elapsed: float = 0.0


class BaselineRAG:
    """Simple single-turn RAG: search once, then answer."""

    def __init__(self):
        from app.tools.search import SearchTool
        self.search = SearchTool()
        self.llm = get_llm()

    def answer(self, question: str) -> str:
        result = self.search.search(question)
        docs_text = "\n".join([
            f"[{i+1}] {d.title}\n{d.content[:300]}"
            for i, d in enumerate(result.documents[:5])
        ]) if result.success else "No search results."

        prompt = f"""Answer the question based on the search results.
Question: {question}

Search results:
{docs_text}

Provide a concise answer:"""

        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate(messages, max_tokens=512)


def load_hotpotqa(path: str, limit: int = 20) -> list[dict]:
    """Load HotpotQA data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


def run_benchmark(
    questions: list[str],
    ground_truths: list[str],
    method: str = "search_r1",
    limit: int = 20,
) -> BenchmarkResult:
    """Run a benchmark comparing two methods on a set of questions."""
    if method == "search_r1":
        agent = ReactAgent()
        runner = agent
    else:
        runner = BaselineRAG()

    result = BenchmarkResult(name=method)
    result.questions = questions[:limit]
    result.ground_truths = ground_truths[:limit]

    start = time.time()
    for i, q in enumerate(result.questions):
        try:
            if method == "search_r1":
                # Warm up agent
                state = runner.run(q)
                pred = state.final_answer
            else:
                pred = runner.answer(q)

            result.predictions.append(pred)
            print(f"  [{i+1}/{limit}] {q[:80]}... -> {pred[:80]}...")
        except Exception as e:
            print(f"  [{i+1}/{limit}] ERROR: {e}")
            result.predictions.append("")
    result.elapsed = time.time() - start

    return result


def run_full_benchmark(data_path: str | None = None, limit: int = 20):
    """Run both baseline and Search-R1 benchmark."""
    # Sample questions from HotpotQA style (multi-hop QA)
    sample_questions = [
        {
            "question": "What government position was held by the woman who portrayed Jane Roe in the 1997 film 'Roe vs. Wade'?",
            "answer": "district attorney",
        },
        {
            "question": "Are both the director of 'Inception' and the director of 'Interstellar' from the same country?",
            "answer": "yes",
        },
        {
            "question": "What year did the team that won the first Super Bowl change their name to their current name?",
            "answer": "1964",
        },
        {
            "question": "Who lived longer, the composer of 'The Magic Flute' or the author of 'The Trial'?",
            "answer": "the author of The Trial",
        },
        {
            "question": "Which film has a higher IMDb rating: The Shawshank Redemption or The Godfather?",
            "answer": "The Shawshank Redemption",
        },
        {
            "question": "What is the capital of the country that produced the inventor of the telephone?",
            "answer": "Ottawa",
        },
        {
            "question": "Who was born first, Albert Einstein or Marie Curie?",
            "answer": "Marie Curie",
        },
        {
            "question": "What major sporting event was held in the same year that Instagram was founded?",
            "answer": "2010 FIFA World Cup",
        },
    ]

    if data_path and os.path.exists(data_path):
        data = load_hotpotqa(data_path, limit)
        questions = [d["question"] for d in data]
        ground_truths = [d["answer"] for d in data]
    else:
        questions = [d["question"] for d in sample_questions]
        ground_truths = [d["answer"] for d in sample_questions]

    print(f"\n{'='*60}")
    print(f"Search-R1 Mini Benchmark")
    print(f"{'='*60}")
    print(f"Questions: {min(len(questions), limit)}")
    print(f"Methods: Baseline RAG vs Search-R1 Mini\n")

    # Baseline
    print("--- Baseline RAG (single search + answer) ---")
    baseline = run_benchmark(questions, ground_truths, method="baseline", limit=limit)
    baseline_metrics = evaluate(baseline.predictions, baseline.ground_truths)

    # Search-R1
    print("\n--- Search-R1 Mini (iterative ReAct search) ---")
    search_r1 = run_benchmark(questions, ground_truths, method="search_r1", limit=limit)
    search_r1_metrics = evaluate(search_r1.predictions, search_r1.ground_truths)

    # Comparison table
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Baseline RAG':<18} {'Search-R1 Mini':<18} {'Improvement'}")
    print(f"{'-'*20} {'-'*18} {'-'*18} {'-'*12}")
    for key in ["exact_match", "accuracy", "f1"]:
        b = baseline_metrics[key]
        s = search_r1_metrics[key]
        improvement = f"+{s - b:.2f}%"
        print(f"{key:<20} {b}%{'':<15} {s}%{'':<15} {improvement}")

    print(f"\nBaseline time: {baseline.elapsed:.1f}s")
    print(f"Search-R1 time: {search_r1.elapsed:.1f}s")
    print(f"Search-R1 avg steps: {sum(1 for _ in search_r1.predictions) / max(1, len(search_r1.predictions)):.1f}")

    return baseline_metrics, search_r1_metrics


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run_full_benchmark(data_path, limit)
