"""
Query rewriter — decomposes complex questions into search-optimized sub-queries.
"""
import json
from app.utils.llm import get_llm

REWRITE_PROMPT = """You are a query decomposition expert. Your task is to break down a complex question into 2-4 simpler sub-questions that are optimized for web search.

Rules:
1. Each sub-question should be self-contained and answerable via a single web search.
2. Decompose by entities, time periods, or aspects.
3. Order sub-questions logically (background first, then specifics).
4. Output ONLY valid JSON.

Example:
Question: "Why did NVIDIA stock rise after earnings in May 2024?"
Output:
{
  "sub_queries": [
    "NVIDIA Q1 2024 earnings report results",
    "NVIDIA data center revenue Q1 2024",
    "analyst expectations NVIDIA earnings May 2024"
  ],
  "reasoning": "Decomposed into: actual earnings results, key revenue driver (data center), and market expectations for comparison."
}

Question: {question}

Output (JSON only):"""


class QueryRewriter:
    """Decomposes complex questions into search-friendly sub-queries."""

    def __init__(self):
        self.llm = get_llm()

    def rewrite(self, question: str) -> list[str]:
        """Decompose a question into sub-queries. Falls back to original question on failure."""
        try:
            prompt = REWRITE_PROMPT.replace("{question}", question)
            messages = [{"role": "user", "content": prompt}]
            raw = self.llm.generate_json(messages, max_tokens=1024)
            data = json.loads(raw)
            sub_queries = data.get("sub_queries", [question])
            return sub_queries if sub_queries else [question]
        except Exception:
            return [question]

    def rewrite_iterative(self, question: str, previous_results: list[str]) -> str:
        """Generate a refined query based on what we've already found.
        Used when the agent decides more information is needed."""
        context = "\n".join(previous_results[-3:])  # Last 3 observations
        prompt = f"""Based on the original question and what we've already found, generate ONE new search query to find missing information.

Original question: {question}

Already found:
{context}

Generate a single, focused search query that fills the remaining information gap. Output only the query text, nothing else."""

        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate(messages, max_tokens=512).strip().strip('"')
