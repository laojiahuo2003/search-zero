"""
Wikipedia Search Utility with Disk Cache
========================================
Provides search and summary retrieval for GRPO training loop.
Caches results to disk to survive restarts and avoid repeated API calls.

Usage:
    searcher = CachedWikiSearcher(cache_dir="wiki_cache")
    results = searcher.search("Python programming language")
    # Returns formatted OBSERVATION string
"""

import hashlib
import json
import os
import time
from pathlib import Path

try:
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False
    print("[WARN] wikipedia package not installed. Run: pip install wikipedia")


class CachedWikiSearcher:
    """Wikipedia search with in-memory + disk cache.

    Handles disambiguation, rate limiting, and formats results
    as OBSERVATION blocks suitable for ReAct trajectories.
    """

    def __init__(self, cache_dir: str = "wiki_cache", rate_limit: float = 0.1):
        """
        Args:
            cache_dir: Directory for disk cache (survives restarts)
            rate_limit: Minimum seconds between API calls
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_call = 0.0

        # In-memory cache for speed
        self._mem_cache: dict[str, str] = {}

        # Load existing disk cache into memory
        self._load_disk_cache()

    def _cache_key(self, query: str) -> str:
        """Generate a short filename-safe key from query."""
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return h

    def _cache_path(self, query: str) -> Path:
        return self.cache_dir / f"{self._cache_key(query)}.json"

    def _rate_limit_wait(self):
        """Ensure minimum interval between API calls."""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_call = time.time()

    def _load_disk_cache(self):
        """Load all cached results from disk into memory."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                query = data.get("query", "")
                result = data.get("result", "")
                if query and result:
                    self._mem_cache[query] = result
                    count += 1
            except (json.JSONDecodeError, KeyError):
                pass
        if count:
            print(f"[WikiCache] Loaded {count} cached entries from {self.cache_dir}")

    def _save_to_disk(self, query: str, result: str):
        """Save a single result to disk cache."""
        data = {"query": query, "result": result, "timestamp": time.time()}
        self._cache_path(query).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def search(self, query: str, top_k: int = 3, sentences: int = 3) -> str:
        """Search Wikipedia and return formatted OBSERVATION.

        Args:
            query: Search query string
            top_k: Number of search results to include
            sentences: Number of summary sentences per result

        Returns:
            Formatted string like:
            OBSERVATION: [1] Page Title
                Summary text...
                URL: https://en.wikipedia.org/wiki/Page_Title

            [2] Page Title 2
                Summary text...
                URL: https://en.wikipedia.org/wiki/Page_Title_2

        If search fails, returns:
            OBSERVATION: No results found for "query".
        """
        query = query.strip()

        # Check cache
        if query in self._mem_cache:
            return self._mem_cache[query]

        if not HAS_WIKIPEDIA:
            result = f'OBSERVATION: [Wikipedia unavailable — "wikipedia" package not installed]'
            self._mem_cache[query] = result
            self._save_to_disk(query, result)
            return result

        self._rate_limit_wait()

        try:
            # Step 1: Search for page titles
            search_results = wikipedia.search(query, results=top_k)

            if not search_results:
                result = f'OBSERVATION: No results found for "{query}".'
                self._mem_cache[query] = result
                self._save_to_disk(query, result)
                return result

            # Step 2: Get summaries for each result
            parts = []
            for i, title in enumerate(search_results[:top_k], 1):
                try:
                    self._rate_limit_wait()
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                    url = page.url
                    parts.append(f"[{i}] {title}\n    {summary}\n    URL: {url}")
                except DisambiguationError as e:
                    # Try the first non-disambiguation option
                    for opt in e.options[:3]:
                        try:
                            self._rate_limit_wait()
                            summary = wikipedia.summary(opt, sentences=sentences, auto_suggest=False)
                            page = wikipedia.page(opt, auto_suggest=False)
                            parts.append(f"[{i}] {opt}\n    {summary}\n    URL: {page.url}")
                            break
                        except (DisambiguationError, PageError):
                            continue
                        except Exception:
                            continue
                    else:
                        # All options failed, include the disambiguation page info
                        parts.append(f"[{i}] {title} (disambiguation page)\n    Multiple meanings available.\n    URL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                except PageError:
                    parts.append(f"[{i}] {title}\n    (Page not found)\n    URL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                except Exception as e:
                    parts.append(f"[{i}] {title}\n    (Error: {e})")

            if not parts:
                result = f'OBSERVATION: No results found for "{query}".'
            else:
                result = "OBSERVATION:\n" + "\n\n".join(parts)

        except Exception as e:
            result = f'OBSERVATION: Search failed for "{query}": {e}'

        # Cache the result
        self._mem_cache[query] = result
        self._save_to_disk(query, result)
        return result

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "memory_entries": len(self._mem_cache),
            "disk_entries": len(list(self.cache_dir.glob("*.json"))),
            "cache_dir": str(self.cache_dir),
        }


# ============================================================
# Local Wikipedia Searcher (offline fallback)
# ============================================================

class LocalWikiSearcher:
    """Local Wikipedia search using pre-built HotpotQA context index.

    No network required. Uses simple word-overlap scoring over
    Wikipedia article sentences extracted from HotpotQA.

    Usage:
        searcher = LocalWikiSearcher("/data/wiki_index.json")
        result = searcher.search("Python programming")
    """

    def __init__(self, index_path: str):
        """Load the search index from JSON file.

        Args:
            index_path: Path to wiki_index.json (built by build_wiki_index.py)
        """
        with open(index_path, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)

        # Build search structures
        self._titles = [a["title"].lower() for a in self.articles]
        self._full_texts = [a.get("full_text", "").lower() for a in self.articles]
        self._cache: dict[str, str] = {}

        print(f"[LocalWiki] Loaded {len(self.articles)} articles from {index_path}")

    def search(self, query: str, top_k: int = 3, sentences: int = 3) -> str:
        """Search local Wikipedia index and return formatted OBSERVATION.

        Args:
            query: Search query string
            top_k: Number of results to return
            sentences: Max sentences per result (in local mode, returns first N)

        Returns:
            Formatted OBSERVATION string
        """
        query = query.strip()
        if query in self._cache:
            return self._cache[query]

        query_lower = query.lower()
        query_words = set(query_lower.split())

        if not query_words:
            result = f'OBSERVATION: Empty search query.'
            self._cache[query] = result
            return result

        # Score each article by word overlap
        scores = []
        for i, (title, full_text) in enumerate(zip(self._titles, self._full_texts)):
            # Title match bonus
            title_words = set(title.split())
            title_overlap = len(query_words & title_words)

            # Content word overlap
            text_words = set(full_text.split())
            text_overlap = len(query_words & text_words)

            # Also check if query is a substring of title or text
            title_substr = 5.0 if query_lower in title else 0.0

            score = title_overlap * 3.0 + text_overlap * 1.0 + title_substr
            if score > 0:
                scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        if not scores:
            result = f'OBSERVATION: No results found for "{query}".'
            self._cache[query] = result
            return result

        # Build observation
        parts = []
        for rank, (score, idx) in enumerate(scores[:top_k], 1):
            article = self.articles[idx]
            title = article["title"]
            sents = article.get("sentences", [])

            # Get first N sentences and join them
            body_sents = sents[:sentences]
            body = " ".join(body_sents) if body_sents else "(no content)"

            # URL
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            parts.append(f"[{rank}] {title}\n    {body}\n    URL: {url}")

        if not parts:
            result = f'OBSERVATION: No results found for "{query}".'
        else:
            result = "OBSERVATION:\n" + "\n\n".join(parts)

        self._cache[query] = result
        return result

    def get_stats(self) -> dict:
        """Return searcher statistics."""
        return {
            "articles": len(self.articles),
            "cache_entries": len(self._cache),
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    searcher = CachedWikiSearcher(cache_dir="wiki_cache_test")
    print("=== Cache Stats ===")
    print(searcher.get_stats())

    print("\n=== Search: 'Python programming language' ===")
    result = searcher.search("Python programming language")
    print(result)

    print("\n=== Search: 'Albert Einstein' (should be cached) ===")
    result = searcher.search("Albert Einstein")
    print(result[:500])

    print("\n=== Cache Stats ===")
    print(searcher.get_stats())
