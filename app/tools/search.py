"""
DuckDuckGo and Tavily search providers.
"""
import json
import urllib.request
import urllib.parse
from typing import Optional
from app.tools.base import Document, ToolResult
from app.utils.config import get_config


class DuckDuckGoSearch:
    """Free web search using the duckduckgo_search library."""

    @staticmethod
    def search(query: str, max_results: int = 5) -> list[Document]:
        try:
            from ddgs import DDGS
            raw = list(DDGS().text(query, max_results=max_results))
        except Exception:
            return DuckDuckGoSearch._fallback(query, max_results)

        docs = []
        for r in raw:
            docs.append(Document(
                title=r.get("title", ""),
                content=r.get("body", ""),
                url=r.get("href", ""),
            ))
        return docs[:max_results]

    @staticmethod
    def _fallback(query: str, max_results: int) -> list[Document]:
        """Fallback to Instant Answer API if library fails."""
        results = []
        try:
            params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
            url = f"https://api.duckduckgo.com/?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data.get("AbstractText"):
                    results.append(Document(
                        title=data.get("Heading", ""),
                        content=data["AbstractText"],
                        url=data.get("AbstractURL", ""),
                    ))
                for item in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(item, dict) and item.get("Text"):
                        results.append(Document(
                            title=item.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            content=item["Text"],
                            url=item.get("FirstURL", ""),
                        ))
        except Exception:
            pass
        return results[:max_results]


class TavilySearch:
    """Tavily search API wrapper (requires API key)."""

    @staticmethod
    def search(query: str, api_key: str, max_results: int = 5) -> list[Document]:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        docs = []
        for r in data.get("results", [])[:max_results]:
            docs.append(Document(
                title=r.get("title", ""),
                content=r.get("content", ""),
                url=r.get("url", ""),
                score=r.get("score", 0.0),
            ))
        return docs


class SearchTool:
    """Unified search interface — selects provider based on config."""

    def __init__(self):
        cfg = get_config()
        self.provider = cfg.search_provider
        self.max_results = cfg.search_max_results
        self.tavily_key = cfg.tavily_api_key

    def search(self, query: str) -> ToolResult:
        try:
            if self.provider == "tavily" and self.tavily_key:
                docs = TavilySearch.search(query, self.tavily_key, self.max_results)
            else:
                docs = DuckDuckGoSearch.search(query, self.max_results)
            return ToolResult(success=True, documents=docs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def format_observation(self, result: ToolResult) -> str:
        if not result.success:
            return f"Search failed: {result.error}"

        lines = []
        for i, doc in enumerate(result.documents, 1):
            lines.append(f"[{i}] {doc.title}")
            lines.append(f"    {doc.content[:300]}")
            if doc.url:
                lines.append(f"    URL: {doc.url}")
            lines.append("")
        return "\n".join(lines) if lines else "No results found."
