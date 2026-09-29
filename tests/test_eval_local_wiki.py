"""
Tests for local-index evaluation mode in eval_with_real_wiki.py.

The eval script now defaults to --wiki_mode local, which uses the same
LocalWikiSearcher backend as training (data/wiki_index.json). These tests
verify: (1) the local search path returns the same OBSERVATION format as the
training backend, (2) the HTTP proxy path is preserved for --wiki_mode url.
"""
import json

import pytest

from scripts.eval_with_real_wiki import wiki_search, LocalWikiSearcher


@pytest.fixture
def index_path(tmp_path):
    articles = [
        {
            "title": "Albert Einstein",
            "sentences": [
                "Albert Einstein was a German-born theoretical physicist.",
                "He developed the theory of relativity.",
            ],
            "full_text": "Albert Einstein was a German-born theoretical physicist. "
                         "He developed the theory of relativity.",
        },
        {
            "title": "Python (programming language)",
            "sentences": [
                "Python is a high-level programming language.",
                "It was created by Guido van Rossum.",
            ],
            "full_text": "Python is a high-level programming language. "
                         "It was created by Guido van Rossum.",
        },
    ]
    p = tmp_path / "wiki_index.json"
    p.write_text(json.dumps(articles), encoding="utf-8")
    return str(p)


def test_local_searcher_loads_and_searches(index_path):
    searcher = LocalWikiSearcher(index_path)
    stats = searcher.get_stats()
    assert stats["articles"] == 2

    result = searcher.search("Einstein relativity")
    assert result.startswith("OBSERVATION:\n")
    assert "[1] Albert Einstein" in result
    assert "theoretical physicist" in result


def test_local_searcher_empty_query(index_path):
    searcher = LocalWikiSearcher(index_path)
    assert searcher.search("   ").startswith("OBSERVATION: Empty search query.")


def test_wiki_search_local_matches_training_backend(index_path):
    """wiki_search with local_searcher must return exactly what the training
    backend (LocalWikiSearcher.search) returns."""
    searcher = LocalWikiSearcher(index_path)
    for query in ["Einstein", "Python programming", "no such thing exists"]:
        expected = searcher.search(query, top_k=3, sentences=3)
        got = wiki_search(query, "", top_k=3, sentences=3, local_searcher=searcher)
        assert got == expected


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_wiki_search_url_branch(monkeypatch):
    """HTTP path (--wiki_mode url) must keep the original OBSERVATION format."""
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({
            "results": [
                {"rank": 1, "title": "T1", "summary": "S1", "url": "U1"},
                {"rank": 2, "title": "T2", "summary": "S2", "url": "U2"},
            ]
        })

    monkeypatch.setattr("scripts.eval_with_real_wiki.requests.get", fake_get)
    got = wiki_search("hello", "http://127.0.0.1:18080/search")

    assert captured["url"] == "http://127.0.0.1:18080/search"
    assert captured["params"] == {"q": "hello", "top_k": 3, "sentences": 3}
    assert got == ("OBSERVATION:\n"
                   "[1] T1\n    S1\n    URL: U1\n\n"
                   "[2] T2\n    S2\n    URL: U2")


def test_wiki_search_url_empty_results(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse({"results": []})

    monkeypatch.setattr("scripts.eval_with_real_wiki.requests.get", fake_get)
    got = wiki_search("hello", "http://127.0.0.1:18080/search")
    assert got == 'OBSERVATION: No results found for "hello".'


def test_wiki_search_url_error(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("scripts.eval_with_real_wiki.requests.get", fake_get)
    got = wiki_search("hello", "http://127.0.0.1:18080/search")
    assert got.startswith("OBSERVATION: Search failed:")
