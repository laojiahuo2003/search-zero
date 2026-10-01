"""
Tests for local-index evaluation mode in eval_with_real_wiki.py.

The eval script now defaults to --wiki_mode local, which uses the same
LocalWikiSearcher backend as training (data/wiki_index.json). These tests
verify: (1) the local search path returns the same OBSERVATION format as the
training backend, (2) the HTTP proxy path is preserved for --wiki_mode url.
"""
import json

import pytest
import torch

import scripts.eval_with_real_wiki as ev
import scripts.train_grpo_search_MI300X as tr
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


# ----------------------------------------------------------------------
# Training-consistency contract: eval rollout must build the exact same
# token context as the GRPO training loop, with greedy decoding.
# ----------------------------------------------------------------------

def test_eval_reuses_training_context_functions():
    """eval must import the training helpers, never re-implement them.

    eval_with_real_wiki.py imports the training module top-level style
    (``train_grpo_search_MI300X``) while this test imports it package style
    (``scripts.train_grpo_search_MI300X``) — two module objects compiled
    from the SAME file, so identity (``is``) cannot hold. Compare source
    instead: it still fails the moment anyone re-implements a helper in
    the eval script or lets it drift from the training version.
    """
    import inspect

    for name in ("make_prompt_ids", "tokenize_observation", "extract_answer"):
        ev_fn = getattr(ev, name)
        tr_fn = getattr(tr, name)
        assert inspect.getsource(ev_fn) == inspect.getsource(tr_fn), name
        assert ev_fn.__module__.endswith("train_grpo_search_MI300X"), name
    assert ev.GRPO_SYSTEM_PROMPT == tr.GRPO_SYSTEM_PROMPT
    assert ev.MAX_TURNS == tr.MAX_TURNS


class _FakeTokenizer:
    """Whitespace-level tokenizer: enough for make_prompt_ids /
    tokenize_observation / generate_answer to run on CPU."""

    def __init__(self):
        self._vocab = {}
        self._inv = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._next = 2

    def _id(self, piece):
        if piece not in self._vocab:
            self._vocab[piece] = self._next
            self._inv[self._next] = piece
            self._next += 1
        return self._vocab[piece]

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._inv[i] for i in ids)


class _ScriptedModel:
    """Returns scripted generation texts, one per generate() call, and
    records every call's input ids + kwargs for assertions."""

    device = torch.device("cpu")

    def __init__(self, tokenizer, scripted_texts):
        self._tok = tokenizer
        self._texts = list(scripted_texts)
        self.calls = []

    def generate(self, input_tensor, **kwargs):
        self.calls.append({"input_ids": input_tensor[0].tolist(), "kwargs": kwargs})
        text = self._texts[len(self.calls) - 1]
        gen = torch.tensor([self._tok.encode(text)], dtype=input_tensor.dtype)
        return torch.cat([input_tensor, gen], dim=1)


def test_generate_answer_mirrors_training_rollout(index_path):
    """SEARCH turn: observation must be appended via tokenize_observation
    (user-turn markers), prompt via make_prompt_ids, decoding greedy."""
    searcher = LocalWikiSearcher(index_path)
    tok = _FakeTokenizer()
    turn1 = "THOUGHT: need info\nACTION: SEARCH: Einstein"
    turn2 = "THOUGHT: got it\nACTION: ANSWER: Albert Einstein [1]"
    model = _ScriptedModel(tok, [turn1, turn2])

    question = "Who developed the theory of relativity?"
    answer, turns = ev.generate_answer(model, tok, question, "", searcher)

    assert answer == "Albert Einstein [1]"
    assert len(turns) == 2
    assert len(model.calls) == 2
    # Greedy decoding for reproducibility
    for call in model.calls:
        assert call["kwargs"]["do_sample"] is False

    # Turn 1 input == training prompt, token for token
    expected_prompt = tr.make_prompt_ids(tok, tr.GRPO_SYSTEM_PROMPT, question)
    assert model.calls[0]["input_ids"] == expected_prompt

    # Turn 2 input == prompt + gen1 + observation wrapped as a user turn
    gen1_ids = tok.encode(turn1)
    obs = searcher.search("Einstein", top_k=3, sentences=3)
    expected_second = expected_prompt + gen1_ids + tr.tokenize_observation(tok, obs)
    assert model.calls[1]["input_ids"] == expected_second


def test_generate_answer_no_action_fallback(index_path):
    """No ACTION in output: stop after one turn, return raw text."""
    tok = _FakeTokenizer()
    model = _ScriptedModel(tok, ["just some plain text"])
    answer, turns = ev.generate_answer(
        model, tok, "Q?", "", LocalWikiSearcher(index_path))
    assert answer == "just some plain text"
    assert len(model.calls) == 1
