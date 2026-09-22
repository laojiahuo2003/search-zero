"""
Tests for the Search-R1 Mini agent components.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.agent.state import AgentState, AgentStep
from app.tools.base import Document, ToolResult
from app.evaluation.metrics import normalize_text, exact_match, contains_match, f1_score_tokens


class TestAgentState:
    def test_initial_state(self):
        state = AgentState(question="What is Python?")
        assert state.question == "What is Python?"
        assert state.step_count == 0
        assert len(state.steps) == 0
        assert state.final_answer == ""

    def test_add_step(self):
        state = AgentState(question="test")
        step = AgentStep(step_num=1, thought="thinking", action="SEARCH", action_query="test query")
        state.add_step(step)
        assert state.step_count == 1
        assert len(state.steps) == 1

    def test_to_trace(self):
        state = AgentState(question="test")
        doc = Document(title="Test", content="Test content", url="http://test.com")
        step = AgentStep(
            step_num=1, thought="thinking", action="SEARCH",
            action_query="test query", observation="found results",
            documents=[doc],
        )
        state.add_step(step)
        trace = state.to_trace()
        assert len(trace) == 1
        assert trace[0]["step"] == 1
        assert trace[0]["thought"] == "thinking"
        assert len(trace[0]["documents"]) == 1


class TestDocument:
    def test_document_creation(self):
        doc = Document(title="T", content="C", url="U", score=0.9)
        assert doc.title == "T"
        assert doc.score == 0.9

    def test_document_to_dict(self):
        doc = Document(title="T", content="C", url="U")
        d = doc.to_dict()
        assert d["title"] == "T"
        assert d["content"] == "C"


class TestMetrics:
    def test_normalize_text(self):
        assert normalize_text("Hello World!") == "hello world"
        assert normalize_text("  Extra   Spaces  ") == "extra spaces"

    def test_exact_match(self):
        assert exact_match("Paris", "paris")
        assert exact_match("The capital is Paris.", "the capital is paris")
        assert not exact_match("Paris", "London")

    def test_contains_match(self):
        assert contains_match("The capital of France is Paris", "Paris")
        assert contains_match("Paris", "The capital of France is Paris")
        assert not contains_match("Paris", "London")

    def test_f1_score_tokens(self):
        assert f1_score_tokens("cat dog", "cat dog") == 1.0
        assert f1_score_tokens("cat", "dog") == 0.0
        score = f1_score_tokens("cat dog", "cat bird")
        assert 0.0 < score < 1.0


class TestToolResult:
    def test_success_result(self):
        docs = [Document(title="T", content="C")]
        result = ToolResult(success=True, documents=docs)
        assert result.success
        assert len(result.documents) == 1

    def test_error_result(self):
        result = ToolResult(success=False, error="timeout")
        assert not result.success
        assert result.error == "timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
