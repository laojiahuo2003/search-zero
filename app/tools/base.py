"""
Base classes for tools and documents.
"""
from dataclasses import dataclass, field


@dataclass
class Document:
    """Unified document representation across all retrieval sources."""
    title: str = ""
    content: str = ""
    url: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"title": self.title, "content": self.content, "url": self.url, "score": self.score}


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    documents: list[Document] = field(default_factory=list)
    error: str = ""
    metadata: dict = field(default_factory=dict)
