"""
BGE Reranker for refining retrieval results.
"""
from sentence_transformers import CrossEncoder
from app.tools.base import Document
from app.utils.config import get_config


class Reranker:
    """Cross-encoder reranker using BGE reranker model."""

    def __init__(self, model_name: str | None = None):
        cfg = get_config()
        self.model_name = model_name or cfg.reranker_model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        """Rerank documents by relevance to query, return top_k."""
        if not documents:
            return []

        pairs = []
        for d in documents:
            text = f"{d.title}\n{d.content}" if d.title else d.content
            pairs.append([query, text])

        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top = []
        for doc, score in scored[:top_k]:
            doc.score = float(score)
            top.append(doc)
        return top
