"""
BGE embedding model wrapper using sentence-transformers.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.config import get_config


class Embedder:
    """Lazy-loaded BGE embedding model."""

    def __init__(self, model_name: str | None = None):
        cfg = get_config()
        self.model_name = model_name or cfg.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings (L2-normalized)."""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.embed([query])[0]
