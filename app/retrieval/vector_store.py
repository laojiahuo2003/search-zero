"""
FAISS vector store with add / search / reset.
"""
import os
import pickle
import numpy as np
import faiss
from app.tools.base import Document
from app.retrieval.embedder import Embedder


class VectorStore:
    """FAISS-based vector store with document storage."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index: faiss.IndexFlatIP | None = None  # Inner Product (cosine on normalized vecs)
        self.documents: list[Document] = []
        self._init_index()

    def _init_index(self):
        self.index = faiss.IndexFlatIP(self.embedder.dim)

    def add(self, documents: list[Document]) -> int:
        """Embed and index documents. Returns number added."""
        if not documents:
            return 0
        texts = [f"{d.title}\n{d.content}" for d in documents]
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings)
        self.documents.extend(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """Search for top-k most similar documents. Returns (doc, score) pairs."""
        if self.index.ntotal == 0:
            return []
        q_emb = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                doc.score = float(score)
                results.append((doc, float(score)))
        return results

    def reset(self):
        """Clear the index and stored documents."""
        self._init_index()
        self.documents = []

    def save(self, path: str):
        """Persist index and documents to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        """Load index and documents from disk."""
        if not os.path.exists(f"{path}.index"):
            return False
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", "rb") as f:
            self.documents = pickle.load(f)
        return True
