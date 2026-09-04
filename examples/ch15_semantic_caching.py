# Chapter 15 — 15.8 Semantic Caching
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """Return a stored answer when a query is near a past one."""

    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.keys = np.empty((0, 384), dtype=np.float32)
        self.values: list[object] = []

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(
            [text], normalize_embeddings=True
        )[0]

    def get(self, query: str):
        if not self.values:
            return None
        # cosine, because the vectors are normalized:
        sims = self.keys @ self._embed(query)
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self.values[best]
        return None

    def put(self, query: str, value) -> None:
        self.keys = np.vstack([self.keys, self._embed(query)])
        self.values.append(value)
