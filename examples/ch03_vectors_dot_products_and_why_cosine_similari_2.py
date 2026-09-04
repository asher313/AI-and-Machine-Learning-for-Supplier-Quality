# Chapter 3 — 3.1 Vectors, Dot Products, and Why Cosine Similarity Is a Dot Product
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: direction only, magnitude ignored."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("zero-length vector")
    return float(a @ b / denom)
