"""Deterministic lexical vectors for offline software tests, not neural embeddings."""

import hashlib

import numpy as np

from sqm_ai.retrieval.fuse import tokenize

SPACE = "synthetic-lexical-sha256-v1:1024"


def hash_vectors(texts):
    vectors = np.zeros((len(texts), 1024), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode()).digest()
            vectors[
                i, int.from_bytes(digest[:4], "big") % 1024
            ] += 1
        norm = np.linalg.norm(vectors[i])
        if not norm:
            raise ValueError("no indexable tokens")
        vectors[i] /= norm
    return vectors
