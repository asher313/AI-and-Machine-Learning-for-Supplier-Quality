"""Optional pinned cross-encoder; its scores are not ground-truth relevance."""

from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-reranker-large"
MODEL_REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME, revision=MODEL_REVISION)


def rerank(question, shortlist, k=8, *, model=None):
    if k < 1:
        raise ValueError("positive k required")
    if not shortlist:
        return []
    model = model if model is not None else get_reranker()
    pairs = [(question, c["content"]) for c in shortlist]
    scores = np.asarray(model.predict(pairs)).reshape(-1)
    if (
        len(scores) != len(shortlist)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("invalid reranker scores")
    return [
        c
        for c, _ in sorted(
            zip(shortlist, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:k]
    ]
