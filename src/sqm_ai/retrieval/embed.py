"""Pinned BGE space; loading is explicit/lazy and never occurs on import."""

from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
DIMENSIONS = 1024
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
EMBED_SPACE = (
    f"{MODEL_NAME}@{MODEL_REVISION}:query-prefix-v1:normalized"
)


@lru_cache
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION
    )


def _encode(texts, *, query=False, model=None):
    model = model if model is not None else get_model()
    texts = (
        [QUERY_PREFIX + t for t in texts]
        if query
        else list(texts)
    )
    lengths = [
        len(model.tokenizer(t, truncation=False)["input_ids"])
        for t in texts
    ]
    if any(n > model.max_seq_length for n in lengths):
        raise ValueError(
            "embedding input too long; split the passage or shorten the query"
        )
    vectors = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if (
        vectors.shape != (len(texts), DIMENSIONS)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(
            "embedding dimension or values do not match configured space"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding")
    return vectors / norms


def embed_query(text, *, model=None):
    return _encode([text], query=True, model=model)[0]


def embed_corpus(texts, *, model=None):
    if not texts:
        return np.empty((0, DIMENSIONS), dtype=np.float32)
    return _encode(texts, model=model)
