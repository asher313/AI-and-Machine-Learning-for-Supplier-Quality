# src/sqm_ai/retrieval/embed.py
#
# Chapter 17.4 describes this module in prose rather than in a
# listing: "The chosen model is wrapped once, in
# src/sqm_ai/retrieval/embed.py, which loads it at import and
# exposes embed_query(text) and embed_corpus(texts). Every
# later listing goes through those two functions, so the
# 'same model for query and corpus' rule is enforced by there
# being one place the model is named."
#
# TODO(book): described but not listed in Chapter 17.4 —
# review before production use.
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # 384 dimensions
_model = SentenceTransformer(MODEL_NAME)


def embed_query(text: str) -> np.ndarray:
    """One query string -> one normalized 384-vector."""
    return _model.encode(text, normalize_embeddings=True)


def embed_corpus(texts: list[str]) -> np.ndarray:
    """Many chunks -> an (n, 384) matrix, same model."""
    return _model.encode(
        texts, normalize_embeddings=True,
        batch_size=64, show_progress_bar=False,
    )
