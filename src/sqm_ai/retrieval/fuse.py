# src/sqm_ai/retrieval/fuse.py
import numpy as np
from rank_bm25 import BM25Okapi

from sqm_ai.retrieval.store import search

# Module state, built once by scripts/index_corpus.py and
# installed at start-up: the sparse index, and the chunk
# ids it was built over, in the same order.
bm25: BM25Okapi | None = None
corpus_ids: list[int] = []


def build_bm25(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi([doc.lower().split() for doc in corpus])


def install_bm25(index: BM25Okapi, ids: list[int]) -> None:
    """Make the sparse index available to this process."""
    global bm25, corpus_ids
    bm25, corpus_ids = index, ids


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked id lists. Ranks only, never scores."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (
                k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid(
    question, qvec, conn, n=50, *,
    user, asof=None, ef_search=100,
):
    """Dense top-n and sparse top-n, fused by RRF."""
    dense = [r["id"] for r in search(
        conn, qvec, user.allowed_source_types,
        user.allowed_programs, k=n,
        asof=asof, ef_search=ef_search)]
    sparse_scores = bm25.get_scores(question.lower().split())
    order = np.argsort(sparse_scores)[::-1][:n]
    sparse = [corpus_ids[i] for i in order]
    return rrf([dense, sparse])[:n]
