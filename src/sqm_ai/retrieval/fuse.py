"""Rank fusion with the same authorized corpus for sparse and dense paths."""

import re

from rank_bm25 import BM25Okapi

from sqm_ai.retrieval.embed import EMBED_SPACE
from sqm_ai.retrieval.store import eligible_chunks, search


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower()
    )


def build_bm25(corpus):
    return BM25Okapi([tokenize(doc) for doc in corpus])


def sparse_rank(question, rows, n=50):
    terms = tokenize(question)
    if not rows or not terms:
        return []
    tokens = [tokenize(row["content"]) for row in rows]
    if not any(tokens):
        return []
    scores = BM25Okapi(tokens).get_scores(terms)
    # Do not fill the pool with rows having no query-token overlap.
    indices = [
        i
        for i, t in enumerate(tokens)
        if set(terms).intersection(t)
    ]
    indices.sort(key=lambda i: (-float(scores[i]), rows[i]["id"]))
    return [rows[i]["id"] for i in indices[:n]]


def rrf(rankings, k=60):
    if k < 1:
        raise ValueError("positive rank constant required")
    scores = {}
    for ranking in rankings:
        seen = set()
        rank = 0
        for doc_id in ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (
                k + rank
            )
    return sorted(scores, key=lambda i: (-scores[i], i))


def hybrid(
    question,
    qvec,
    conn,
    n=50,
    *,
    user,
    asof=None,
    ef_search=100,
    embed_model=EMBED_SPACE,
):
    """Teaching baseline rebuilds BM25 over eligible rows per query.

    Production needs a revision-aware authorized sparse index, not a global
    unfiltered index. A final scoped content load remains mandatory.
    """
    dense = [
        r["id"]
        for r in search(
            conn,
            qvec,
            user.allowed_source_types,
            user.allowed_programs,
            k=n,
            asof=asof,
            ef_search=ef_search,
            embed_model=embed_model,
        )
    ]
    rows = eligible_chunks(
        conn,
        user.allowed_source_types,
        user.allowed_programs,
        asof,
        embed_model=embed_model,
    )
    sparse = sparse_rank(question, rows, n)
    return rrf([dense, sparse])[:n]
