"""Scoped retrieval and a second authorization check before content leaves SQL."""

import datetime as dt

from sqm_ai.retrieval.embed import EMBED_SPACE, embed_query
from sqm_ai.retrieval.fuse import hybrid
from sqm_ai.retrieval.rerank import rerank
from sqm_ai.retrieval.store import connect, eligible_chunks


def load_chunks(
    conn, ids, *, user, asof=None, embed_model=EMBED_SPACE
):
    return eligible_chunks(
        conn,
        user.allowed_source_types,
        user.allowed_programs,
        asof,
        embed_model=embed_model,
        ids=ids,
    )


def fetch_chunks(question, user, asof=None, pool=50, keep=8):
    asof = asof or dt.datetime.now(dt.UTC).date()
    ef = 200 if len(user.allowed_programs) < 3 else 100
    with connect() as conn:
        ids = hybrid(
            question,
            embed_query(question),
            conn,
            n=pool,
            user=user,
            asof=asof,
            ef_search=ef,
        )
        rows = load_chunks(conn, ids, user=user, asof=asof)
    return rerank(question, rows, k=keep)
