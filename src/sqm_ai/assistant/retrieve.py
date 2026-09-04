# src/sqm_ai/assistant/retrieve.py
import datetime as dt

import psycopg

from sqm_ai.retrieval.embed import embed_query
from sqm_ai.retrieval.fuse import hybrid
from sqm_ai.retrieval.rerank import rerank
from sqm_ai.retrieval.store import connect

LOAD = """
SELECT id, document_id, source_type, clause, content
  FROM sqm.doc_chunks
 WHERE id = ANY(%s)
"""


def load_chunks(conn, ids: list[int]) -> list[dict]:
    """Rows for the fused ids, kept in the fused order."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(LOAD, (ids,))
        by_id = {r["id"]: r for r in cur.fetchall()}
    return [by_id[i] for i in ids if i in by_id]


def fetch_chunks(
    question: str, user, asof: dt.date | None = None,
    pool: int = 50, keep: int = 8,
) -> list[dict]:
    """The only way Build 4 retrieves anything."""
    asof = asof or dt.date.today()
    ef = 200 if len(user.programs) < 3 else 100
    with connect() as conn:
        ids = hybrid(
            question, embed_query(question), conn,
            n=pool, user=user, asof=asof, ef_search=ef,
        )
        rows = load_chunks(conn, ids)
    return rerank(question, rows, k=keep)
