# src/sqm_ai/retrieval/store.py
import datetime as dt

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from sqm_ai.settings import get_settings

SEARCH = open("sql/chunk_search.sql").read()


def connect() -> psycopg.Connection:
    conn = psycopg.connect(get_settings().database_url)
    register_vector(conn)
    return conn


def insert_chunks(conn, rows: list[dict]) -> None:
    """Bulk insert. rows carry 'embedding' as np.ndarray."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO sqm.doc_chunks (document_id, "
            "source_type, chunk_index, clause, content, "
            "token_count, embedding, embed_model, metadata, "
            "valid_from) VALUES (%(document_id)s, "
            "%(source_type)s, %(chunk_index)s, %(clause)s, "
            "%(content)s, %(token_count)s, %(embedding)s, "
            "%(embed_model)s, %(metadata)s, %(valid_from)s)",
            rows,
        )
    conn.commit()


def search(
    conn, qvec: np.ndarray, types: list[str],
    programs: list[str], k: int = 50,
    asof: dt.date | None = None,
    ef_search: int = 100,
) -> list[dict]:
    """Top-k chunks the caller is allowed to see."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SET LOCAL hnsw.ef_search = %s", (ef_search,))
        cur.execute(SEARCH, {
            "q": qvec, "types": types,
            "programs": programs, "k": k,
            # read by the as-of clause Section 18.7 adds
            "asof": asof or dt.date.today(),
        })
        return cur.fetchall()
