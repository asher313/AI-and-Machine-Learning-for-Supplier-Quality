"""Consistent SQL authorization, validity, and embedding-space predicates."""

import datetime as dt
from importlib.resources import files

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sqlalchemy.engine import make_url

from sqm_ai.retrieval.embed import DIMENSIONS, EMBED_SPACE
from sqm_ai.settings import get_settings

SEARCH = (
    files("sqm_ai.retrieval")
    .joinpath("chunk_search.sql")
    .read_text()
)
# The same predicate governs dense, sparse, and final content loading.
SCOPE = SEARCH[SEARCH.index("WHERE ") : SEARCH.index("ORDER BY ")]
COLUMNS = "id, document_id, document_revision, source_type, clause, content, metadata, token_count, token_model, valid_from, valid_to"


def connect(url=None):
    url = make_url(url or get_settings().database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("PostgreSQL URL required")
    # psycopg accepts a PostgreSQL URI, not SQLAlchemy's +psycopg scheme.
    dsn = url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    conn = psycopg.connect(dsn)
    try:
        register_vector(conn)
        conn.commit()  # registration may have opened an implicit transaction
        return conn
    except Exception:
        conn.close()
        raise


def _params(types, programs, asof, embed_model):
    return {
        "types": list(types),
        "programs": list(programs),
        "asof": asof or dt.datetime.now(dt.UTC).date(),
        "embed_model": embed_model,
    }


def insert_chunks(conn, rows):
    """Caller owns transaction; duplicate revision/index/space fails explicitly."""
    if not rows:
        return
    columns = [
        "document_id",
        "document_revision",
        "source_type",
        "chunk_index",
        "clause",
        "content",
        "token_count",
        "token_model",
        "embedding",
        "embed_model",
        "metadata",
        "valid_from",
        "valid_to",
    ]
    prepared = []
    for row in rows:
        row = dict(row)
        vector = np.asarray(row["embedding"], dtype=np.float32)
        if (
            vector.shape != (DIMENSIONS,)
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0
        ):
            raise ValueError("invalid vector")
        row["embedding"] = vector
        if not isinstance(row["metadata"], Jsonb):
            row["metadata"] = Jsonb(row["metadata"])
        prepared.append(row)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO sqm.doc_chunks ({','.join(columns)}) VALUES ({','.join('%(' + c + ')s' for c in columns)})",
            prepared,
        )


def search(
    conn,
    qvec,
    types,
    programs,
    k=50,
    asof=None,
    ef_search=100,
    *,
    embed_model=EMBED_SPACE,
):
    if not 1 <= k <= 1000 or not 1 <= ef_search <= 1000:
        raise ValueError("k and ef_search must be in [1,1000]")
    qvec = np.asarray(qvec, dtype=np.float32)
    if (
        qvec.shape != (DIMENSIONS,)
        or not np.isfinite(qvec).all()
        or np.linalg.norm(qvec) == 0
    ):
        raise ValueError(
            "query vector incompatible with configured dimension"
        )
    args = _params(types, programs, asof, embed_model) | {
        "q": qvec,
        "k": k,
    }
    with (
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(ef_search),),
        )
        cur.execute(
            "SELECT set_config('hnsw.iterative_scan', 'strict_order', true)"
        )
        cur.execute(SEARCH, args)
        return cur.fetchall()


def eligible_chunks(
    conn,
    types,
    programs,
    asof=None,
    *,
    embed_model=EMBED_SPACE,
    ids=None,
):
    args = _params(types, programs, asof, embed_model)
    query = f"SELECT {COLUMNS} FROM sqm.doc_chunks " + SCOPE
    if ids is not None:
        query += " AND id = ANY(%(ids)s)"
        args["ids"] = list(ids)
    query += " ORDER BY id"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, args)
        rows = cur.fetchall()
    if ids is None:
        return rows
    by_id = {row["id"]: row for row in rows}
    return [by_id[i] for i in ids if i in by_id]
