"""Real PostgreSQL + pgvector tests; deterministic vectors isolate access logic."""

import datetime as dt
from pathlib import Path

import numpy as np
from sqlalchemy import text

from sqm_ai.assistant.access import AssistantUser
from sqm_ai.assistant.retrieve import load_chunks
from sqm_ai.retrieval.embed import EMBED_SPACE
from sqm_ai.retrieval.fuse import hybrid
from sqm_ai.retrieval.store import connect, insert_chunks, search


def test_dense_sparse_and_content_fetch_share_access_and_asof(db):
    sql = (
        Path(__file__).resolve().parents[2] / "sql/doc_chunks.sql"
    )
    with db.begin() as c:
        for statement in "\n".join(
            line
            for line in sql.read_text().splitlines()
            if not line.lstrip().startswith("--")
        ).split(";"):
            if statement.strip():
                c.execute(text(statement))
    query = np.zeros(1024, dtype=np.float32)
    query[0] = 1
    base = {
        "document_revision": "v1",
        "source_type": "manual",
        "chunk_index": 0,
        "clause": "8.4.2",
        "token_count": 10,
        "token_model": "test-only",
        "embedding": query,
        "embed_model": EMBED_SPACE,
        "valid_from": dt.date(2025, 1, 1),
        "valid_to": None,
    }
    rows = []
    for doc, metadata, content, overrides in [
        (
            "visible",
            {"visibility": "program", "program": "NL-KESTREL"},
            "Visible 7741-B procedure",
            {},
        ),
        (
            "hidden",
            {"visibility": "program", "program": "NL-MERLIN"},
            "Secret 7741-B 7741-B 7741-B",
            {},
        ),
        (
            "shared",
            {"visibility": "shared"},
            "Shared policy for all authorized manual readers",
            {},
        ),
        (
            "future",
            {"visibility": "shared"},
            "Future 7741-B",
            {"valid_from": dt.date(2027, 1, 1)},
        ),
        (
            "expired",
            {"visibility": "shared"},
            "Old 7741-B",
            {"valid_to": dt.date(2026, 1, 1)},
        ),
        (
            "wrongspace",
            {"visibility": "shared"},
            "Other embedding 7741-B",
            {"embed_model": "different-model"},
        ),
    ]:
        rows.append(
            base
            | {
                "document_id": doc,
                "metadata": metadata,
                "content": content,
            }
            | overrides
        )
    user = AssistantUser("test", ("NL-KESTREL",))
    asof = dt.date(2026, 9, 1)
    with connect(
        db.url.render_as_string(hide_password=False)
    ) as conn:
        insert_chunks(conn, rows)
        dense = search(
            conn,
            query,
            user.allowed_source_types,
            user.allowed_programs,
            k=10,
            asof=asof,
        )
        assert {r["document_id"] for r in dense} == {
            "visible",
            "shared",
        }
        ids = hybrid(
            "7741-B", query, conn, n=10, user=user, asof=asof
        )
        fetched = load_chunks(conn, ids, user=user, asof=asof)
        assert {r["document_id"] for r in fetched} == {
            "visible",
            "shared",
        }
        # Even forged/unfiltered IDs must not bypass the final content filter.
        all_ids = [
            r[0]
            for r in conn.execute("SELECT id FROM sqm.doc_chunks")
        ]
        loaded = load_chunks(conn, all_ids, user=user, asof=asof)
        assert {r["document_id"] for r in loaded} == {
            "visible",
            "shared",
        }
        none = AssistantUser("none", (), ())
        assert (
            hybrid(
                "7741-B", query, conn, n=10, user=none, asof=asof
            )
            == []
        )


def test_parent_scope_is_checked_independently(db):
    from importlib.resources import files

    from sqm_ai.retrieval.parents import small_to_big

    scripts = [
        Path(__file__).resolve().parents[2]
        / "sql/doc_chunks.sql",
        files("sqm_ai.retrieval").joinpath("parents.sql"),
    ]
    with db.begin() as c:
        for script in scripts:
            statements = "\n".join(
                line
                for line in script.read_text().splitlines()
                if not line.lstrip().startswith("--")
            )
            for statement in statements.split(";"):
                if statement.strip():
                    c.execute(text(statement))
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1
    with connect(
        db.url.render_as_string(hide_password=False)
    ) as conn:
        parent_id = conn.execute("""INSERT INTO sqm.doc_parents
        (document_id,document_revision,source_type,metadata,valid_from,content)
        VALUES ('p','v1','manual','{"visibility":"program","program":"NL-MERLIN"}',
        '2025-01-01','Restricted parent text') RETURNING id""").fetchone()[
            0
        ]
        row = {
            "document_id": "p",
            "document_revision": "v1",
            "source_type": "manual",
            "chunk_index": 0,
            "clause": "1",
            "content": "Permitted child excerpt",
            "token_count": 3,
            "token_model": "test-only",
            "embedding": vector,
            "embed_model": EMBED_SPACE,
            "metadata": {
                "visibility": "program",
                "program": "NL-KESTREL",
            },
            "valid_from": "2025-01-01",
            "valid_to": None,
        }
        insert_chunks(conn, [row])
        conn.execute(
            "UPDATE sqm.doc_chunks SET parent_id=%s", (parent_id,)
        )
        assert (
            small_to_big(
                conn,
                vector,
                AssistantUser("test", ("NL-KESTREL",)),
                asof=dt.date(2026, 9, 1),
            )
            == []
        )
