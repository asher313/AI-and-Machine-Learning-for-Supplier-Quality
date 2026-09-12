import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from sqm_ai.assistant.versioning import replace_revision
from sqm_ai.retrieval.embed import EMBED_SPACE
from sqm_ai.retrieval.store import connect, insert_chunks, search


def test_revision_boundary_and_failed_insert_rollback(db):
    script = (
        Path(__file__).resolve().parents[2] / "sql/doc_chunks.sql"
    )
    with db.begin() as c:
        for statement in "\n".join(
            line
            for line in script.read_text().splitlines()
            if not line.startswith("--")
        ).split(";"):
            if statement.strip():
                c.execute(text(statement))
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1
    old = {
        "document_id": "manual",
        "document_revision": "v1",
        "source_type": "manual",
        "chunk_index": 0,
        "clause": "1",
        "content": "Old fictional requirement",
        "token_count": 4,
        "token_model": "test",
        "embedding": vector,
        "embed_model": EMBED_SPACE,
        "metadata": {"visibility": "shared"},
        "valid_from": dt.date(2025, 1, 1),
        "valid_to": None,
    }
    new = old | {
        "document_revision": "v2",
        "content": "New fictional requirement",
        "valid_from": dt.date(2026, 3, 1),
    }
    with connect(
        db.url.render_as_string(hide_password=False)
    ) as conn:
        insert_chunks(conn, [old])
        conn.commit()
        assert (
            replace_revision(
                conn,
                "manual",
                dt.date(2026, 3, 1),
                [new],
                embed_model=EMBED_SPACE,
            )
            == 1
        )
        assert (
            search(
                conn,
                vector,
                ["manual"],
                [],
                asof=dt.date(2026, 2, 28),
            )[0]["document_revision"]
            == "v1"
        )
        assert (
            search(
                conn,
                vector,
                ["manual"],
                [],
                asof=dt.date(2026, 3, 1),
            )[0]["document_revision"]
            == "v2"
        )
        invalid = new | {
            "document_revision": "v3",
            "valid_from": dt.date(2026, 4, 1),
            "embedding": np.zeros(1024),
        }
        with pytest.raises(ValueError, match="invalid vector"):
            replace_revision(
                conn,
                "manual",
                dt.date(2026, 4, 1),
                [invalid],
                embed_model=EMBED_SPACE,
            )
        assert (
            search(
                conn,
                vector,
                ["manual"],
                [],
                asof=dt.date(2026, 5, 1),
            )[0]["document_revision"]
            == "v2"
        )
        with pytest.raises(ValueError, match="after all"):
            replace_revision(
                conn,
                "manual",
                dt.date(2026, 3, 1),
                [new],
                embed_model=EMBED_SPACE,
            )
