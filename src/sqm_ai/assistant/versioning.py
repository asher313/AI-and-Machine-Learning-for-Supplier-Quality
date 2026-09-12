"""Atomic forward revision replacement within one embedding space."""

import datetime as dt

from sqm_ai.retrieval.store import insert_chunks


def replace_revision(
    conn, document_id, effective, new_rows, *, embed_model
):
    """Serialize writers for a document; preserve history and roll back on failure.

    Caller may own an outer transaction. Backdated corrections require a separate
    reviewed migration; this helper accepts only forward replacements.
    """
    if not isinstance(effective, dt.date) or not new_rows:
        raise ValueError(
            "effective date and nonempty replacement required"
        )
    revisions = {r["document_revision"] for r in new_rows}
    indices = [r["chunk_index"] for r in new_rows]
    if (
        len(revisions) != 1
        or len(indices) != len(set(indices))
        or sorted(indices) != list(range(len(indices)))
    ):
        raise ValueError(
            "one complete revision with contiguous unique chunk indices required"
        )
    for row in new_rows:
        start = (
            dt.date.fromisoformat(row["valid_from"])
            if isinstance(row["valid_from"], str)
            else row["valid_from"]
        )
        if (
            row["document_id"] != document_id
            or row["embed_model"] != embed_model
            or start != effective
            or row.get("valid_to") is not None
        ):
            raise ValueError(
                "replacement rows do not match document, space, or interval"
            )
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (document_id,),
        )
        cur.execute(
            "SELECT max(valid_from) FROM sqm.doc_chunks WHERE document_id=%s AND embed_model=%s",
            (document_id, embed_model),
        )
        latest = cur.fetchone()[0]
        if latest is not None and effective <= latest:
            raise ValueError(
                "replacement must start after all recorded revisions in this space"
            )
        cur.execute(
            "UPDATE sqm.doc_chunks SET valid_to=%s WHERE document_id=%s AND embed_model=%s AND valid_to IS NULL",
            (effective, document_id, embed_model),
        )
        closed = cur.rowcount
        insert_chunks(conn, new_rows)
        return closed
