"""Parent expansion authorizes the parent independently before reading its text."""

import datetime as dt

from sqm_ai.retrieval.store import search

PARENTS = """
SELECT c.id AS child_id, p.id AS parent_id, p.content
FROM sqm.doc_chunks c JOIN sqm.doc_parents p ON p.id=c.parent_id
WHERE c.id=ANY(%(ids)s)
  AND p.document_id=c.document_id AND p.document_revision=c.document_revision
  AND p.source_type=ANY(%(types)s)
  AND (p.metadata->>'visibility'='shared' OR
       (p.metadata->>'visibility'='program' AND p.metadata->>'program'=ANY(%(programs)s)))
  AND p.valid_from<=%(asof)s AND (p.valid_to IS NULL OR p.valid_to>%(asof)s)
"""


def small_to_big(conn, qvec, user, k=5, asof=None):
    if k < 1 or k > 250:
        raise ValueError("k must be in [1,250]")
    asof = asof or dt.datetime.now(dt.UTC).date()
    children = search(
        conn,
        qvec,
        user.allowed_source_types,
        user.allowed_programs,
        k=k * 4,
        asof=asof,
    )
    with conn.cursor() as cur:
        cur.execute(
            PARENTS,
            {
                "ids": [c["id"] for c in children],
                "types": user.allowed_source_types,
                "programs": user.allowed_programs,
                "asof": asof,
            },
        )
        rows = {row[0]: (row[1], row[2]) for row in cur}
    seen, out = set(), []
    for child in children:
        row = rows.get(child["id"])
        if row is not None and row[0] not in seen:
            seen.add(row[0])
            out.append(row[1])
        if len(out) == k:
            break
    return out
