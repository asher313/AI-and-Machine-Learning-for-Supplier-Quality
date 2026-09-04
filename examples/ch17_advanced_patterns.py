# Chapter 17 — 17.9 Advanced Patterns
PARENTS = """
SELECT c.id AS child_id, p.id AS parent_id, p.content
  FROM sqm.doc_chunks c
  JOIN sqm.doc_parents p ON p.id = c.parent_id
 WHERE c.id = ANY(%s)
"""


def small_to_big(conn, qvec, user, k: int = 5) -> list[str]:
    """Match small children; return their large parents."""
    children = search(
        conn, qvec, user.allowed_source_types,
        user.allowed_programs, k=k * 4)
    with conn.cursor() as cur:
        cur.execute(PARENTS, ([c["id"] for c in children],))
        rows = {r[0]: (r[1], r[2]) for r in cur}
    seen, out = set(), []
    for child in children:
        row = rows.get(child["id"])
        if row is None or row[0] in seen:
            continue
        seen.add(row[0])
        out.append(row[1])
        if len(out) == k:
            break
    return out
