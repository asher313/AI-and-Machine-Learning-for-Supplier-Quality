# Chapter 17 — 17.9 Advanced Patterns
def guarded_search(conn, qvec, user, k: int = 50):
    """Filter by entitlement first, then search vectors."""
    return search(
        conn, qvec,
        types=user.allowed_source_types,
        programs=user.allowed_programs,   # SQL filter
        k=k,
    )
