# Chapter 17 teaching listing. Supply the inputs described in the text.
def guarded_search(conn, qvec, user, k: int = 50):
    """Return only rows allowed by the query's entitlement predicate."""
    return search(
        conn, qvec,
        types=user.allowed_source_types,
        programs=user.allowed_programs,   # SQL filter
        k=k,
    )
