-- sql/chunk_search.sql. One statement, and only one:
-- search() issues "SET LOCAL hnsw.ef_search = %s" ahead
-- of it, in the same transaction.
--
-- Chapter 17.5 wrote the statement; Chapter 18.5 replaced
-- Chapter 17's single `valid_to IS NULL` line with the
-- as-of window. Ruling R12: Chapter 18's version is
-- canonical.
SELECT id, content, clause, metadata,
       1 - (embedding <=> %(q)s::vector) AS similarity
FROM   sqm.doc_chunks
WHERE  source_type = ANY(%(types)s)
  AND  (metadata->>'program' IS NULL
        OR metadata->>'program' = ANY(%(programs)s))
  AND  valid_from <= %(asof)s                    -- as-of
  AND  (valid_to IS NULL OR valid_to > %(asof)s) -- as-of
ORDER  BY embedding <=> %(q)s::vector
LIMIT  %(k)s;
