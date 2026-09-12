SELECT id, document_id, document_revision, source_type, content, clause,
       metadata, token_count, token_model, valid_from, valid_to,
       1 - (embedding <=> %(q)s::vector) AS similarity
FROM sqm.doc_chunks
WHERE source_type = ANY(%(types)s)
  AND embed_model = %(embed_model)s
  AND (metadata->>'visibility' = 'shared' OR
       (metadata->>'visibility' = 'program' AND metadata->>'program' = ANY(%(programs)s)))
  AND valid_from <= %(asof)s
  AND (valid_to IS NULL OR valid_to > %(asof)s)
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s;
