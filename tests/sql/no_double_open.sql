-- Must return zero rows: revisions may not overlap within one embedding space.
SELECT DISTINCT a.document_id, a.embed_model,
       a.document_revision AS earlier_key, b.document_revision AS later_key
FROM sqm.doc_chunks a JOIN sqm.doc_chunks b
  ON a.document_id=b.document_id AND a.embed_model=b.embed_model
 AND a.document_revision<b.document_revision
WHERE daterange(a.valid_from,a.valid_to,'[)') && daterange(b.valid_from,b.valid_to,'[)');
