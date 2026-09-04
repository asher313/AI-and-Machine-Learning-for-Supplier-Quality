-- tests/sql/no_double_open.sql — must return zero rows
SELECT document_id, chunk_index, count(*)
  FROM sqm.doc_chunks
 WHERE valid_to IS NULL
 GROUP BY document_id, chunk_index
HAVING count(*) > 1;
