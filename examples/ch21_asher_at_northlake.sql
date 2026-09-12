-- blocks by code, last 30 days; count distinct traces, not lifecycle events
SELECT code, count(DISTINCT trace_id) AS n
FROM sqm.llm_audit_events
CROSS JOIN LATERAL jsonb_array_elements_text(
  COALESCE(record->'block_codes', '[]'::jsonb)
) AS codes(code)
WHERE event IN ('blocked','failed')
  AND ts >= now() - interval '30 days'
GROUP BY code ORDER BY n DESC;
