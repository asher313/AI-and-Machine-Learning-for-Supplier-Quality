-- Chapter 21 — Asher at Northlake
-- blocks by code, last 30 days
SELECT unnest(block_codes) AS code, count(*) AS n
FROM llm_audit
WHERE blocked AND ts >= now() - interval '30 days'
GROUP BY code ORDER BY n DESC;
