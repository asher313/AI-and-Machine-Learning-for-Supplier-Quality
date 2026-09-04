-- Chapter 16 — 16.8 The Schema and the Review Queue
SELECT ncr_id, category, severity, confidence,
       supplier_id, reasoning, classified_at
FROM   sqm.ncr_classifications
WHERE  reviewed_at IS NULL AND routed = 'review'
ORDER  BY severity DESC, confidence ASC, classified_at
LIMIT  50;
