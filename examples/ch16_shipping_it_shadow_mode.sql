-- Chapter 16 — 16.10 Shipping It: Shadow Mode
SELECT c.category = h.category AS cat_match,
       c.severity = h.severity AS sev_match,
       count(*)                AS n
FROM   sqm.ncr_classifications c
JOIN   sqm.ncrs h USING (ncr_id)
WHERE  c.classified_at >= now() - interval '30 days'
GROUP  BY 1, 2 ORDER BY n DESC;
