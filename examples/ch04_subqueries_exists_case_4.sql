-- Chapter 4 — 4.5 Subqueries, EXISTS, CASE
-- Simple CASE: a tier weight
SELECT supplier_id,
       CASE tier
         WHEN 'A' THEN 1.0
         WHEN 'B' THEN 0.7
         WHEN 'C' THEN 0.4
       END AS tier_weight
FROM sqm.suppliers;

-- Searched CASE: a priority label
SELECT ncr_id,
       CASE
         WHEN severity >= 4 AND cost_impact_usd > 50000
           THEN 'CRITICAL'
         WHEN severity >= 3 THEN 'HIGH'
         WHEN severity >= 2 THEN 'MEDIUM'
         ELSE 'LOW'
       END AS priority
FROM sqm.ncrs;
