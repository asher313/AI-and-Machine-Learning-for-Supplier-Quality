-- Chapter 4 — 4.3 Window Functions
-- Top-3 worst NCRs per supplier
SELECT supplier_id, ncr_id, severity, cost_impact_usd
FROM (
  SELECT supplier_id, ncr_id, severity, cost_impact_usd,
         ROW_NUMBER() OVER (
           PARTITION BY supplier_id
           ORDER BY severity DESC, cost_impact_usd DESC
         ) AS rn
  FROM sqm.ncrs
) ranked
WHERE rn <= 3;
