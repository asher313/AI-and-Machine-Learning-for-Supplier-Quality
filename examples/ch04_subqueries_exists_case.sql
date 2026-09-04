-- Chapter 4 — 4.5 Subqueries, EXISTS, CASE
-- Each audit score against the fleet-wide average
SELECT supplier_id, audit_date, audit_score,
       audit_score - (SELECT AVG(audit_score) FROM sqm.audits)
         AS delta_from_avg
FROM sqm.audits;
