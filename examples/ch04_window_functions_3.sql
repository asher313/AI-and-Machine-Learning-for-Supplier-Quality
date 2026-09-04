-- Chapter 4 — 4.3 Window Functions
-- Audit score change since the previous audit
SELECT supplier_id, audit_date, audit_score,
       LAG(audit_score) OVER (
         PARTITION BY supplier_id ORDER BY audit_date
       ) AS prev_score,
       audit_score - LAG(audit_score) OVER (
         PARTITION BY supplier_id ORDER BY audit_date
       ) AS delta
FROM sqm.audits;
