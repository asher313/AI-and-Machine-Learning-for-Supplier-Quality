-- Chapter 5 — 5.2 Dialect Differences That Bite
-- HANA dialect, all in one place
SELECT ADD_DAYS(CURRENT_DATE, -30) FROM DUMMY;
SELECT DAYS_BETWEEN(DATE '2025-09-01', CURRENT_DATE)
  FROM DUMMY;

-- LOCATE(haystack, needle), 1-indexed, 0 = not found
SELECT ncr_id, description
  FROM SQM.NCRS
 WHERE LOCATE(LOWER(description), 'hole position') > 0;

-- TOP goes with the projection
SELECT TOP 10 ncr_id, supplier_id, severity
  FROM SQM.NCRS
 ORDER BY severity DESC, discovered_at DESC;

-- A window query is identical to Chapter 4
SELECT supplier_id, ncr_id, severity,
       ROW_NUMBER() OVER (
         PARTITION BY supplier_id
         ORDER BY discovered_at DESC) AS rn
  FROM SQM.NCRS
 WHERE discovered_at >= ADD_MONTHS(CURRENT_DATE, -3);
