-- Chapter 5 — 5.3 Calculation Views
-- Consuming a plain calculation view
SELECT supplier_id, defect_count, scrap_cost
  FROM "_SYS_BIC"."sqm/CV_SUPPLIER_DEFECTS"
 WHERE supplier_id = 'S-0417';

-- Consuming one with input parameters
SELECT supplier_id, fiscal_period, defect_count
  FROM "_SYS_BIC"."sqm/CV_SUPPLIER_METRICS"
       ('PLACEHOLDER' = ('$$IP_FISCAL_YEAR$$', '2026'),
        'PLACEHOLDER' = ('$$IP_PLANT$$', 'TUL1'))
 WHERE supplier_id LIKE 'S-04%';
