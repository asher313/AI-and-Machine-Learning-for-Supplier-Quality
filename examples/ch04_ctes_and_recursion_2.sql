-- Chapter 4 — 4.4 CTEs and Recursion
-- Explode wing-rib assembly ASSY-7700 to every child part
WITH RECURSIVE bom AS (
  -- Anchor: the direct children of the top assembly
  SELECT part_id, parent_id, quantity, 1 AS level
  FROM sqm.bill_of_materials
  WHERE parent_id = 'ASSY-7700'

  UNION ALL

  -- Step: children of every part found so far
  SELECT b.part_id, b.parent_id, b.quantity, bom.level + 1
  FROM sqm.bill_of_materials b
  JOIN bom ON b.parent_id = bom.part_id
  WHERE bom.level < 20        -- guard against a cyclic BOM
)
SELECT level, parent_id, part_id, quantity
FROM bom
ORDER BY level, part_id;
