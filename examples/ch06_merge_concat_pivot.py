# Chapter 6 — 6.9 Merge, Concat, Pivot
m = pd.merge(ncrs, suppliers, on="supplier_id", how="inner")
m = pd.merge(ncrs, suppliers, on="supplier_id", how="left")

m = pd.merge(
    ncrs, suppliers,
    left_on="supplier_ref", right_on="id",
    how="left",
    validate="many_to_one",
)

m = pd.merge(ncrs, parts, on=["supplier_id", "part_number"])
