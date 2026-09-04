# Chapter 6 — 6.9 Merge, Concat, Pivot
grid = df.pivot_table(
    index="supplier_id",
    columns="category",
    values="ncr_id",
    aggfunc="count",
    fill_value=0,
)
