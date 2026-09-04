# Chapter 1 — Asher at Northlake
share = df["category"].value_counts(normalize=True)
print(share.round(2))

by_supplier = (
    df.groupby(["supplier_id", "supplier_name"])["ncr_id"]
      .count()
      .sort_values(ascending=False)
)
print(by_supplier.head(3))

# Pareto: how many suppliers cause 80% of all NCRs?
cumulative = by_supplier.cumsum() / by_supplier.sum()
vital_few = cumulative[cumulative <= 0.80]
print(len(vital_few), "of", df["supplier_id"].nunique())
