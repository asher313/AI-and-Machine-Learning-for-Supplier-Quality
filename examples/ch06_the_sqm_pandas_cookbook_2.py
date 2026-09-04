# Chapter 6 — 6.11 The SQM pandas Cookbook
counts = (
    df.groupby("supplier_id")["ncr_id"]
      .count()
      .sort_values(ascending=False)
)
cum_pct = counts.cumsum() / counts.sum()
vital_few = cum_pct[cum_pct <= 0.80].index.tolist()

print(len(vital_few), "of", counts.size, "suppliers")
