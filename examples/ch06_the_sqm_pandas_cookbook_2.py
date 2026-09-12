# Chapter 6 — 6.11 The SQM pandas Cookbook
counts = (
    df.groupby("supplier_id")["ncr_id"]
      .count()
      .sort_values(ascending=False)
)
cum_pct = counts.cumsum() / counts.sum()
n_to_80 = int(cum_pct.searchsorted(0.80)) + 1
vital_few = counts.iloc[:n_to_80].index.tolist()

print(len(vital_few), "of", counts.size, "suppliers")
