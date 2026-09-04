# Chapter 6 — 6.11 The SQM pandas Cookbook
top10 = (
    df.groupby("supplier_id")["ncr_id"]
      .count()
      .nlargest(10)
)
