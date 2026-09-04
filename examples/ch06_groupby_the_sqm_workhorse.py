# Chapter 6 — 6.8 Groupby, the SQM Workhorse
stats = df.groupby("supplier_id").agg({
    "severity": ["mean", "max"],
    "cost_impact_usd": "sum",
    "ncr_id": "count",
})
