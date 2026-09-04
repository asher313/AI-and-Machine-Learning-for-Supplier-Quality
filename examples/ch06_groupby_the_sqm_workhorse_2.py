# Chapter 6 — 6.8 Groupby, the SQM Workhorse
stats = (
    df.groupby("supplier_id")
      .agg(
          ncr_count=("ncr_id", "count"),
          avg_severity=("severity", "mean"),
          max_severity=("severity", "max"),
          total_cost=("cost_impact_usd", "sum"),
      )
      .reset_index()
)
