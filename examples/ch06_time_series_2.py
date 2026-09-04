# Chapter 6 — 6.10 Time Series
monthly = ts.resample("ME").agg(
    ncr_count=("ncr_id", "count"),
    total_cost=("cost_impact_usd", "sum"),
)
