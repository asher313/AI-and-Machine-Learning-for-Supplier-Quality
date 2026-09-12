# Chapter 6 — 6.11 The SQM pandas Cookbook
flagged = rolling_z(daily_fpy)
anomalies = flagged.loc[
    (flagged["z"].abs() > 3) | flagged["flat_baseline_change"]
]
