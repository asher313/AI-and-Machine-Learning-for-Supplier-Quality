# Chapter 6 — 6.10 Time Series
df["year"] = df["discovered_at"].dt.year
df["quarter"] = df["discovered_at"].dt.quarter
df["dow"] = df["discovered_at"].dt.dayofweek   # Monday = 0
