# Chapter 6 — 6.10 Time Series
ts["ncr_30d"] = ts["severity"].rolling(30).count()
ts["fpy_30d"] = ts["fpy"].rolling("30D").mean()
