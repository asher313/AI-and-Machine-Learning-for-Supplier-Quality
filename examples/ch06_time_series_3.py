# Chapter 6 — 6.10 Time Series
ts["ncr_last_30_rows"] = ts["ncr_id"].rolling(30).count()
ts["ncr_30d"] = ts["ncr_id"].rolling("30D").count()
