# Chapter 6 — 6.10 Time Series
df["discovered_at"] = pd.to_datetime(df["discovered_at"])
ts = df.set_index("discovered_at").sort_index()
