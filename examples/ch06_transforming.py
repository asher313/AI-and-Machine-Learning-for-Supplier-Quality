# Chapter 6 — 6.7 Transforming
df["risk_raw"] = (1 - df["fpy"]) * 0.5 + (1 - df["otd"]) * 0.5
