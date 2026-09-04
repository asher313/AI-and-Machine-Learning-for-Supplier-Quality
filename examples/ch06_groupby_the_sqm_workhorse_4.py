# Chapter 6 — 6.8 Groupby, the SQM Workhorse
df["supplier_avg_sev"] = (
    df.groupby("supplier_id")["severity"].transform("mean")
)
df["sev_deviation"] = df["severity"] - df["supplier_avg_sev"]
