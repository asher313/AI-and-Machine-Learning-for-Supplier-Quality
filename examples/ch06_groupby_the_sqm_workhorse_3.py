# Chapter 6 — 6.8 Groupby, the SQM Workhorse
watch = df.groupby("supplier_id").filter(
    lambda g: len(g) >= 20
)
