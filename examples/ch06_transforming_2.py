# Chapter 6 — 6.7 Transforming
# Works, but slow and hard to read
df["grade"] = df["audit_score"].apply(
    lambda s: "A" if s >= 90 else "B" if s >= 80 else "C"
)

# Vectorized: binning is what pd.cut is for
df["grade"] = pd.cut(
    df["audit_score"],
    bins=[0, 80, 90, 101],
    labels=["C", "B", "A"],
    right=False,
)
