# Chapter 6 — 6.6 Selecting and Filtering
# Correct
mask = (df["fpy"] > 0.9) & (df["otd"] > 0.9)
mask = (df["fpy"] < 0.8) | (df["audit_score"] < 70)
low = df.loc[~(df["severity"] >= 3)]

# Wrong: raises ValueError about ambiguous truth value
mask = df["fpy"] > 0.9 and df["otd"] > 0.9

# Wrong: raises for these floating-point Series
mask = df["fpy"] > 0.9 & df["otd"] > 0.9
