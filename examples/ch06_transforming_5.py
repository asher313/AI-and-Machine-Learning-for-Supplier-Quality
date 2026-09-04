# Chapter 6 — 6.7 Transforming
df.dropna()                      # any NaN anywhere: rarely right
df.dropna(subset=["fpy"])        # only where fpy is missing
df["fpy"] = df["fpy"].fillna(df["fpy"].mean())
df["fpy"] = df["fpy"].ffill()    # carry the last value forward
