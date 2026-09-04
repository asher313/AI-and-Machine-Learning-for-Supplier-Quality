# Chapter 6 — 6.6 Selecting and Filtering
df["fpy"]                      # one column -> Series
df[["supplier_id", "fpy"]]     # several -> DataFrame

df.iloc[0]        # by position: first row
df.iloc[0:3, 0:2] # first 3 rows, first 2 columns

df.loc[0]                          # by label
df.loc[df["fpy"] > 0.95]           # boolean filter
df.loc[df["fpy"] > 0.95, ["supplier_id", "fpy"]]

df.query("fpy > 0.95 and audit_score > 80")
