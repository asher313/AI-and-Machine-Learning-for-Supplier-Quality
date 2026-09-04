# Chapter 6 — 6.7 Transforming
df = df.rename(columns={"fpy": "first_pass_yield"})
df = df.drop(columns=["temp_col"])
df = df.drop(index=[0, 1])
