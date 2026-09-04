# Chapter 6 — 6.4 DataFrames In and Out
df = pd.read_csv("ncrs_2025-09_2026-08.csv")
df = pd.read_excel("audit_log.xlsx", sheet_name="2026")
df = pd.read_parquet("supplier_month.parquet")
df = pd.read_sql("SELECT * FROM sqm.suppliers", engine)

df.to_csv("out.csv", index=False)
df.to_parquet("out.parquet")
df.to_sql("supplier_month", engine, if_exists="replace")
