# Chapter 1 — Asher at Northlake
import pandas as pd

df = pd.read_csv(
    "data/ncrs_2025-09_2026-08.csv",
    parse_dates=["discovered_at", "closed_at"],
)

print(df.shape)            # rows, columns
print(df.dtypes)           # is each column the type you expect?
print(df.isnull().sum())   # missing values per column
