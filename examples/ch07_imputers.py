# Chapter 7 — Imputers
from sklearn.impute import KNNImputer, SimpleImputer

# constant or statistic; keep a missingness flag
imp = SimpleImputer(strategy="median", add_indicator=True)

# use similar rows' values; slow on large tables
imp = KNNImputer(n_neighbors=5)
