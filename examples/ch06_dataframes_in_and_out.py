# Chapter 6 — 6.4 DataFrames In and Out
import pandas as pd

df = pd.DataFrame({
    "supplier_id": ["S-0417", "S-1130", "S-0088"],
    "fpy": [0.912, 0.933, 0.981],
    "otd": [0.83, 0.88, 0.96],
    "audit_score": [71, 68, 92],
})
