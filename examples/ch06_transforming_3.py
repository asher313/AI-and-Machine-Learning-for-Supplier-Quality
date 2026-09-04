# Chapter 6 — 6.7 Transforming
import numpy as np

conds = [
    df["severity"] >= 4,
    df["severity"] == 3,
]
choices = ["escalate", "car"]
df["action"] = np.select(conds, choices, default="log")
