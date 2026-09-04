# Chapter 7 — 7.4 Cross-Validation Done Right
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate,
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=42)
cv = TimeSeriesSplit(n_splits=5, test_size=3, gap=3)
