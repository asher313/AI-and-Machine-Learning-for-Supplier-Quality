# Chapter 9 — 9.6 Explainability
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=["fpy", "otd_pct", ("fpy", "otd_pct")],
)
