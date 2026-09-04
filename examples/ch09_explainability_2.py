# Chapter 9 — 9.6 Explainability
from sklearn.inspection import permutation_importance

res = permutation_importance(
    model, X_test, y_test, scoring="average_precision",
    n_repeats=10, n_jobs=-1, random_state=42,
)
out = pd.DataFrame({
    "feature": X.columns,
    "mean": res.importances_mean,
    "std": res.importances_std,
}).sort_values("mean", ascending=False)
