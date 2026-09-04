# Chapter 7 — 7.4 Cross-Validation Done Right
from sklearn.model_selection import cross_validate

folds = list(month_folds(df))
res = cross_validate(
    pipeline, X, y, cv=folds,
    scoring=["roc_auc", "average_precision", "recall"],
    return_train_score=True, n_jobs=-1,
)
for k in sorted(res):
    if k.startswith(("test_", "train_")):
        print(f"{k:24s} {res[k].mean():.3f} "
              f"+/- {res[k].std():.3f}")
