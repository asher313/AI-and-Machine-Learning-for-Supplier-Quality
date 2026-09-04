# Chapter 8 — 8.3 Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=500,        # more is better, then flat
    max_depth=None,          # trees grow full; the average
                             # is the regularizer
    min_samples_leaf=1,
    max_features="sqrt",     # features tried per split
    bootstrap=True,
    oob_score=True,          # free validation, see below
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"OOB score: {model.oob_score_:.4f}")
