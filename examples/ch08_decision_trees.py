# Chapter 8 — 8.2 Decision Trees
from sklearn.tree import (
    DecisionTreeClassifier, export_text, plot_tree,
)

model = DecisionTreeClassifier(
    max_depth=6,             # hard cap on question depth
    min_samples_split=50,    # do not split a tiny node
    min_samples_leaf=25,     # do not create a tiny leaf
    criterion="gini",        # or "entropy"
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)
print(export_text(model, feature_names=list(X.columns)))
