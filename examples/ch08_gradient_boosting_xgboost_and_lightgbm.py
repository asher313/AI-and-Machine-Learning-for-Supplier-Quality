# Chapter 8 — 8.4 Gradient Boosting: XGBoost and LightGBM
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000,        # upper bound; stopping decides
    learning_rate=0.05,       # shrink each tree's correction
    max_depth=5,              # depth of each tree
    min_child_weight=5,       # min evidence in a leaf
    subsample=0.8,            # rows sampled per tree
    colsample_bytree=0.8,     # columns sampled per tree
    gamma=0.0,                # min loss drop to make a split
    reg_alpha=0.0,            # L1 on leaf weights
    reg_lambda=1.0,           # L2 on leaf weights
    scale_pos_weight=1.0,     # >1 for rare positives (Ch 9)
    objective="binary:logistic",
    eval_metric="aucpr",      # see Chapter 9 on why not auc
    early_stopping_rounds=50,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=100,
)
print("best iteration:", model.best_iteration)
