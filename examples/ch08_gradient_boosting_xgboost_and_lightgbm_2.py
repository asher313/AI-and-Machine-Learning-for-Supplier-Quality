# Chapter 8 — 8.4 Gradient Boosting: XGBoost and LightGBM
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,            # main complexity knob
    min_child_samples=25,
    feature_fraction=0.8,     # = colsample_bytree
    bagging_fraction=0.8,     # = subsample
    bagging_freq=5,           # bag every 5 iterations
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="average_precision",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
