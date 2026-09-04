# Chapter 9 — 9.3 Metrics Decoded
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, log_loss, roc_auc_score,
)

p = model.predict_proba(X_test)[:, 1]
yhat = (p >= 0.30).astype(int)

print(roc_auc_score(y_test, p))            # ranking
print(average_precision_score(y_test, p))  # PR-AUC
print(log_loss(y_test, p))                 # calibration
print(confusion_matrix(y_test, yhat))
print(classification_report(
    y_test, yhat, target_names=["ok", "at risk"]))
