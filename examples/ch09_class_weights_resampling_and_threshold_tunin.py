# Chapter 9 — 9.2 Class Weights, Resampling, and Threshold Tuning
import numpy as np
from sklearn.metrics import precision_recall_curve

p = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, p)

# Best F1 (a default when you have no cost numbers)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_f1_threshold = thr[int(np.argmax(f1[:-1]))]

# Lowest cost (much better when you do)
def expected_cost(t, y_true, proba, fp_cost, fn_cost):
    pred = (proba >= t).astype(int)
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return fp * fp_cost + fn * fn_cost

# 540 and 12600 are Ravi's prices, in dollars, from the
# end of this chapter: one false alarm, one expected miss
grid = np.arange(0.01, 1.0, 0.01)
costs = [expected_cost(t, y_val, p, 540, 12600)
         for t in grid]
best_cost_threshold = float(grid[int(np.argmin(costs))])
