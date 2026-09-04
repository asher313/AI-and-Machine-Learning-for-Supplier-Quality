# Chapter 9 — Asher at Northlake
import numpy as np
import pandas as pd

p = model.predict_proba(X.loc[te])[:, 1]
y_te = y.loc[te].to_numpy()
grid = np.arange(0.02, 0.90, 0.01)

rows = []
for t in grid:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    rows.append({
        "threshold": round(float(t), 2),
        "flagged": tp + fp,
        "recall": round(tp / (tp + fn), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "cost": fp * 540 + fn * 12600,
    })
sweep = pd.DataFrame(rows)
print(sweep.loc[sweep["cost"].idxmin()])
