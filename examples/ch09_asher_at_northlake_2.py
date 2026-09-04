# Chapter 9 — Asher at Northlake
import shap

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_month)


def top_drivers(i, k=5):
    contrib = sv[i]
    order = sorted(range(len(contrib)),
                   key=lambda j: abs(contrib[j]),
                   reverse=True)[:k]
    return [
        {
            "feature": X_month.columns[j],
            "value": float(X_month.iloc[i, j]),
            "impact": round(float(contrib[j]), 3),
            "direction": ("increases risk"
                          if contrib[j] > 0
                          else "decreases risk"),
        }
        for j in order
    ]
