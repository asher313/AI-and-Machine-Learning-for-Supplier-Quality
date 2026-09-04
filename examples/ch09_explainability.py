# Chapter 9 — 9.6 Explainability
import pandas as pd

imp = pd.Series(model.feature_importances_,
                index=X.columns)
print(imp.sort_values(ascending=False).head(10))
