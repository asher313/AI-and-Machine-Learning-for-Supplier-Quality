# Chapter 7 — Asher at Northlake
baseline.fit(X, y)
names = baseline["prep"].get_feature_names_out()
coef = pd.Series(baseline["model"].coef_[0], index=names)
print(coef.reindex(coef.abs().sort_values(
    ascending=False).index).head(5).round(3))
