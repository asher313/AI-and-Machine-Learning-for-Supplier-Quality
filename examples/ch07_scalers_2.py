# Chapter 7 — Scalers
# WRONG: the scaler has seen the test rows' mean and std
X_all = StandardScaler().fit_transform(X)
X_tr, X_te = X_all[:40000], X_all[40000:]

# RIGHT: the pipeline fits the scaler on training rows only
pipe.fit(X_tr, y_tr)
pipe.predict(X_te)
