# Chapter 7 — Scalers
# WRONG: the scaler has seen the test rows' mean and std
X_all = StandardScaler().fit_transform(X)
X_tr, X_te = X_all[:40000], X_all[40000:]

# RIGHT: split RAW rows; the pipeline fits preprocessing
X_tr, X_te = X.iloc[:40000], X.iloc[40000:]
y_tr = y.iloc[:40000]
pipe.fit(X_tr, y_tr)
pipe.predict(X_te)
