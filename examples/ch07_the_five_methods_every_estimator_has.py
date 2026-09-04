# Chapter 7 — 7.1 The Five Methods Every Estimator Has
estimator.fit(X, y)            # learn from data
estimator.predict(X)           # predict a label or a number
estimator.predict_proba(X)     # class probabilities (classifiers)
estimator.score(X, y)          # a default metric; often wrong
estimator.transform(X)         # reshape data (scalers, encoders)
