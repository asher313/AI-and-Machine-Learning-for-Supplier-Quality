# Chapter 8 — 8.7 Unsupervised: K-Means, DBSCAN, PCA
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5, metric="euclidean",
            n_jobs=-1)
labels = db.fit_predict(X_scaled)
outliers = (labels == -1).sum()
