# Chapter 8 — 8.7 Unsupervised: K-Means, DBSCAN, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=5, init="k-means++", n_init=10,
            max_iter=300, random_state=42)
labels = km.fit_predict(X_scaled)
centroids = km.cluster_centers_

for k in range(2, 11):
    m = KMeans(n_clusters=k, n_init=10, random_state=42)
    lab = m.fit_predict(X_scaled)
    print(k, round(m.inertia_),
          round(silhouette_score(X_scaled, lab), 3))
