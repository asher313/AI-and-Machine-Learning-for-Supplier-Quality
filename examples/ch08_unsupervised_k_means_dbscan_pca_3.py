# Chapter 8 — 8.7 Unsupervised: K-Means, DBSCAN, PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_small = pca.fit_transform(X_scaled)
print(X.shape[1], "->", X_small.shape[1])
print(pca.explained_variance_ratio_.cumsum().round(3))
