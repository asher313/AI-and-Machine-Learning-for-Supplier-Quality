# Chapter 8 — 8.5 SVMs and k-NN
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=25,
    weights="distance",    # nearer neighbours count more
    metric="minkowski", p=2,   # p=2 is Euclidean
    n_jobs=-1,
)
