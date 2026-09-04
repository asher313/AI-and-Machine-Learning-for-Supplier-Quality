# Chapter 8 — 8.5 SVMs and k-NN
from sklearn.svm import SVC, LinearSVC

model = SVC(
    C=1.0,                 # inverse regularization
    kernel="rbf",          # "linear", "poly", "rbf"
    gamma="scale",         # RBF width
    probability=True,      # needed for predict_proba; slow
    class_weight="balanced",
    random_state=42,
)
# Large data: linear kernel, far faster
model = LinearSVC(C=1.0, class_weight="balanced",
                  max_iter=10000)
