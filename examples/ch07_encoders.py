# Chapter 7 — Encoders
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

onehot = OneHotEncoder(
    handle_unknown="ignore",   # unseen category -> all zeros
    sparse_output=False,       # return a plain array
    min_frequency=10,          # rare categories -> one bucket
)

# tier is ordered: C < B < A. Say so.
ordinal = OrdinalEncoder(
    categories=[["C", "B", "A"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
