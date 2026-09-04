# Chapter 3 — 3.2 Matrices: Every Layer Is a Multiply
import numpy as np

rng = np.random.default_rng(0)

X = rng.standard_normal((1800, 16))   # suppliers x columns
W = rng.standard_normal((16, 8))      # features x hidden
b = np.zeros(8)

H = np.maximum(X @ W + b, 0)          # ReLU(XW + b)
print(X.shape, W.shape, H.shape)
