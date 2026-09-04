# Chapter 3 — 3.8 Information Theory in Three Formulas
import numpy as np

# Northlake's NCR category split (Chapter 1)
p = np.array([0.46, 0.24, 0.21, 0.09])
H = -(p * np.log2(p)).sum()

print(round(H, 3), round(2**H, 2))
