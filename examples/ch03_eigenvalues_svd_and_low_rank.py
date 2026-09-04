# Chapter 3 — 3.3 Eigenvalues, SVD, and Low Rank
import numpy as np

# M: (1800, 12) counts, built in Chapter 4 from sqm.ncrs.
C = M - M.mean(axis=0)          # centre each month
U, s, Vt = np.linalg.svd(C, full_matrices=False)

energy = s**2 / (s**2).sum()
print(np.round(s[:4], 1))
print(np.round(np.cumsum(energy)[:4], 3))

k = 2
C_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print(round(np.linalg.norm(C - C_k) / np.linalg.norm(C), 3))
