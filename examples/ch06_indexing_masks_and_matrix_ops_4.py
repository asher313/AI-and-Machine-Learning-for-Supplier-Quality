# Chapter 6 — 6.3 Indexing, Masks, and Matrix Ops
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B     # [[5, 12], [21, 32]]   element-wise
A @ B     # [[19, 22], [43, 50]]  matrix multiply
A.T       # [[1, 3], [2, 4]]      transpose
