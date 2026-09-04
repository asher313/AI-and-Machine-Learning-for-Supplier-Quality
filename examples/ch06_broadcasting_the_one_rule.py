# Chapter 6 — 6.2 Broadcasting: The One Rule
# Scalar and array: (3,) with a scalar
a = np.array([1, 2, 3])
a + 10                 # [11, 12, 13]

# Row vector and matrix: (2,3) with (3,)
m = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([10, 20, 30])
m + v
# [[11, 22, 33],
#  [14, 25, 36]]

# Column vector and matrix: (2,3) with (2,1)
col = np.array([[100], [200]])
m + col
# [[101, 102, 103],
#  [204, 205, 206]]
