# Chapter 6 — 6.3 Indexing, Masks, and Matrix Ops
a = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

a[0, 1]        # 1            row 0, column 1
a[0]           # [0 1 2 3]    row 0
a[:, 0]        # [0 4 8]      column 0
a[1:, 1:3]     # [[5 6], [9 10]]
