# Chapter 6 — 6.1 Arrays, Shapes, Axes
a = np.array([[1, 2, 3], [4, 5, 6]])
a.sum(axis=0)     # [5, 7, 9]        per-column sums
a.sum(axis=1)     # [6, 15]          per-row sums
a.mean(axis=0)    # [2.5, 3.5, 4.5]
