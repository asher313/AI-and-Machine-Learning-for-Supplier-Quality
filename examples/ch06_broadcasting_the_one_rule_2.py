# Chapter 6 — 6.2 Broadcasting: The One Rule
# Centre each supplier's months on that supplier's mean
x = np.array([[0.94, 0.91, 0.96],
              [0.88, 0.85, 0.87]])
row_means = x.mean(axis=1, keepdims=True)   # (2, 1)
centred = x - row_means                     # (2, 3)
