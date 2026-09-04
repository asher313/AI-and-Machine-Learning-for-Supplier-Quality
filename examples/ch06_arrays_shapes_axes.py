# Chapter 6 — 6.1 Arrays, Shapes, Axes
import numpy as np

# From data
a = np.array([1, 2, 3])            # 1-D, shape (3,)
b = np.array([[1, 2], [3, 4]])     # 2-D, shape (2, 2)

# Constant arrays
zeros = np.zeros((3, 4))           # 3x4 of zeros
ones = np.ones((2, 3, 4))          # 2x3x4 of ones
full = np.full((2, 2), 7)          # 2x2 of 7s
eye = np.eye(4)                    # 4x4 identity

# Ranges
r = np.arange(0, 10, 2)            # [0 2 4 6 8]
ls = np.linspace(0, 1, 11)         # 11 points, 0 to 1

# Random: always seed, or you cannot reproduce a bug
rng = np.random.default_rng(seed=42)
u = rng.random((3, 3))             # uniform [0, 1)
n = rng.normal(loc=0, scale=1, size=(3, 3))
i = rng.integers(0, 10, size=5)
