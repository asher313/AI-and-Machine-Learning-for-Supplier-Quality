# Chapter 3 — 3.1 Vectors, Dot Products, and Why Cosine Similarity Is a Dot Product
import numpy as np

# [ncr_count, avg_severity, fpy, otd] for August 2026
cobalt = np.array([47.0, 2.5, 0.912, 0.83])
redline = np.array([39.0, 2.9, 0.933, 0.88])

print(cobalt @ redline)
print(np.linalg.norm(cobalt))
