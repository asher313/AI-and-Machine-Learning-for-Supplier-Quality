# Chapter 3 — 3.1 Vectors, Dot Products, and Why Cosine Similarity Is a Dot Product
import numpy as np

# Rounded [ncr_count, avg_severity, fpy, otd], August 2026
cobalt = np.array([47.0, 2.5, 0.912, 0.83])
redline = np.array([39.0, 2.9, 0.933, 0.88])

print(round(float(cobalt @ redline), 3))
print(round(float(np.linalg.norm(cobalt)), 4))
