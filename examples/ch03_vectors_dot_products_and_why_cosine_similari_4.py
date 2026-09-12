# Chapter 3 — 3.1 Vectors, Dot Products, and Why Cosine Similarity Is a Dot Product
# Illustrative fitted reference values for this geometry example.
# They are not the generated full-cohort August statistics.
mu = np.array([16.3, 2.1, 0.958, 0.91])
sd = np.array([14.8, 0.6, 0.031, 0.07])

z_cobalt = (cobalt - mu) / sd
z_redline = (redline - mu) / sd
z_apex = (apex - mu) / sd

print(round(cosine(z_cobalt, z_redline), 4))
print(round(cosine(z_cobalt, z_apex), 4))
