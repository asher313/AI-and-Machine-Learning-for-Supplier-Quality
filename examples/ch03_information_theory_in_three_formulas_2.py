# Chapter 3 — 3.8 Information Theory in Three Formulas
q_flat = np.array([0.25, 0.25, 0.25, 0.25])
q_good = np.array([0.40, 0.28, 0.22, 0.10])

for q in (q_flat, q_good):
    ce = -(p * np.log2(q)).sum()
    kl = (p * np.log2(p / q)).sum()
    print(round(ce, 3), round(kl, 3))
