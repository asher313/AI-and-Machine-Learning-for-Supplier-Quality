# Chapter 6 — 6.3 Indexing, Masks, and Matrix Ops
mask = a > 5
a[mask]        # [6 7 8 9 10 11]  — always 1-D
a[a > 5] = 0   # assign through the mask, in place
