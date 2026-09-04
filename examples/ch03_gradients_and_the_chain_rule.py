# Chapter 3 — 3.4 Gradients and the Chain Rule
import math

x = 1.5      # standardized ncr_count_3mo for one supplier
y = 1.0      # it did have a severity-3+ NCR in 90 days
w, b = 0.8, -0.5

z = w * x + b                      # 0.7
p = 1 / (1 + math.exp(-z))         # sigmoid -> 0.6682
loss = -math.log(p)                # -log likelihood -> 0.4032

dz = p - y                         # -0.3318  (the whole trick)
dw = dz * x                        # -0.4977
db = dz                            # -0.3318

lr = 0.1
w, b = w - lr * dw, b - lr * db    # 0.8498, -0.4668
z2 = w * x + b
p2 = 1 / (1 + math.exp(-z2))
print(round(p2, 4), round(-math.log(p2), 4))
