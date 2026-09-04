# Chapter 3 — 3.6 Distributions You Will Meet
from scipy import stats

mu, var = 120.0, 313.0
r = mu**2 / (var - mu)
p = mu / var

print(round(1 - stats.nbinom.cdf(152, r, p), 4))
print(int(stats.nbinom.ppf(0.99, r, p)))
