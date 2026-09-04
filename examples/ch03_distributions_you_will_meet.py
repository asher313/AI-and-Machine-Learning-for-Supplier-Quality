# Chapter 3 — 3.6 Distributions You Will Meet
from scipy import stats

lam = 120
print(round(stats.poisson.pmf(120, lam), 4))
print(round(1 - stats.poisson.cdf(152, lam), 4))
print(int(stats.poisson.ppf(0.99, lam)))
