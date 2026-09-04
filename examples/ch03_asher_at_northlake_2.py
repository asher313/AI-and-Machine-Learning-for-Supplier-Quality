# Chapter 3 — Asher at Northlake
mean_fn = lambda x: x.mean()

lo, hi = bootstrap_diff_ci(before, after, mean_fn)
print(round(lo, 4), round(hi, 4))

lo_a, hi_a = bootstrap_ci(after, mean_fn)
print(round(lo_a, 4), round(hi_a, 4))

u_stat, p_u = stats.mannwhitneyu(before, after)
print(round(p_u, 5))
