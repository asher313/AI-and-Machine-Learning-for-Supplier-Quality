# Chapter 3 — Asher at Northlake
import pandas as pd
from scipy import stats

from sqm_ai.stats import bootstrap_ci, bootstrap_diff_ci

lots = pd.read_csv("data/cobalt_lot_fpy.csv")
before = lots.loc[lots.period == "before", "fpy"].to_numpy()
after = lots.loc[lots.period == "after", "fpy"].to_numpy()

print(len(before), round(before.mean(), 4))
print(len(after), round(after.mean(), 4))

t_stat, p_value = stats.ttest_ind(
    before, after, equal_var=False
)
print("t=%.2f p=%.5f" % (t_stat, p_value))
