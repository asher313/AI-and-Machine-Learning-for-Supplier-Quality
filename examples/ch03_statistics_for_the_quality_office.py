# Chapter 3 — 3.7 Statistics for the Quality Office
import pandas as pd
from scipy import stats

# Did the new process change first-pass yield?
t_stat, p_value = stats.ttest_ind(
    fpy_before, fpy_after, equal_var=False
)

# Is defect category independent of supplier?
table = pd.crosstab(df["supplier_id"], df["category"])
expected = stats.contingency.expected_freq(table.to_numpy())
if (expected < 5).any():
    print("Sparse table: do not report the asymptotic p-value.")
    print("Expected cells below 5:", int((expected < 5).sum()))
else:
    chi2, p_chi, dof, expected = stats.chi2_contingency(table)
