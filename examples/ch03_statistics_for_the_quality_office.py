# Chapter 3 — 3.7 Statistics for the Quality Office
import pandas as pd
from scipy import stats

# Did the new process change first-pass yield?
t_stat, p_value = stats.ttest_ind(
    fpy_before, fpy_after, equal_var=False
)

# Is defect category independent of supplier?
table = pd.crosstab(df["supplier_id"], df["category"])
chi2, p_chi, dof, expected = stats.chi2_contingency(table)
