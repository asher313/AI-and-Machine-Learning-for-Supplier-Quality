# Chapter 6 — 6.11 The SQM pandas Cookbook
# np and pd as imported in Sections 6.1 and 6.4
from scipy import stats


def fpy_slope(g: pd.DataFrame) -> float:
    """Slope of FPY per month; negative means degrading."""
    g = g.dropna(subset=["fpy"])
    if len(g) < 6:
        return np.nan
    x = np.arange(len(g), dtype=float)
    return stats.linregress(x, g["fpy"].to_numpy()).slope


trends = (
    monthly.sort_values(["supplier_id", "month"])
           .groupby("supplier_id")[["fpy"]]
           .apply(fpy_slope)
)
degrading = trends[trends < -0.01].sort_values()
