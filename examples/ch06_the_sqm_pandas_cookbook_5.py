# Chapter 6 — 6.11 The SQM pandas Cookbook
# np and pd as imported in Sections 6.1 and 6.4
from scipy import stats


def fpy_slope(g: pd.DataFrame) -> float:
    """Slope of FPY per month; negative means degrading."""
    g = g.dropna(subset=["fpy", "month"]).sort_values("month")
    if len(g) < 6:
        return np.nan
    if g["month"].duplicated().any():
        raise ValueError("one observation per calendar month")
    months = pd.to_datetime(g["month"]).dt.to_period("M")
    x = months.astype("int64").to_numpy(dtype=float)
    x -= x.min()
    return float(stats.linregress(x, g["fpy"].to_numpy()).slope)


trends = (
    monthly.sort_values(["supplier_id", "month"])
           .groupby("supplier_id")[["month", "fpy"]]
           .apply(fpy_slope)
)
degrading = trends[trends < -0.01].sort_values()
