# Chapter 6 — 6.11 The SQM pandas Cookbook
# np and pd as imported in Sections 6.1 and 6.4
def rolling_z(
    daily: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Flag supplier-days whose FPY departs from trend."""
    d = daily.sort_values(["supplier_id", "day"]).copy()
    g = d.groupby("supplier_id")["fpy"]
    d["roll_mean"] = g.transform(
        lambda s: s.rolling(window, min_periods=20).mean()
    )
    d["roll_std"] = g.transform(
        lambda s: s.rolling(window, min_periods=20).std()
    )
    d["z"] = (
        (d["fpy"] - d["roll_mean"])
        / d["roll_std"].replace(0, np.nan)
    )
    return d
