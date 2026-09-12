import numpy as np
import pandas as pd

from sqm_ai.build1.explain import (
    explainer_for,
    top_drivers,
    transformed_frame,
)
from sqm_ai.build1.train import fit_regressor
from sqm_ai.features import NUMERIC


def test_shap_values_reconstruct_saved_regressor(tmp_path):
    import joblib

    rng = np.random.default_rng(21)
    X = pd.DataFrame(
        rng.normal(size=(60, len(NUMERIC))), columns=NUMERIC
    )
    X["tier"] = np.resize(["A", "B", "C"], len(X))
    X.loc[0, "fpy"] = np.nan
    y = pd.Series(10 + 2 * X.ncr_count_90d + X.avg_days_late)
    fitted = fit_regressor(
        X,
        y,
        np.arange(len(X)),
        {"n_estimators": 10, "max_depth": 2},
    )
    joblib.dump(fitted, tmp_path / "reg.joblib")
    model = joblib.load(tmp_path / "reg.joblib")
    transformed = transformed_frame(model, X)
    ex = explainer_for(model)
    sv = ex.shap_values(transformed)
    np.testing.assert_allclose(
        sv.sum(axis=1) + ex.expected_value,
        model.predict(X),
        atol=1e-4,
    )
    drivers = top_drivers(ex, transformed, 0, raw=X, values=sv[0])
    assert len(drivers) == 5
    assert all(
        d["impact_units"]
        == "raw predicted harm before zero floor"
        for d in drivers
    )
