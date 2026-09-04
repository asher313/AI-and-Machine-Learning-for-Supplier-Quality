# tests/build1/test_explain_sums_to_prediction.py
import numpy as np


def test_shap_values_reconstruct_the_margin(model, X):
    from sqm_ai.build1.explain import explainer_for
    ex = explainer_for(model)
    sv = ex.shap_values(X.iloc[:20])
    base = ex.expected_value
    booster = model.calibrated_classifiers_[0].estimator
    raw = booster.predict(X.iloc[:20], output_margin=True)
    np.testing.assert_allclose(
        sv.sum(axis=1) + base, raw, atol=1e-4)
