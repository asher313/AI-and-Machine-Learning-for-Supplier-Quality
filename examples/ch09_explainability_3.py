# Chapter 9 — 9.6 Explainability
import shap

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)   # (rows, features)

shap.summary_plot(sv, X_test)        # global picture
shap.force_plot(explainer.expected_value,
                sv[0], X_test.iloc[0])   # one row
