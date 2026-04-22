import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def get_shap_values(model, X, feature_names, model_name="Random Forest", max_samples=200):
    X_sample = X[:max_samples] if len(X) > max_samples else X
    try:
        if model_name in ("Random Forest", "Gradient Boosting"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 50))
        shap_values = explainer.shap_values(X_sample)[:, :, 1]

    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return explainer, shap_values, X_sample


def get_top_features(shap_values, feature_names, top_n=10):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Importance": mean_abs})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def shap_for_customer(explainer, shap_values_row, feature_names, feature_values):
    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": feature_values,
        "SHAP Impact": shap_values_row,
    })
    df["Direction"] = df["SHAP Impact"].apply(
        lambda x: "Increases Churn" if x > 0 else "Reduces Churn"
    )
    df["Abs Impact"] = df["SHAP Impact"].abs()
    return df.sort_values("Abs Impact", ascending=False).head(10).reset_index(drop=True)
