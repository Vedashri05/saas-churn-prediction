import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
warnings.filterwarnings("ignore")


def train_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }
    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        results[name] = {
            "accuracy": acc, "auc": auc, "report": report,
            "confusion_matrix": cm, "fpr": fpr, "tpr": tpr,
            "precision_curve": prec, "recall_curve": rec,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        }
        trained[name] = model
    best_name = max(results, key=lambda k: results[k]["auc"])
    return trained, results, best_name, X_train, X_test, y_train, y_test


def get_predictions(model, X, feature_names, original_df):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    risk = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    out = original_df.copy().reset_index(drop=True)
    out["Churn Probability"] = np.round(probs * 100, 1)
    out["Prediction"] = ["Churn" if p == 1 else "Retain" for p in preds]
    out["Risk Level"] = risk
    return out.sort_values("Churn Probability", ascending=False)
