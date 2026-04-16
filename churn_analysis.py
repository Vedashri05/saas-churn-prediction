# ============================================================
# SaaS Customer Churn Prediction & Retention Optimization
# Dataset: Telco Customer Churn (Kaggle)
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 2. LOAD DATASET ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("Telco-Customer-Churn.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── 3. DATA PREPROCESSING ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# Drop irrelevant column
df.drop(columns=["customerID"], inplace=True)

# Fix TotalCharges — sometimes imported as object due to spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values — fill numeric NaNs with median
print(f"\nMissing values before:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
print("Missing values filled with median.")

# Convert Churn to binary: Yes → 1, No → 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode binary Yes/No columns with Label Encoding
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encode multi-category columns
multi_cols = [
    "gender", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Final safety net: fill any remaining NaNs
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"\nDataset shape after encoding: {df.shape}")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# ── 4. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis (EDA)")
print("=" * 60)

print("\nSummary Statistics:")
print(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].describe())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Telco Customer Churn – EDA", fontsize=16, fontweight="bold")

# --- Plot 1: Churn Distribution ---
# Insight: ~26% customers churned — class imbalance to keep in mind
churn_counts = df["Churn"].value_counts()
axes[0, 0].bar(
    ["Not Churned (0)", "Churned (1)"],
    churn_counts.values,
    color=["#2ecc71", "#e74c3c"],
    edgecolor="white"
)
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0, 0].text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")

# --- Plot 2: Monthly Charges vs Churn ---
# Insight: Churned customers tend to have higher monthly charges
axes[0, 1].boxplot(
    [df[df["Churn"] == 0]["MonthlyCharges"], df[df["Churn"] == 1]["MonthlyCharges"]],
    labels=["Not Churned", "Churned"],
    patch_artist=True,
    boxprops=dict(facecolor="#3498db", alpha=0.7)
)
axes[0, 1].set_title("Monthly Charges vs Churn")
axes[0, 1].set_ylabel("Monthly Charges ($)")

# --- Plot 3: Tenure vs Churn ---
# Insight: New customers (low tenure) churn much more — onboarding is critical
axes[1, 0].hist(
    df[df["Churn"] == 0]["tenure"], bins=30, alpha=0.6, label="Not Churned", color="#2ecc71"
)
axes[1, 0].hist(
    df[df["Churn"] == 1]["tenure"], bins=30, alpha=0.6, label="Churned", color="#e74c3c"
)
axes[1, 0].set_title("Tenure Distribution by Churn")
axes[1, 0].set_xlabel("Tenure (Months)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

# --- Plot 4: Correlation Heatmap (top features) ---
# Insight: tenure & contract type are strongly negatively correlated with churn
top_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
corr = df[top_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title("Correlation Heatmap (Key Features)")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nEDA plots saved as 'eda_plots.png'")

# ── 5. MODEL BUILDING ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Building")
print("=" * 60)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 80/20 train-test split, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ── Logistic Regression ──────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("\nLogistic Regression trained ✓")

# ── Random Forest ────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Random Forest trained ✓")

# ── 6. MODEL EVALUATION ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*40}")
    print(f"  Model : {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    return acc

lr_acc = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
rf_acc = evaluate_model("Random Forest",        rf_model, X_test, y_test)

# ── Best Model ───────────────────────────────────────────────
print("\n" + "=" * 60)
if rf_acc >= lr_acc:
    best_model = rf_model
    best_name  = "Random Forest"
else:
    best_model = lr_model
    best_name  = "Logistic Regression"

print(f"  ★  BEST MODEL: {best_name}  (Accuracy: {max(lr_acc, rf_acc):.4f})")
print("=" * 60)

# ── 7. PREDICTION ON SAMPLE INPUT ────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Sample Prediction")
print("=" * 60)

# Build a sample row aligned with training feature columns
sample_raw = pd.DataFrame([{col: 0 for col in X.columns}])

# Set meaningful values for a high-risk customer:
# - New customer, month-to-month contract, high charges, no security add-ons
if "tenure"         in sample_raw.columns: sample_raw["tenure"]         = 2
if "MonthlyCharges" in sample_raw.columns: sample_raw["MonthlyCharges"] = 85.0
if "TotalCharges"   in sample_raw.columns: sample_raw["TotalCharges"]   = 170.0
if "SeniorCitizen"  in sample_raw.columns: sample_raw["SeniorCitizen"]  = 0

# One-hot flags: month-to-month, electronic check payment
for col in ["Contract_One year", "Contract_Two year"]:
    if col in sample_raw.columns:
        sample_raw[col] = 0   # Month-to-month (base category)

pred       = best_model.predict(sample_raw)[0]
pred_proba = best_model.predict_proba(sample_raw)[0][1]

print(f"\nSample Customer Profile:")
print(f"  Tenure        : 2 months")
print(f"  Monthly Charge: $85.00")
print(f"  Contract      : Month-to-Month")
print(f"\nPrediction  : {'CHURN ⚠️' if pred == 1 else 'NO CHURN ✅'}")
print(f"Churn Prob  : {pred_proba:.2%}")

# ── 8. RETENTION STRATEGY ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Business Retention Recommendations")
print("=" * 60)

recommendations = [
    "1. 📅  PROMOTE LONG-TERM CONTRACTS  — Month-to-month customers churn 3× more.\n"
    "       Offer discounts (e.g. 20% off) to switch to 1- or 2-year plans.",

    "2. 🎯  EARLY ONBOARDING PROGRAM  — Churn risk is highest in the first 6 months.\n"
    "       Assign a dedicated success rep and send proactive check-in calls.",

    "3. 💰  PRICE SENSITIVITY ALERTS  — Flag customers paying >$70/month with low tenure.\n"
    "       Offer a loyalty discount or a plan downgrade to retain them.",

    "4. 🔒  BUNDLE SECURITY ADD-ONS  — Customers without Online Security or Tech Support\n"
    "       churn more. Create affordable bundles to increase stickiness.",

    "5. 📊  CHURN SCORE DASHBOARD  — Use this model's probability scores weekly.\n"
    "       Prioritize outreach to customers with >60% churn probability."
]

for rec in recommendations:
    print(f"\n{rec}")

# ── 9. SAVE MODEL ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving Best Model & Feature Columns")
print("=" * 60)

joblib.dump(best_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print(f"  ✓ Model saved      → churn_model.pkl")
print(f"  ✓ Features saved   → model_features.pkl")
print("\nAll done! 🎉")
