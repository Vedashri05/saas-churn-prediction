import pandas as pd

RETENTION_RULES = [
    {
        "condition": lambda row: row.get("Contract", "") == "Month-to-month",
        "feature": "Contract",
        "strategy": "Offer Annual Plan Discount",
        "detail": "Customer is on a month-to-month contract. Offer a 15-20% discount for annual conversion.",
        "priority": "High",
        "impact": "High",
    },
    {
        "condition": lambda row: str(row.get("TechSupport", "")).lower() in ("no", "0", "none"),
        "feature": "TechSupport",
        "strategy": "Enable Free Tech Support Trial",
        "detail": "Customer lacks tech support. Offer a 30-day support trial to improve satisfaction.",
        "priority": "High",
        "impact": "Medium",
    },
    {
        "condition": lambda row: float(row.get("MonthlyCharges", 0)) > 80,
        "feature": "MonthlyCharges",
        "strategy": "Offer Loyalty Discount",
        "detail": "Monthly charges are high. Offer a 10% loyalty discount or a better-value bundle.",
        "priority": "High",
        "impact": "High",
    },
    {
        "condition": lambda row: int(row.get("tenure", 100)) < 12,
        "feature": "tenure",
        "strategy": "New Customer Onboarding Program",
        "detail": "Customer tenure is under 12 months. Assign a customer success manager and onboarding sequence.",
        "priority": "Medium",
        "impact": "Medium",
    },
    {
        "condition": lambda row: str(row.get("OnlineSecurity", "")).lower() in ("no", "0", "none"),
        "feature": "OnlineSecurity",
        "strategy": "Bundle Online Security Add-on",
        "detail": "No online security subscription. Offer a discounted bundle with security features.",
        "priority": "Medium",
        "impact": "Medium",
    },
    {
        "condition": lambda row: str(row.get("PaymentMethod", "")).lower() == "electronic check",
        "feature": "PaymentMethod",
        "strategy": "Promote Auto-Pay Incentive",
        "detail": "Electronic check users churn more often. Offer monthly savings for switching to auto-pay.",
        "priority": "Low",
        "impact": "Low",
    },
    {
        "condition": lambda row: str(row.get("InternetService", "")).lower() == "fiber optic",
        "feature": "InternetService",
        "strategy": "Run Proactive Service Quality Check",
        "detail": "Fiber customers may see service friction. Trigger a proactive quality check and outreach.",
        "priority": "Medium",
        "impact": "Medium",
    },
    {
        "condition": lambda row: int(row.get("SeniorCitizen", 0)) == 1,
        "feature": "SeniorCitizen",
        "strategy": "Enroll in Assisted Support Program",
        "detail": "Provide simplified billing and dedicated support routing for senior customers.",
        "priority": "Medium",
        "impact": "Medium",
    },
]

GENERIC_STRATEGIES = [
    {
        "Strategy": "Schedule Proactive Outreach Call",
        "Detail": "Arrange a customer success call within 7 days to identify friction and priorities.",
        "Priority": "High",
        "Expected Impact": "High",
    },
    {
        "Strategy": "Send Personalized Retention Email",
        "Detail": "Send a personalized email with relevant product value and recommended next steps.",
        "Priority": "Medium",
        "Expected Impact": "Medium",
    },
]


def get_retention_strategies(customer_row: dict, churn_prob: float, top_shap_features: list = None):
    strategies = []
    for rule in RETENTION_RULES:
        try:
            if rule["condition"](customer_row):
                priority = rule["priority"]
                if top_shap_features and rule["feature"] in top_shap_features[:3]:
                    priority = "High"
                strategies.append({
                    "Strategy": rule["strategy"],
                    "Detail": rule["detail"],
                    "Priority": priority,
                    "Expected Impact": rule["impact"],
                })
        except Exception:
            pass
    if churn_prob > 0.6:
        strategies.extend(GENERIC_STRATEGIES)
    if not strategies:
        strategies.append({
            "Strategy": "Regular Engagement Check-in",
            "Detail": "Monitor behavior monthly and send proactive engagement nudges.",
            "Priority": "Low",
            "Expected Impact": "Low",
        })
    return pd.DataFrame(strategies).drop_duplicates(subset=["Strategy"])


def bulk_retention_summary(predictions_df):
    high = (predictions_df["Churn Probability"] > 60).sum()
    med = ((predictions_df["Churn Probability"] > 30) & (predictions_df["Churn Probability"] <= 60)).sum()
    low = (predictions_df["Churn Probability"] <= 30).sum()
    return {"High Risk": int(high), "Medium Risk": int(med), "Low Risk": int(low)}
