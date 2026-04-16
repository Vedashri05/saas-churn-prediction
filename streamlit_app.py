# ============================================================
# Streamlit App – Customer Churn Prediction
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: #070b12;
        color: #e7edf7;
    }
    section[data-testid="stSidebar"] {
        background: #0b111c;
        border-right: 1px solid #1d2838;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1420px;
    }
    h1, h2, h3, label, .stMarkdown, .stRadio, .stSelectbox, .stMultiSelect {
        color: #e7edf7;
    }
    .sidebar-brand {
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 18px 16px;
        background: #111827;
        margin-bottom: 18px;
    }
    .brand-kicker {
        color: #7c8da6;
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 7px;
    }
    .brand-title {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 900;
        line-height: 1;
    }
    .brand-subtitle {
        color: #9aa7ba;
        font-size: 13px;
        margin-top: 8px;
    }
    .sidebar-section {
        color: #7c8da6;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin: 22px 0 8px;
    }
    div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 4px;
    }
    div[role="radiogroup"] label:hover {
        background: #121b2a;
        border-color: #263448;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: #2563eb;
        border-color: #3b82f6;
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
        font-weight: 800;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #0f1725;
        border-color: #263448;
    }
    .dashboard-hero {
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 20px 24px;
        background: #0d1421;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: center;
    }
    .dashboard-eyebrow {
        color: #60a5fa;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .dashboard-title {
        font-size: 34px;
        font-weight: 900;
        line-height: 1.1;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .dashboard-subtitle {
        color: #b8c4d4;
        font-size: 16px;
        max-width: 780px;
    }
    .dashboard-meta {
        min-width: 210px;
        border-left: 1px solid #233047;
        padding-left: 20px;
    }
    .meta-label {
        color: #7c8da6;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
    }
    .meta-value {
        color: #e7edf7;
        font-size: 15px;
        font-weight: 800;
        margin: 4px 0 12px;
    }
    .metric-box {
        background: #101826;
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 20px 18px;
        box-shadow: 0 18px 34px rgba(0,0,0,0.25);
        min-height: 108px;
    }
    .metric-label { font-size: 14px; color: #9aa7ba; margin-bottom: 8px; font-weight: 700; }
    .metric-value { font-size: 32px; font-weight: 900; color: #f8fafc; }
    .metric-note { font-size: 12px; color: #7c8da6; margin-top: 8px; }
    .churn-yes { color: #e74c3c !important; }
    .churn-no  { color: #27ae60 !important; }
    .section-header {
        font-size: 18px; font-weight: 800; color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding-left: 12px; margin: 28px 0 14px;
    }
    .chart-panel {
        background: #101826;
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 18px 18px 10px;
        box-shadow: 0 14px 32px rgba(0,0,0,0.20);
        min-height: 430px;
    }
    div[data-testid="stVegaLiteChart"] {
        background: #101826;
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 14px 32px rgba(0,0,0,0.20);
    }
    .insight-card {
        background: #101826;
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 18px;
        min-height: 150px;
    }
    .insight-card strong {
        color: #f8fafc;
        display: block;
        margin-bottom: 8px;
    }
    .insight-card span {
        color: #b8c4d4;
        font-size: 14px;
    }
    div[data-testid="stMetric"] {
        background: #101826;
        border: 1px solid #1d2838;
        border-radius: 8px;
        padding: 16px;
    }
    .dataframe {
        border: 1px solid #1d2838;
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #1d2838;
        border-radius: 8px;
        background: #101826;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #3b82f6;
        background: #2563eb;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load or Train Model ──────────────────────────────
@st.cache_resource
def load_model_and_features(csv_path: str):
    """Load saved model if present, otherwise train a fresh one."""

    model_path    = "churn_model.pkl"
    features_path = "model_features.pkl"

    if os.path.exists(model_path) and os.path.exists(features_path):
        model    = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features

    # ── Train on-the-fly ────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model    import LogisticRegression
    from sklearn.preprocessing   import LabelEncoder

    df = pd.read_csv(csv_path)
    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_cols = [
        "gender", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model,           model_path)
    joblib.dump(list(X.columns), features_path)
    return model, list(X.columns)


@st.cache_data
def load_raw_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


PALETTE = {
    "retained": "#22c55e",
    "churned": "#ef4444",
    "accent": "#60a5fa",
    "amber": "#f59e0b",
    "panel": "#101826",
    "grid": "#2a3548",
    "text": "#e7edf7",
    "muted": "#9aa7ba",
}


def dashboard_chart(chart: alt.Chart, height: int = 320) -> alt.Chart:
    """Apply a consistent dashboard treatment to Altair charts."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(
            labelColor=PALETTE["muted"],
            titleColor=PALETTE["muted"],
            gridColor=PALETTE["grid"],
            domainColor=PALETTE["grid"],
            tickColor=PALETTE["grid"],
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=PALETTE["text"],
            titleColor=PALETTE["muted"],
            orient="top",
            symbolType="square",
        )
    )


def chart_panel(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def close_panel():
    return None


# ── Find CSV ─────────────────────────────────────────────────
CSV_PATH = "Telco-Customer-Churn.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "/mnt/user-data/uploads/Telco-Customer-Churn.csv"

# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="brand-kicker">Retention Analytics</div>
    <div class="brand-title">ChurnIQ</div>
    <div class="brand-subtitle">Customer churn prediction and portfolio monitoring</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Predict Churn", "Dashboard"], label_visibility="collapsed")
st.sidebar.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
st.sidebar.caption("Algorithm: Logistic Regression")
st.sidebar.caption("Dataset: Telco Customer Churn")


# ════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT CHURN
# ════════════════════════════════════════════════════════════
if page == "Predict Churn":

    st.title("Customer Churn Prediction")
    st.markdown("Fill in the customer details and click **Predict** to see churn risk.")
    st.markdown("---")

    model, features = load_model_and_features(CSV_PATH)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)

        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=9000.0,
            value=float(tenure * monthly_charges), step=1.0
        )

        st.markdown("---")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        st.markdown("---")
        senior      = st.checkbox("Senior Citizen")
        partner     = st.checkbox("Has Partner")
        dependents  = st.checkbox("Has Dependents")
        paperless   = st.checkbox("Paperless Billing", value=True)
        phone_svc   = st.checkbox("Phone Service", value=True)

        st.markdown("---")
        predict_btn = st.button("Predict Churn", use_container_width=True, type="primary")

    with col_right:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build input row aligned to model features
            row = {col: 0 for col in features}

            # Numeric
            row["tenure"]         = tenure
            row["MonthlyCharges"] = monthly_charges
            row["TotalCharges"]   = total_charges
            row["SeniorCitizen"]  = int(senior)
            row["Partner"]        = int(partner)
            row["Dependents"]     = int(dependents)
            row["PaperlessBilling"] = int(paperless)
            row["PhoneService"]   = int(phone_svc)

            # Contract one-hot
            if contract == "One year"  and "Contract_One year"  in row: row["Contract_One year"]  = 1
            if contract == "Two year"  and "Contract_Two year"  in row: row["Contract_Two year"]  = 1

            # Internet one-hot
            if internet == "Fiber optic" and "InternetService_Fiber optic" in row:
                row["InternetService_Fiber optic"] = 1
            if internet == "No"          and "InternetService_No"          in row:
                row["InternetService_No"]          = 1

            # Payment one-hot
            pay_map = {
                "Mailed check":               "PaymentMethod_Mailed check",
                "Bank transfer (automatic)":  "PaymentMethod_Bank transfer (automatic)",
                "Credit card (automatic)":    "PaymentMethod_Credit card (automatic)",
            }
            key = pay_map.get(payment)
            if key and key in row:
                row[key] = 1

            sample = pd.DataFrame([row])
            pred       = model.predict(sample)[0]
            pred_proba = model.predict_proba(sample)[0][1]

            # ── Result card ─────────────────────────────────
            if pred == 1:
                st.error("### HIGH CHURN RISK")
                verdict = "WILL CHURN"
                color   = "#e74c3c"
            else:
                st.success("### LOW CHURN RISK")
                verdict = "WILL NOT CHURN"
                color   = "#27ae60"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value" style="color:{color}; font-size:18px;">{verdict}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value" style="color:{color};">{pred_proba:.1%}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("&nbsp;")

            # Probability gauge bar
            st.markdown("**Risk Level**")
            risk_label = "Low" if pred_proba < 0.4 else ("Medium" if pred_proba < 0.65 else "High")
            st.progress(float(pred_proba), text=f"{risk_label}  ({pred_proba:.1%})")

            # ── Action tips ─────────────────────────────────
            st.markdown("---")
            st.markdown("**Recommended Actions**")
            if pred == 1:
                tips = [
                    "Assign a customer success rep immediately",
                    "Offer a loyalty discount or free upgrade",
                    "Propose switching to a 1- or 2-year contract",
                    "Bundle Online Security or Tech Support add-ons",
                ]
            else:
                tips = [
                    "Customer appears satisfied - maintain service quality",
                    "Consider upselling premium add-ons",
                    "Enroll in loyalty newsletter for engagement",
                ]
            for t in tips:
                st.markdown(f"- {t}")
        else:
            st.info("Fill in the customer details on the left, then click **Predict Churn**.")
            st.markdown("""
            **What this app does:**
            - Uses a trained Logistic Regression model
            - Predicts whether a customer is likely to churn
            - Shows churn probability and risk level
            - Recommends retention actions
            """)


# ════════════════════════════════════════════════════════════
# PAGE 2 – DASHBOARD
# ════════════════════════════════════════════════════════════
else:
    df = load_raw_data(CSV_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["CustomerStatus"] = np.where(df["Churn"] == "Yes", "Churned", "Retained")

    with st.sidebar:
        st.markdown("### Dashboard Filters")
        contracts = st.multiselect(
            "Contract",
            sorted(df["Contract"].dropna().unique()),
            default=sorted(df["Contract"].dropna().unique()),
        )
        internet_services = st.multiselect(
            "Internet service",
            sorted(df["InternetService"].dropna().unique()),
            default=sorted(df["InternetService"].dropna().unique()),
        )
        tenure_range = st.slider(
            "Tenure range",
            int(df["tenure"].min()),
            int(df["tenure"].max()),
            (int(df["tenure"].min()), int(df["tenure"].max())),
        )

    filtered = df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internet_services)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
    ].copy()

    total = len(filtered)
    churned = int((filtered["Churn"] == "Yes").sum())
    retained = total - churned
    churn_rate = churned / total if total else 0
    avg_charge = filtered["MonthlyCharges"].mean() if total else 0

    st.markdown("""
    <div class="dashboard-hero">
        <div>
            <div class="dashboard-eyebrow">Telco Customer Churn</div>
            <div class="dashboard-title">Retention Dashboard</div>
            <div class="dashboard-subtitle">
                Track churn risk, billing patterns, tenure health, and contract behavior in one interactive view.
            </div>
        </div>
        <div class="dashboard-meta">
            <div class="meta-label">Model</div>
            <div class="meta-value">Logistic Regression</div>
            <div class="meta-label">View</div>
            <div class="meta-value">Filtered Portfolio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    metric_data = [
        (k1, "Customers", f"{total:,}", "Filtered sample", PALETTE["accent"]),
        (k2, "Churn Rate", f"{churn_rate:.1%}", f"{churned:,} customers churned", PALETTE["churned"]),
        (k3, "Retained", f"{retained:,}", f"{(1 - churn_rate):.1%} retention", PALETTE["retained"]),
        (k4, "Avg Monthly Charge", f"${avg_charge:.2f}", "Across selected customers", PALETTE["amber"]),
    ]
    for col, label, value, note, color in metric_data:
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>""", unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        st.stop()

    status_scale = alt.Scale(
        domain=["Retained", "Churned"],
        range=[PALETTE["retained"], PALETTE["churned"]],
    )

    col1, col2 = st.columns([0.95, 1.25], gap="large")

    with col1:
        chart_panel("Churn Mix")
        churn_summary = (
            filtered.groupby("CustomerStatus", as_index=False)
            .size()
            .rename(columns={"size": "Customers"})
        )
        churn_summary["Share"] = churn_summary["Customers"] / churn_summary["Customers"].sum()
        bar = alt.Chart(churn_summary).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("CustomerStatus:N", title=None, sort=["Retained", "Churned"]),
            y=alt.Y("Customers:Q", title="Customers"),
            color=alt.Color("CustomerStatus:N", scale=status_scale, legend=None),
            tooltip=[
                alt.Tooltip("CustomerStatus:N", title="Status"),
                alt.Tooltip("Customers:Q", format=","),
                alt.Tooltip("Share:Q", format=".1%"),
            ],
        )
        labels = alt.Chart(churn_summary).mark_text(dy=-12, color=PALETTE["text"], fontWeight="bold").encode(
            x=alt.X("CustomerStatus:N", sort=["Retained", "Churned"]),
            y="Customers:Q",
            text=alt.Text("Share:Q", format=".1%"),
        )
        st.altair_chart(dashboard_chart(bar + labels, 300), use_container_width=True)
        close_panel()

    with col2:
        chart_panel("Monthly Charges by Churn")
        charge_hist = alt.Chart(filtered).mark_bar(opacity=0.82, binSpacing=1).encode(
            x=alt.X("MonthlyCharges:Q", bin=alt.Bin(maxbins=32), title="Monthly charges ($)"),
            y=alt.Y("count():Q", title="Customers"),
            color=alt.Color("CustomerStatus:N", scale=status_scale, title=None),
            tooltip=[
                alt.Tooltip("CustomerStatus:N", title="Status"),
                alt.Tooltip("count():Q", title="Customers", format=","),
            ],
        )
        mean_rules = alt.Chart(filtered).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
            x=alt.X("mean(MonthlyCharges):Q"),
            color=alt.Color("CustomerStatus:N", scale=status_scale, title=None),
            tooltip=[
                alt.Tooltip("CustomerStatus:N", title="Status"),
                alt.Tooltip("mean(MonthlyCharges):Q", title="Average charge", format="$.2f"),
            ],
        )
        st.altair_chart(dashboard_chart(charge_hist + mean_rules, 300), use_container_width=True)
        close_panel()

    col3, col4 = st.columns(2, gap="large")

    with col3:
        chart_panel("Tenure Health")
        tenure_hist = alt.Chart(filtered).mark_area(opacity=0.55, interpolate="monotone").encode(
            x=alt.X("tenure:Q", bin=alt.Bin(step=4), title="Tenure (months)"),
            y=alt.Y("count():Q", stack=None, title="Customers"),
            color=alt.Color("CustomerStatus:N", scale=status_scale, title=None),
            tooltip=[
                alt.Tooltip("CustomerStatus:N", title="Status"),
                alt.Tooltip("count():Q", title="Customers", format=","),
            ],
        )
        st.altair_chart(dashboard_chart(tenure_hist, 300), use_container_width=True)
        close_panel()

    with col4:
        chart_panel("Churn Rate by Contract")
        contract_churn = (
            filtered.groupby("Contract", as_index=False)
            .agg(Customers=("Churn", "size"), Churned=("Churn", lambda x: (x == "Yes").sum()))
        )
        contract_churn["ChurnRate"] = contract_churn["Churned"] / contract_churn["Customers"]
        contract_chart = alt.Chart(contract_churn).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Contract:N", title=None, sort="-y"),
            y=alt.Y("ChurnRate:Q", title="Churn rate", axis=alt.Axis(format="%")),
            color=alt.Color(
                "ChurnRate:Q",
                scale=alt.Scale(range=[PALETTE["retained"], PALETTE["amber"], PALETTE["churned"]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Contract:N"),
                alt.Tooltip("Customers:Q", format=","),
                alt.Tooltip("Churned:Q", format=","),
                alt.Tooltip("ChurnRate:Q", title="Churn rate", format=".1%"),
            ],
        )
        contract_labels = alt.Chart(contract_churn).mark_text(
            dy=-10, color=PALETTE["text"], fontWeight="bold"
        ).encode(
            x=alt.X("Contract:N", sort="-y"),
            y="ChurnRate:Q",
            text=alt.Text("ChurnRate:Q", format=".1%"),
        )
        st.altair_chart(dashboard_chart(contract_chart + contract_labels, 300), use_container_width=True)
        close_panel()

    st.markdown('<div class="section-header">Segment Performance</div>', unsafe_allow_html=True)
    segment_table = (
        filtered.groupby(["Contract", "InternetService"], as_index=False)
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", lambda x: (x == "Yes").sum()),
            AvgMonthlyCharge=("MonthlyCharges", "mean"),
            AvgTenure=("tenure", "mean"),
        )
    )
    segment_table["ChurnRate"] = segment_table["Churned"] / segment_table["Customers"]
    segment_table["ChurnRatePct"] = segment_table["ChurnRate"] * 100
    segment_table = segment_table.sort_values(["ChurnRate", "Customers"], ascending=[False, False])
    segment_table = segment_table.drop(columns=["ChurnRate"])
    st.dataframe(
        segment_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Contract": st.column_config.TextColumn("Contract"),
            "InternetService": st.column_config.TextColumn("Internet Service"),
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "Churned": st.column_config.NumberColumn("Churned", format="%d"),
            "AvgMonthlyCharge": st.column_config.NumberColumn("Avg Monthly Charge", format="$%.2f"),
            "AvgTenure": st.column_config.NumberColumn("Avg Tenure", format="%.1f mo"),
            "ChurnRatePct": st.column_config.ProgressColumn(
                "Churn Rate",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insight1, insight2, insight3 = st.columns(3)
    insight1.markdown("""
    <div class="insight-card">
        <strong>Short tenure needs attention</strong>
        <span>Early customers carry the highest churn concentration, so onboarding and first-month check-ins matter most.</span>
    </div>
    """, unsafe_allow_html=True)
    insight2.markdown("""
    <div class="insight-card">
        <strong>Price pressure is visible</strong>
        <span>Churned customers average higher monthly charges. Use plan-fit reviews before billing friction turns into cancellation.</span>
    </div>
    """, unsafe_allow_html=True)
    insight3.markdown("""
    <div class="insight-card">
        <strong>Longer contracts stabilize revenue</strong>
        <span>Month-to-month accounts churn far more often than one-year and two-year contracts.</span>
    </div>
    """, unsafe_allow_html=True)
