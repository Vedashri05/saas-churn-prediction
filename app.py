import os
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from data.generate_sample import generate_telco_churn
from models.train import get_predictions, train_models
from utils.explainability import get_shap_values, get_top_features, shap_for_customer
from utils.preprocessing import (
    auto_detect_churn_column,
    get_feature_df,
    preprocess,
    preprocess_new_data,
)
from utils.retention import bulk_retention_summary, get_retention_strategies

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0E1525",
    "bg_soft": "#131D34",
    "glass": "rgba(255,255,255,0.06)",
    "glass_border": "rgba(255,255,255,0.16)",
    "text": "#E7EEF9",
    "muted": "#AAB7CC",
    "accent": "#40C4AA",
    "accent_2": "#61A3FF",
    "high": "#EF6A78",
    "medium": "#F2C66D",
    "low": "#53C996",
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# ── CSS ───────────────────────────────────────────────────────────────────────
def apply_style() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

        :root {{
            --bg: {COLORS['bg']};
            --bg-soft: {COLORS['bg_soft']};
            --glass: {COLORS['glass']};
            --glass-border: {COLORS['glass_border']};
            --text: {COLORS['text']};
            --muted: {COLORS['muted']};
            --accent: {COLORS['accent']};
            --accent2: {COLORS['accent_2']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Source Sans 3', sans-serif;
            color: var(--text);
        }}

        .stApp {{
            background:
                radial-gradient(800px 400px at -10% -5%, rgba(97,163,255,0.12), transparent 65%),
                radial-gradient(700px 350px at 110% 100%, rgba(64,196,170,0.10), transparent 65%),
                linear-gradient(160deg, var(--bg) 0%, #101A2D 45%, #0C1321 100%);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15,25,42,0.88), rgba(13,22,38,0.88)) !important;
            border-right: 1px solid var(--glass-border);
            backdrop-filter: blur(18px);
        }}

        .hero-title {{
            font-family: 'Manrope', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0.25rem;
        }}

        .hero-subtitle {{
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 1rem;
        }}

        .glass-card {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 18px;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 30px rgba(5,10,20,0.35);
            animation: cardFade .45s ease-out;
        }}

        @keyframes cardFade {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0);   }}
        }}

        /* result cards */
        .result-card-churn {{
            background: rgba(239,106,120,0.12);
            border: 1px solid rgba(239,106,120,0.45);
            border-radius: 18px;
            padding: 24px;
            text-align: center;
            animation: cardFade .45s ease-out;
        }}
        .result-card-retain {{
            background: rgba(83,201,150,0.12);
            border: 1px solid rgba(83,201,150,0.45);
            border-radius: 18px;
            padding: 24px;
            text-align: center;
            animation: cardFade .45s ease-out;
        }}
        .result-verdict {{
            font-family: 'Manrope', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .result-prob {{
            font-size: 1.1rem;
            color: var(--muted);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: var(--muted);
            font-weight: 600;
            border-radius: 9px;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(64,196,170,0.14) !important;
            color: var(--text) !important;
            border: 1px solid rgba(64,196,170,0.35);
        }}

        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 10px !important;
            border: 1px solid var(--glass-border) !important;
            background: linear-gradient(120deg, rgba(64,196,170,0.16), rgba(97,163,255,0.12)) !important;
            color: var(--text) !important;
            font-weight: 600 !important;
        }}

        [data-testid="stPlotlyChart"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--glass-border);
            background: rgba(255,255,255,0.03);
            animation: cardFade .45s ease-out;
        }}

        [data-testid="metric-container"] {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 10px;
            backdrop-filter: blur(12px);
        }}
        [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .75rem;
        }}
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: var(--text);
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
        }}

        #MainMenu, header, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Session state ─────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "trained_models": None, "results": None, "best_name": None,
        "X_test": None, "y_test": None, "feature_names": None,
        "scaler": None, "encoders": None, "predictions_df": None,
        "shap_values": None, "shap_X": None, "explainer": None,
        "raw_df": None, "target_col": None, "trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Plotly helpers ────────────────────────────────────────────────────────────
def animated_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans 3, sans-serif", "color": COLORS["text"]},
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 30, "r": 20, "t": 55, "b": 40},
        transition={"duration": 700, "easing": "cubic-in-out"},
    )
    return fig


def _as_list(v):
    if v is None: return []
    if isinstance(v, np.ndarray): return v.tolist()
    return list(v)


def _scale_numeric(values, p):
    return (np.asarray(values, dtype=float) * p).tolist()


def _trace_start(t):
    tt = t.get("type"); s = {}
    if tt == "pie" and t.get("values") is not None:
        s["values"] = [0.0] * len(_as_list(t["values"]))
    elif tt == "bar":
        if t.get("orientation") == "h" and t.get("x") is not None:
            s["x"] = [0.0] * len(_as_list(t["x"]))
        elif t.get("y") is not None:
            s["y"] = [0.0] * len(_as_list(t["y"]))
    elif tt == "scatter":
        x, y = _as_list(t.get("x")), _as_list(t.get("y"))
        mode = str(t.get("mode", ""))
        if "lines" in mode and x and y:
            s["x"] = x[:2]; s["y"] = y[:2]
        elif y:
            s["y"] = [0.0] * len(y)
    elif tt == "heatmap" and t.get("z") is not None:
        s["z"] = np.zeros_like(np.asarray(t["z"], dtype=float)).tolist()
    return s


def _trace_progress(t, p):
    tt = t.get("type"); f = {}
    if tt == "pie" and t.get("values") is not None:
        f["values"] = _scale_numeric(_as_list(t["values"]), p)
    elif tt == "bar":
        if t.get("orientation") == "h" and t.get("x") is not None:
            f["x"] = _scale_numeric(_as_list(t["x"]), p)
        elif t.get("y") is not None:
            f["y"] = _scale_numeric(_as_list(t["y"]), p)
    elif tt == "scatter":
        x, y = _as_list(t.get("x")), _as_list(t.get("y"))
        mode = str(t.get("mode", ""))
        if "lines" in mode and x and y:
            n = max(2, int(len(x) * p)); f["x"] = x[:n]; f["y"] = y[:n]
        elif y:
            f["y"] = _scale_numeric(y, p)
    elif tt == "heatmap" and t.get("z") is not None:
        f["z"] = (np.asarray(t["z"], dtype=float) * p).tolist()
    return f


def with_data_animation(fig: go.Figure, steps: int = 24) -> go.Figure:
    final_data = [tr.to_plotly_json() for tr in fig.data]
    for tr, fd in zip(fig.data, final_data):
        tr.update(_trace_start(fd))
    frames = []
    for i in range(1, steps + 1):
        p = i / steps
        fd2 = []
        for td in final_data:
            pl = {"type": td.get("type")}
            pl.update(_trace_progress(td, p))
            fd2.append(pl)
        frames.append(go.Frame(name=f"f{i}", data=fd2, traces=list(range(len(final_data)))))
    fig.frames = frames
    fig.update_layout(updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 1.0, "y": 1.18, "xanchor": "right", "yanchor": "top",
        "buttons": [{"label": "Replay", "method": "animate",
                     "args": [None, {"frame": {"duration": 34, "redraw": True},
                                     "transition": {"duration": 0}, "fromcurrent": False}]}],
    }])
    return fig


def render_animated_chart(fig: go.Figure, height: int = 430) -> None:
    html = fig.to_html(
        full_html=False, include_plotlyjs=True, config=PLOTLY_CONFIG,
        auto_play=True, default_width="100%", default_height=f"{height}px",
    )
    components.html(html, height=height + 10, scrolling=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:12px 6px 4px;'>
            <div style='font-family:Manrope,sans-serif;font-size:1.35rem;font-weight:800;'>
                ChurnGuard AI
            </div>
            <div style='color:#AAB7CC;font-size:0.82rem;'>
                Churn Prediction and Retention Intelligence
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### Data Source")
        data_mode = st.radio("Dataset Mode", ["Use Sample Dataset", "Upload My Dataset"],
                             label_visibility="collapsed")
        uploaded_file = None
        if data_mode == "Upload My Dataset":
            uploaded_file = st.file_uploader("Upload CSV (with Churn column)", type=["csv"])
            if uploaded_file:
                st.success("File uploaded")

        st.markdown("---")
        st.markdown("### Settings")
        test_size = st.slider("Test Split %", 10, 40, 20) / 100
        show_all_models = st.checkbox("Compare All Models", value=True)

        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training models..."):
                if data_mode == "Use Sample Dataset":
                    df = generate_telco_churn(1200)
                else:
                    if uploaded_file is None:
                        st.error("Please upload a CSV file first.")
                        st.stop()
                    df = pd.read_csv(uploaded_file)

                target_col = auto_detect_churn_column(df)
                st.session_state["raw_df"] = df
                st.session_state["target_col"] = target_col

                X, y, feature_names, scaler, encoders = preprocess(df, target_col)
                trained, results, best_name, _, X_te, _, y_te = train_models(X, y, test_size=test_size)
                best_model = trained[best_name]
                explainer, shap_values, shap_X = get_shap_values(best_model, X, feature_names, best_name)
                display_df = get_feature_df(df, target_col)
                preds_df = get_predictions(best_model, X, feature_names, display_df)

                # Save model artifacts using pickle
                model_artifact = {
                    "model": best_model,
                    "scaler": scaler,
                    "encoders": encoders,
                    "feature_names": feature_names,
                    "model_name": best_name,
                }
                with open("saved_model.pkl", "wb") as f:
                    pickle.dump(model_artifact, f)

                st.session_state.update({
                    "trained_models": trained, "results": results, "best_name": best_name,
                    "X_test": X_te, "y_test": y_te, "feature_names": feature_names,
                    "scaler": scaler, "encoders": encoders, "predictions_df": preds_df,
                    "shap_values": shap_values, "shap_X": shap_X, "explainer": explainer,
                    "trained": True,
                })
                st.success(f"Training complete. Best model: {best_name}")

        st.markdown("---")
        if st.session_state["trained"]:
            auc = st.session_state["results"][st.session_state["best_name"]]["auc"]
            st.markdown(f"""
            <div class='glass-card'>
                <div style='font-size:0.8rem;color:#AAB7CC;'>Model status</div>
                <div style='font-weight:700;margin-top:4px;'>{st.session_state['best_name']}</div>
                <div style='font-size:0.8rem;color:#AAB7CC;margin-top:8px;'>AUC</div>
                <div style='font-weight:700;color:{COLORS['accent']};'>{auc:.3f}</div>
            </div>""", unsafe_allow_html=True)

    return test_size, show_all_models


# ── Landing ───────────────────────────────────────────────────────────────────
def render_landing() -> None:
    col1, col2, col3 = st.columns(3)
    cards = [
        ("Train on Any Dataset",
         "Upload your company's CSV or use the built-in sample. The system auto-detects columns and trains fresh models every time."),
        ("Predict New Customers",
         "After training, enter a single customer's details manually OR upload a new unlabeled CSV to get live churn predictions."),
        ("Explain & Retain",
         "SHAP shows exactly why each customer is at risk. The retention engine suggests personalized actions to keep them."),
    ]
    for col, (title, text) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='min-height:180px;'>
                <div style='font-family:Manrope,sans-serif;font-weight:700;margin-bottom:10px;'>{title}</div>
                <div style='color:{COLORS['muted']};font-size:0.94rem;line-height:1.5;'>{text}</div>
            </div>""", unsafe_allow_html=True)
    st.info("Choose a dataset and click **Train Model** in the sidebar to start.")


# ── Tab 1 — Overview ──────────────────────────────────────────────────────────
def render_overview(preds_df, best_res, feature_names):
    total = len(preds_df)
    churn_n = int((preds_df["Prediction"] == "Churn").sum())
    churn_rate = churn_n / total * 100
    avg_prob = float(preds_df["Churn Probability"].mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{total:,}")
    m2.metric("Predicted Churn", f"{churn_n:,}", f"{churn_rate:.1f}%")
    m3.metric("Average Churn Probability", f"{avg_prob:.1f}%")
    m4.metric("Best AUC", f"{best_res['auc']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Risk Distribution")
        counts = preds_df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig = go.Figure(data=[go.Pie(
            labels=counts.index, values=counts.values, hole=0.58,
            marker={"colors": [COLORS["high"], COLORS["medium"], COLORS["low"]],
                    "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            sort=False,
        )])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        render_animated_chart(with_data_animation(animated_layout(fig, "Customer Risk Segments")))

    with c2:
        st.markdown("### Churn Probability Distribution")
        fig = px.histogram(preds_df, x="Churn Probability", nbins=24,
                           color_discrete_sequence=[COLORS["accent_2"]], opacity=0.9)
        fig.add_vline(x=60, line_dash="dash", line_color=COLORS["high"])
        fig.add_vline(x=30, line_dash="dash", line_color=COLORS["medium"])
        fig.update_xaxes(title="Churn Probability (%)")
        fig.update_yaxes(title="Customer Count")
        render_animated_chart(with_data_animation(animated_layout(fig, "Probability Spread")))

    st.markdown("### Dataset Summary")
    raw = st.session_state["raw_df"]
    target = st.session_state["target_col"]
    actual_rate = raw[target].astype(str).str.lower().isin(["1", "yes", "true"]).mean()
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{len(raw):,}")
    d2.metric("Features", f"{len(feature_names)}")
    d3.metric("Target Column", target)
    d4.metric("Actual Churn Rate", f"{actual_rate:.1%}")


# ── Tab 2 — Model Performance ─────────────────────────────────────────────────
def render_performance(results, best_name, best_res, show_all_models):
    st.markdown("### Model Comparison")
    rows = []
    visible = results.items() if show_all_models else [(best_name, results[best_name])]
    for name, res in visible:
        rows.append({
            "Model": f"Best - {name}" if name == best_name else name,
            "Accuracy": f"{res['accuracy']:.3f}", "AUC-ROC": f"{res['auc']:.3f}",
            "CV AUC Mean": f"{res['cv_mean']:.3f}", "CV Std": f"+/-{res['cv_std']:.3f}",
            "Precision (Churn)": f"{res['report'].get('1', res['report'].get(1, {})).get('precision', 0):.3f}",
            "Recall (Churn)": f"{res['report'].get('1', res['report'].get(1, {})).get('recall', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    palette = {"Logistic Regression": COLORS["accent"],
               "Random Forest": COLORS["accent_2"], "Gradient Boosting": COLORS["medium"]}
    with c1:
        st.markdown("### ROC Curves")
        fig = go.Figure()
        for name, res in visible:
            fig.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"], mode="lines",
                name=f"{name} ({res['auc']:.3f})",
                line={"width": 3 if name == best_name else 2, "color": palette.get(name, "#E7EEF9")},
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline",
                                 line={"dash": "dash", "color": "rgba(255,255,255,0.35)", "width": 1.2}))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        render_animated_chart(with_data_animation(animated_layout(fig, "ROC Performance")))

    with c2:
        st.markdown("### Confusion Matrix")
        cm = np.array(best_res["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True,
                        color_continuous_scale=[[0.0, "#1A2740"], [0.6, COLORS["accent_2"]], [1.0, COLORS["accent"]]],
                        x=["Retain", "Churn"], y=["Retain", "Churn"])
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")
        render_animated_chart(with_data_animation(animated_layout(fig, f"Confusion Matrix - {best_name}")))

    st.markdown("### Classification Report")
    report = best_res["report"]
    report_rows = []
    for cls in ["0", "1", "macro avg", "weighted avg"]:
        key = int(cls) if cls in ["0", "1"] and int(cls) in report else cls
        if key not in report: key = cls
        if key in report and isinstance(report[key], dict):
            r = report[key]
            report_rows.append({
                "Class": {"0": "Retain", "1": "Churn"}.get(str(cls), cls),
                "Precision": f"{r.get('precision',0):.3f}", "Recall": f"{r.get('recall',0):.3f}",
                "F1-Score": f"{r.get('f1-score',0):.3f}", "Support": int(r.get("support", 0)),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


# ── Tab 3 — Predictions ───────────────────────────────────────────────────────
def render_predictions(preds_df):
    st.markdown("### Customer Predictions")
    c1, c2, c3 = st.columns(3)
    risk_filter = c1.selectbox("Risk Level", ["All", "High", "Medium", "Low"])
    pred_filter = c2.selectbox("Prediction", ["All", "Churn", "Retain"])
    min_prob = c3.slider("Minimum Churn Probability %", 0, 100, 0)

    disp = preds_df.copy()
    if risk_filter != "All": disp = disp[disp["Risk Level"] == risk_filter]
    if pred_filter != "All": disp = disp[disp["Prediction"] == pred_filter]
    disp = disp[disp["Churn Probability"] >= min_prob]
    st.info(f"Showing {len(disp):,} of {len(preds_df):,} customers")

    def style_prob(val):
        try:
            v = float(val)
            if v >= 60: return f"color:{COLORS['high']};font-weight:700"
            if v >= 30: return f"color:{COLORS['medium']};font-weight:700"
            return f"color:{COLORS['low']};font-weight:700"
        except: return ""

    styled = disp.head(200).style.map(style_prob, subset=["Churn Probability"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
    csv = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Predictions CSV", csv, "churn_predictions.csv", "text/csv")


# ── Tab 4 — SHAP ──────────────────────────────────────────────────────────────
def render_shap(preds_df, shap_values, shap_x, feature_names):
    st.markdown("### Global Feature Importance")
    c1, c2 = st.columns([1.4, 1])
    top_features = get_top_features(shap_values, feature_names, top_n=15)
    with c1:
        fig = px.bar(
            top_features.sort_values("Importance", ascending=True),
            x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[[0.0, COLORS["accent_2"]], [1.0, COLORS["accent"]]],
        )
        fig.update_yaxes(title="")
        fig.update_xaxes(title="Mean |SHAP Value|")
        render_animated_chart(with_data_animation(animated_layout(fig, "Top SHAP Drivers")))
    with c2:
        table = top_features.copy()
        table["Importance"] = table["Importance"].round(4)
        table.index = range(1, len(table) + 1)
        st.dataframe(table, use_container_width=True)

    st.markdown("### Individual Customer Explanation")
    max_idx = min(len(shap_x) - 1, len(preds_df) - 1)
    cust_idx = st.slider("Customer Index", 0, max_idx, 0)
    prob = preds_df["Churn Probability"].iloc[cust_idx]
    risk = preds_df["Risk Level"].iloc[cust_idx]
    pred = preds_df["Prediction"].iloc[cust_idx]
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{prob:.1f}%")
    m2.metric("Prediction", pred)
    m3.metric("Risk Level", str(risk))

    sv_row = shap_values[cust_idx]
    fv_row = shap_x[cust_idx]
    customer_shap = shap_for_customer(None, sv_row, feature_names, fv_row)
    fig = px.bar(
        customer_shap.sort_values("SHAP Impact"),
        x="SHAP Impact", y="Feature", orientation="h",
        color=customer_shap.sort_values("SHAP Impact")["SHAP Impact"].apply(
            lambda x: "Increases churn" if x > 0 else "Reduces churn"),
        color_discrete_map={"Increases churn": COLORS["high"], "Reduces churn": COLORS["low"]},
    )
    fig.update_xaxes(title="SHAP Impact")
    fig.update_yaxes(title="")
    render_animated_chart(with_data_animation(animated_layout(fig, f"Customer {cust_idx} SHAP Breakdown")))
    st.dataframe(customer_shap[["Feature", "Value", "SHAP Impact", "Direction"]],
                 use_container_width=True, hide_index=True)


# ── Tab 5 — Retention ─────────────────────────────────────────────────────────
def render_retention(preds_df, feature_names):
    st.markdown("### Retention Intelligence")
    summary = bulk_retention_summary(preds_df)
    m1, m2, m3 = st.columns(3)
    m1.metric("High Risk Customers", summary["High Risk"])
    m2.metric("Medium Risk Customers", summary["Medium Risk"])
    m3.metric("Low Risk Customers", summary["Low Risk"])

    summary_df = pd.DataFrame({
        "Risk Level": ["High", "Medium", "Low"],
        "Customers": [summary["High Risk"], summary["Medium Risk"], summary["Low Risk"]],
    })
    fig = px.bar(summary_df, x="Risk Level", y="Customers", color="Risk Level",
                 color_discrete_map={"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]})
    render_animated_chart(with_data_animation(animated_layout(fig, "Retention Priority Load")))

    st.markdown("### Customer Retention Plan")
    raw_df = st.session_state["raw_df"]
    high_risk = preds_df[preds_df["Risk Level"] == "High"].head(50)
    if high_risk.empty:
        st.info("No high-risk customers found.")
        return

    selected_customer = st.selectbox(
        "Select a high-risk customer",
        options=high_risk.index.tolist(),
        format_func=lambda i: f"Customer {i} | Churn Probability: {preds_df.loc[i, 'Churn Probability']:.1f}%",
    )
    if selected_customer is not None:
        cust_row = raw_df.iloc[selected_customer].to_dict()
        cprob = preds_df.loc[selected_customer, "Churn Probability"] / 100
        top_shap_feats = []
        if st.session_state["shap_values"] is not None and selected_customer < len(st.session_state["shap_values"]):
            sv = st.session_state["shap_values"][selected_customer]
            top_shap_feats = [feature_names[i] for i in np.argsort(np.abs(sv))[-5:][::-1]]
        strategies_df = get_retention_strategies(cust_row, cprob, top_shap_feats)

        with st.expander("Customer Profile", expanded=False):
            profile_items = {k: v for k, v in cust_row.items() if k != st.session_state["target_col"]}
            st.dataframe(pd.DataFrame([profile_items]), use_container_width=True, hide_index=True)

        st.markdown(f"{len(strategies_df)} recommended actions")
        for _, row in strategies_df.iterrows():
            pc = {"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}.get(row["Priority"], "#d0d7e2")
            st.markdown(f"""
            <div class='glass-card' style='margin-bottom:12px;'>
                <div style='display:flex;justify-content:space-between;gap:12px;'>
                    <div style='font-family:Manrope,sans-serif;font-weight:700;'>{row['Strategy']}</div>
                    <div style='font-size:0.78rem;color:{pc};border:1px solid {pc};padding:2px 8px;border-radius:999px;'>
                        {row['Priority']} Priority
                    </div>
                </div>
                <div style='margin-top:6px;color:{COLORS['muted']};font-size:0.92rem;'>{row['Detail']}</div>
                <div style='margin-top:6px;font-size:0.78rem;color:#c8d3e6;'>Expected Impact: {row['Expected Impact']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Bulk Retention Export")
    if st.button("Generate Retention Report"):
        all_strategies = []
        for idx in high_risk.index[:100]:
            cust = raw_df.iloc[idx].to_dict()
            prob = preds_df.loc[idx, "Churn Probability"] / 100
            strats = get_retention_strategies(cust, prob)
            for _, s in strats.iterrows():
                all_strategies.append({"Customer Index": idx,
                                       "Churn Probability %": preds_df.loc[idx, "Churn Probability"],
                                       **s.to_dict()})
        bulk_df = pd.DataFrame(all_strategies)
        st.dataframe(bulk_df, use_container_width=True, hide_index=True)
        csv = bulk_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Retention Report", csv, "retention_report.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW Tab 6 — Single Customer Predictor ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def render_single_predictor(feature_names, scaler, encoders, raw_df, shap_values, shap_X):
    st.markdown("### Predict a Single New Customer")
    st.markdown(
        f"<div style='color:{COLORS['muted']};font-size:0.92rem;margin-bottom:16px;'>"
        "Fill in the customer details below. These are <b>new customers not in your training data</b> "
        "— the model will predict if they will churn.</div>",
        unsafe_allow_html=True,
    )

    best_model = st.session_state["trained_models"][st.session_state["best_name"]]

    # Build input form dynamically from feature names
    input_data = {}
    cols_per_row = 3
    feature_cols = [c for c in raw_df.columns if c != st.session_state["target_col"]
                    and not (raw_df[c].dtype == object and raw_df[c].nunique() == len(raw_df))]

    # Only show features that exist in feature_names
    display_features = [f for f in feature_names if f in raw_df.columns]

    st.markdown(
        f"<div class='glass-card' style='margin-bottom:20px;'>",
        unsafe_allow_html=True,
    )

    rows = [display_features[i:i+cols_per_row] for i in range(0, len(display_features), cols_per_row)]
    for row_features in rows:
        cols = st.columns(cols_per_row)
        for col, feat in zip(cols, row_features):
            with col:
                col_data = raw_df[feat]
                if col_data.dtype == object or col_data.nunique() <= 6:
                    options = sorted(col_data.dropna().unique().tolist())
                    input_data[feat] = st.selectbox(feat, options, key=f"sp_{feat}")
                elif col_data.dtype in [np.int64, np.int32, int]:
                    mn, mx = int(col_data.min()), int(col_data.max())
                    med = int(col_data.median())
                    input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                        value=med, step=1, key=f"sp_{feat}")
                else:
                    mn, mx = float(col_data.min()), float(col_data.max())
                    med = float(col_data.median())
                    input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                        value=round(med, 2), step=0.01, key=f"sp_{feat}")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict This Customer", use_container_width=True):
        # Build a single-row DataFrame
        row_df = pd.DataFrame([input_data])

        # Encode using saved encoders
        for col in row_df.select_dtypes(include='object').columns:
            if col in encoders:
                le = encoders[col]
                known = set(le.classes_)
                row_df[col] = row_df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                row_df[col] = le.transform(row_df[col])
            else:
                row_df[col] = 0

        # Align columns
        for f in feature_names:
            if f not in row_df.columns:
                row_df[f] = 0
        row_df = row_df[feature_names]
        row_df = row_df.fillna(0)

        X_new = scaler.transform(row_df.values.astype(float))
        prob = best_model.predict_proba(X_new)[0][1]
        pred = "Churn" if prob >= 0.5 else "Retain"
        risk = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")
        risk_color = {"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}[risk]

        # Result card
        card_class = "result-card-churn" if pred == "Churn" else "result-card-retain"
        verdict_color = COLORS["high"] if pred == "Churn" else COLORS["low"]
        st.markdown(f"""
        <div class='{card_class}' style='margin: 20px 0;'>
            <div class='result-verdict' style='color:{verdict_color};'>
                {'⚠️ Likely to Churn' if pred == 'Churn' else '✅ Likely to Stay'}
            </div>
            <div class='result-prob'>Churn Probability: <b>{prob*100:.1f}%</b></div>
            <div style='margin-top:8px;font-size:0.9rem;color:{risk_color};
                 border:1px solid {risk_color};display:inline-block;
                 padding:3px 14px;border-radius:999px;'>
                {risk} Risk
            </div>
        </div>""", unsafe_allow_html=True)

        # SHAP for this customer
        try:
            from utils.explainability import get_shap_values as gsv, shap_for_customer as sfc
            _, sv_single, _ = gsv(best_model, X_new, feature_names, st.session_state["best_name"], max_samples=1)
            cshap = sfc(None, sv_single[0], feature_names, X_new[0])
            st.markdown("#### Why? — SHAP Factor Breakdown")
            fig = px.bar(
                cshap.sort_values("SHAP Impact"),
                x="SHAP Impact", y="Feature", orientation="h",
                color=cshap.sort_values("SHAP Impact")["SHAP Impact"].apply(
                    lambda x: "Increases churn" if x > 0 else "Reduces churn"),
                color_discrete_map={"Increases churn": COLORS["high"], "Reduces churn": COLORS["low"]},
            )
            fig.update_xaxes(title="SHAP Impact")
            fig.update_yaxes(title="")
            render_animated_chart(with_data_animation(animated_layout(fig, "Churn Factors for This Customer")), height=360)
        except Exception:
            pass

        # Retention strategies
        top_shap = []
        try:
            sv_arr = sv_single[0]
            top_shap = [feature_names[i] for i in np.argsort(np.abs(sv_arr))[-5:][::-1]]
        except Exception:
            pass

        strategies_df = get_retention_strategies(input_data, prob, top_shap)
        if pred == "Churn":
            st.markdown("#### Recommended Retention Actions")
            for _, row in strategies_df.iterrows():
                pc = {"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}.get(row["Priority"], "#d0d7e2")
                st.markdown(f"""
                <div class='glass-card' style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <div style='font-weight:700;'>{row['Strategy']}</div>
                        <div style='font-size:0.78rem;color:{pc};border:1px solid {pc};
                             padding:2px 8px;border-radius:999px;'>{row['Priority']} Priority</div>
                    </div>
                    <div style='margin-top:6px;color:{COLORS['muted']};font-size:0.9rem;'>{row['Detail']}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW Tab 7 — Predict New Unlabeled CSV ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def render_new_csv_predictor(feature_names, scaler, encoders):
    st.markdown("### Predict Churn on New Customer Data")
    st.markdown(
        f"<div style='color:{COLORS['muted']};font-size:0.92rem;margin-bottom:16px;'>"
        "Upload a <b>new CSV without a Churn column</b> — these are your <b>active customers today</b>. "
        "The trained model will predict which ones are at risk of churning.</div>",
        unsafe_allow_html=True,
    )

    # Show what columns are expected
    with st.expander("Expected columns (from training data)"):
        st.code(", ".join(feature_names))

    new_file = st.file_uploader(
        "Upload new customer CSV (no Churn column needed)",
        type=["csv"],
        key="new_csv_uploader",
    )

    if new_file is None:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;padding:40px;'>
            <div style='font-size:2rem;margin-bottom:12px;'>📂</div>
            <div style='font-family:Manrope,sans-serif;font-weight:700;margin-bottom:8px;'>
                No file uploaded yet
            </div>
            <div style='color:{COLORS['muted']};font-size:0.9rem;'>
                Upload a CSV with your current active customers.<br>
                The model will predict churn probability for each one.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    try:
        new_df = pd.read_csv(new_file)
        st.success(f"Loaded {len(new_df):,} customers from uploaded file.")

        # Check if churn column accidentally included — remove it
        churn_cols = [c for c in new_df.columns if any(
            kw in c.lower() for kw in ['churn', 'churned', 'attrition', 'target', 'label'])]
        if churn_cols:
            st.warning(f"Found churn-like column(s) {churn_cols} — removing them for prediction.")
            new_df.drop(columns=churn_cols, inplace=True)

        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(new_df.head(10), use_container_width=True, hide_index=True)

        if st.button("Run Churn Prediction on New Data", use_container_width=True):
            best_model = st.session_state["trained_models"][st.session_state["best_name"]]

            with st.spinner("Running predictions..."):
                X_new = preprocess_new_data(new_df, feature_names, scaler, encoders)
                probs = best_model.predict_proba(X_new)[:, 1]
                preds = best_model.predict(X_new)
                risk = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])

            result_df = new_df.copy().reset_index(drop=True)
            result_df["Churn Probability %"] = np.round(probs * 100, 1)
            result_df["Prediction"] = ["Churn" if p == 1 else "Retain" for p in preds]
            result_df["Risk Level"] = risk
            result_df = result_df.sort_values("Churn Probability %", ascending=False)

            # Summary metrics
            total = len(result_df)
            churn_n = int((result_df["Prediction"] == "Churn").sum())
            high_n = int((result_df["Risk Level"] == "High").sum())
            avg_p = float(result_df["Churn Probability %"].mean())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers", f"{total:,}")
            m2.metric("Predicted to Churn", f"{churn_n:,}", f"{churn_n/total*100:.1f}%")
            m3.metric("High Risk", f"{high_n:,}")
            m4.metric("Avg Churn Probability", f"{avg_p:.1f}%")

            # Charts
            c1, c2 = st.columns(2)
            with c1:
                counts = result_df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
                fig = go.Figure(data=[go.Pie(
                    labels=counts.index, values=counts.values, hole=0.55,
                    marker={"colors": [COLORS["high"], COLORS["medium"], COLORS["low"]],
                            "line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
                    sort=False,
                )])
                fig.update_traces(textposition="inside", textinfo="percent+label")
                render_animated_chart(with_data_animation(animated_layout(fig, "New Customer Risk Breakdown")))

            with c2:
                fig = px.histogram(result_df, x="Churn Probability %", nbins=20,
                                   color_discrete_sequence=[COLORS["accent_2"]], opacity=0.85)
                fig.add_vline(x=60, line_dash="dash", line_color=COLORS["high"])
                fig.add_vline(x=30, line_dash="dash", line_color=COLORS["medium"])
                render_animated_chart(with_data_animation(animated_layout(fig, "Predicted Churn Probabilities")))

            # Full results table
            st.markdown("### Full Prediction Results")

            def style_prob2(val):
                try:
                    v = float(val)
                    if v >= 60: return f"color:{COLORS['high']};font-weight:700"
                    if v >= 30: return f"color:{COLORS['medium']};font-weight:700"
                    return f"color:{COLORS['low']};font-weight:700"
                except: return ""

            styled = result_df.head(300).style.map(style_prob2, subset=["Churn Probability %"])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

            # Download
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                csv_bytes,
                "new_customer_predictions.csv",
                "text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    apply_style()
    init_state()
    _, show_all_models = render_sidebar()

    st.markdown("<div class='hero-title'>ChurnGuard AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Professional churn analytics — predict, explain, and retain customers.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state["trained"]:
        render_landing()
        return

    results = st.session_state["results"]
    best_name = st.session_state["best_name"]
    best_res = results[best_name]
    preds_df = st.session_state["predictions_df"]
    shap_values = st.session_state["shap_values"]
    shap_x = st.session_state["shap_X"]
    feature_names = st.session_state["feature_names"]
    scaler = st.session_state["scaler"]
    encoders = st.session_state["encoders"]
    raw_df = st.session_state["raw_df"]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview",
        "Model Performance",
        "Predictions",
        "Explainability",
        "Retention Strategies",
        "Single Customer",       # ← NEW
        "Predict New CSV",       # ← NEW
    ])

    with tab1:
        render_overview(preds_df, best_res, feature_names)
    with tab2:
        render_performance(results, best_name, best_res, show_all_models)
    with tab3:
        render_predictions(preds_df)
    with tab4:
        render_shap(preds_df, shap_values, shap_x, feature_names)
    with tab5:
        render_retention(preds_df, feature_names)
    with tab6:
        render_single_predictor(feature_names, scaler, encoders, raw_df, shap_values, shap_x)
    with tab7:
        render_new_csv_predictor(feature_names, scaler, encoders)


if __name__ == "__main__":
    main()
