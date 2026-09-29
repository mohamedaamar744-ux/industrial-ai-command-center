"""
MetroGuard AI — Overview
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared import (
    inject_css, sidebar_brand, sidebar_footer, kpi_card, status_from_risk,
    page_header, plotly_dark, load_data, train_models, score_dataframe, validate_upload,
    CLUSTER_NAMES, COLORS, MODEL_RESULTS, PRIMARY_MODEL_NAME,
)

st.set_page_config(page_title="MetroGuard AI · Overview", page_icon="🛡️", layout="wide")
inject_css()
sidebar_brand()

uploaded = st.sidebar.file_uploader("Upload metropt3_features.csv", type=["csv"])
if uploaded is not None:
    df = load_data(uploaded)
    is_valid, msg = validate_upload(df)
    if not is_valid:
        st.sidebar.error(msg)
    else:
        models = train_models(df)
        scored = score_dataframe(df, models)
        st.session_state["scored_df"] = scored
        st.session_state["models"] = models

if "scored_df" not in st.session_state:
    sidebar_footer()
    st.markdown("# Industrial AI Command Center")
    st.caption("MetroGuard AI · AI-Powered Predictive Maintenance for Rail Compressor Systems")
    st.info("👈 Upload `metropt3_features.csv` (produced by notebook `03_feature_engineering.ipynb`) "
            "from the sidebar to activate the dashboard.")
    st.stop()

df = st.session_state["scored_df"]

st.sidebar.markdown("---")
min_ts, max_ts = df['timestamp'].min(), df['timestamp'].max()
snapshot_ts = st.sidebar.slider(
    "Snapshot moment", min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(),
    value=max_ts.to_pydatetime(), format="YYYY-MM-DD HH:mm",
)
sidebar_footer()
idx = (df['timestamp'] - pd.Timestamp(snapshot_ts)).abs().idxmin()
latest = df.loc[idx]
risk = latest["failure_prob"]
status_color, status_text = status_from_risk(risk)
n_anomalies = int((df["anomaly_score"] > df["anomaly_score"].quantile(0.98)).sum())

# ------------------------------------------------------------------
page_header("Overview", "What is the current health of the system?")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Machine Health", f"{(1-risk)*100:.1f} / 100",
             f"as of {latest['timestamp']}", status_color)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Risk</div>
        <div class="status-pill" style="background:{status_color}22; color:{status_color};">
            <span class="dot" style="background:{status_color};"></span>{status_text}
        </div>
        <div class="kpi-sub">Logistic Regression · P(failure)={risk*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    kpi_card("Anomalies Detected", n_anomalies,
             "One-Class SVM · top 2% scores, full dataset", COLORS["blue"])
with c4:
    kpi_card("Active Sensors", "12 features / 4 raw sensors",
             "TP2 · H1 · Oil Temp · Motor Current", COLORS["indigo"])

st.write("")

reasons = []
if latest["TP2_rolling_mean"] < 5 and latest["cluster"] == 3:
    reasons.append("TP2 pressure sits below its normal full-load peak")
if latest["Oil_temperature_rolling_std"] > 1.5:
    reasons.append("oil temperature is fluctuating more than usual")
if latest["H1_rolling_mean"] < 1 and latest["TP2_rolling_mean"] > 5:
    reasons.append("H1 sensor has gone silent while under load")
insight = (f"Elevated risk ({risk*100:.0f}%) is associated with " + ", ".join(reasons) + "."
           if reasons and risk >= 0.1 else
           "All monitored sensors fall within expected ranges for the current operating state.")
st.markdown(f'<div class="insight-box"><b>🧠 AI Insight</b><br>{insight}</div>', unsafe_allow_html=True)

st.write("")
col_a, col_b = st.columns([1.3, 1])

with col_a:
    st.markdown("#### Recent Sensor Trend — TP2 Pressure")
    recent = df.tail(720)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['TP2'],
                              line=dict(color=COLORS["blue"], width=1.3), name="TP2"))
    st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

with col_b:
    st.markdown("#### Operating States (last known)")
    dist = df['cluster'].map(CLUSTER_NAMES).value_counts()
    fig2 = go.Figure(go.Pie(labels=dist.index, values=dist.values, hole=0.55,
                             marker=dict(colors=["#4f8dfd", "#5b6b85", "#f5a742", "#f5484a"])))
    st.plotly_chart(plotly_dark(fig2, 280), use_container_width=True)

st.markdown("#### Model Summary")
st.dataframe(MODEL_RESULTS.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
             use_container_width=True, hide_index=True)
st.caption(f"Primary detection model: **{PRIMARY_MODEL_NAME}** (best F1 among tested configurations)")
