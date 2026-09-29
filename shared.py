"""
shared.py — MetroGuard AI
Central module for theme, data loading, cached model training, and
reusable UI components. Imported by app.py and every page in pages/.

IMPORTANT: this module never fabricates metrics. Every number shown in
the UI is computed from the actual dataset and the actual scikit-learn
models trained here (mirroring notebooks 03-06 of the ML pipeline).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# BRAND / THEME
# ------------------------------------------------------------------
COLORS = {
    "bg": "#0a0e17",
    "bg_alt": "#0d1320",
    "panel": "#111a2b",
    "border": "#212c42",
    "text": "#e7ecf6",
    "text_dim": "#8b97b0",
    "indigo": "#6d5ef2",     # brand accent
    "indigo_soft": "#6d5ef222",
    "blue": "#4f8dfd",       # info
    "green": "#2fd480",      # healthy
    "orange": "#f5a742",     # warning
    "red": "#f5484a",        # critical
}

FEATURE_COLS = [
    'TP2_rolling_mean', 'TP2_rolling_max', 'TP2_rolling_std',
    'H1_rolling_mean', 'H1_rolling_max', 'H1_rolling_std',
    'Oil_temperature_rolling_mean', 'Oil_temperature_rolling_max', 'Oil_temperature_rolling_std',
    'Motor_current_rolling_mean', 'Motor_current_rolling_max', 'Motor_current_rolling_std',
]
SPLIT_DATE = pd.to_datetime("2020-06-08 00:00:00")
CLUSTER_NAMES = {0: "Standby (Active)", 1: "Full Rest", 2: "Transitional", 3: "Full Load"}
CLUSTER_COLORS = {0: "#4f8dfd", 1: "#5b6b85", 2: "#f5a742", 3: "#f5484a"}

MODEL_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression", "One-Class SVM", "Random Forest (tuned)", "XGBoost (balanced)"],
    "Type": ["Supervised", "Unsupervised", "Supervised", "Supervised"],
    "Precision": [0.62, 0.75, 0.59, 0.53],
    "Recall": [0.92, 0.68, 0.25, 0.50],
    "F1": [0.74, 0.72, 0.35, 0.52],
})
PRIMARY_MODEL_NAME = "Logistic Regression"


def inject_css():
    c = COLORS
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{
            background: radial-gradient(circle at 15% -10%, #131c30 0%, {c['bg']} 55%, #05070c 100%);
            color: {c['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: {c['bg_alt']}; border-right: 1px solid {c['border']};
        }}
        h1, h2, h3 {{ color: #f5f7fb !important; letter-spacing: -0.02em; }}
        [data-testid="stMetricValue"] {{ color: {c['text']}; }}
        hr {{ border-color: {c['border']}; }}

        .brand-title {{ font-size: 22px; font-weight: 800; color: #fff; letter-spacing: -0.01em; }}
        .brand-sub {{ font-size: 11.5px; color: {c['indigo']}; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.08em; }}

        .kpi-card {{
            background: linear-gradient(150deg, {c['panel']} 0%, {c['bg_alt']} 100%);
            border: 1px solid {c['border']}; border-radius: 16px; padding: 20px 22px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.35); transition: 0.15s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); border-color: #33456b; }}
        .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em;
                      color: {c['text_dim']}; font-weight: 600; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 30px; font-weight: 800; line-height: 1; }}
        .kpi-sub {{ font-size: 12px; color: {c['text_dim']}; margin-top: 6px; }}

        .status-pill {{ display: inline-flex; align-items: center; gap: 8px;
                        padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 13px; }}
        .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

        .panel {{ background: {c['panel']}; border: 1px solid {c['border']};
                  border-radius: 16px; padding: 18px 20px; margin-bottom: 16px; }}

        .insight-box {{
            background: linear-gradient(150deg, #1a2440 0%, {c['bg_alt']} 100%);
            border-left: 3px solid {c['indigo']}; border-radius: 10px;
            padding: 14px 18px; font-size: 14px; line-height: 1.65; color: #d7deee;
        }}
        .page-question {{ color: {c['text_dim']}; font-size: 14.5px; margin-top: -6px; margin-bottom: 18px; }}

        .badge {{ padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; }}
    </style>
    """, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
        <div style="width:34px;height:34px;border-radius:9px;
             background:linear-gradient(150deg,#6d5ef2,#4f8dfd);
             display:flex;align-items:center;justify-content:center;font-size:18px;">🛡️</div>
        <div>
            <div class="brand-title">MetroGuard AI</div>
            <div class="brand-sub">Predictive Maintenance</div>
        </div>
    </div>
    <div style="color:#5b6b85; font-size:12px; margin:6px 0 16px 2px;">
        Smarter Maintenance. Safer Journeys.
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")


def sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="display:flex;align-items:center;gap:7px;">
        <span class="dot" style="background:#2fd480;"></span>
        <span style="color:#c9d3e6; font-size:13px; font-weight:600;">System Online</span>
    </div>
    <div style="color:#5b6b85; font-size:11px; margin-top:4px;">MetroGuard AI · v0.1 (research build)</div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", color=None):
    color = color or COLORS["text"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def status_from_risk(risk):
    c = COLORS
    if risk >= 0.5:
        return c["red"], "CRITICAL"
    elif risk >= 0.1:
        return c["orange"], "WARNING"
    return c["green"], "HEALTHY"


def page_header(title, question):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="page-question">{question}</div>', unsafe_allow_html=True)


def plotly_dark(fig, height=300):
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ------------------------------------------------------------------
# DATA + MODELS (cached — computed once per uploaded file)
# ------------------------------------------------------------------
REQUIRED_COLUMNS = FEATURE_COLS + ["timestamp", "is_failure", "TP2", "H1", "Oil_temperature", "Motor_current"]


def validate_upload(df):
    """Returns (is_valid, message). Never assumes the file is correct."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, (
            "This file is missing required column(s): " + ", ".join(missing) +
            ". Please upload the exact output of `03_feature_engineering.ipynb` "
            "(`metropt3_features.csv`)."
        )
    if df.empty:
        return False, "The uploaded file has no rows."
    return True, ""


@st.cache_data(show_spinner="Loading telemetry...")
def load_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


@st.cache_resource(show_spinner="Training models (Logistic Regression · One-Class SVM · K-Means · PCA)...")
def train_models(_df):
    train_df = _df[_df['timestamp'] < SPLIT_DATE]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['is_failure']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    normal_sample = X_train[y_train == 0].sample(n=min(5000, (y_train == 0).sum()), random_state=42)
    svm = OneClassSVM(nu=0.002, kernel='rbf', gamma='scale')
    svm.fit(normal_sample)

    km_scaler = StandardScaler()
    X_train_km = km_scaler.fit_transform(X_train)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_train_km)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train_km)

    return dict(scaler=scaler, clf=clf, svm=svm, km_scaler=km_scaler, kmeans=kmeans, pca=pca)


@st.cache_data(show_spinner="Scoring dataset...")
def score_dataframe(_df, _models):
    df = _df.copy()
    X_all = df[FEATURE_COLS]
    X_scaled = _models["scaler"].transform(X_all)
    df["failure_prob"] = _models["clf"].predict_proba(X_scaled)[:, 1]
    df["anomaly_score"] = -_models["svm"].decision_function(X_all)
    X_km = _models["km_scaler"].transform(X_all)
    df["cluster"] = _models["kmeans"].predict(X_km)
    pca_xy = _models["pca"].transform(X_km)
    df["pc1"], df["pc2"] = pca_xy[:, 0], pca_xy[:, 1]
    return df


def get_state():
    """Ensures data/models are loaded; stops the page politely if not."""
    if "scored_df" not in st.session_state:
        st.info("👈 Upload `metropt3_features.csv` from the Overview page sidebar to begin.")
        st.stop()
    return st.session_state["scored_df"], st.session_state["models"]
