"""
frontend/dashboard.py
─────────────────────────────────────────────────────────────────
Refactored Streamlit dashboard with JWT authentication and role-aware navigation.

Uses the backend configuration system, the singleton model repository,
and the shared feature engineering functions to preserve 100% of the
original layout and features while adding login/logout and RBAC page visibility.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import requests

# Ensure project root is in python path so we can import from backend and ml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.repositories.model_repository import ModelRepository
from ml.features.feature_engineering import engineer_features_from_df
from backend.constants.ml_constants import RISK_COLORS, RISK_MESSAGES, RISK_HIGH, RISK_MEDIUM, RISK_LOW

# Backend API base URL loaded from env settings
API_BASE_URL = settings.BACKEND_URL

# ── Page Config ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="ERP Anomaly Detector",
        page_icon="🔍",
        layout="wide"
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .stAlert { border-radius: 10px; }
        .block-container { padding-top: 2rem; }
        .login-container {
            max-width: 400px;
            margin: 4rem auto;
            padding: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize authentication session state variables."""
    defaults = {
        "authenticated": False,
        "access_token": None,
        "refresh_token": None,
        "username": None,
        "role": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def do_login(username: str, password: str) -> tuple[bool, str | None]:
    """
    Authenticate against the backend API and store tokens in session state.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/v1/auth/login",
            json={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["authenticated"] = True
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            st.session_state["username"] = data["username"]
            st.session_state["role"] = data["role"]
            return True, None
        elif resp.status_code == 401:
            return False, "Invalid username or password."
        elif resp.status_code == 403:
            return False, "Account is deactivated or forbidden."
        else:
            return False, f"Unexpected response from backend API (HTTP {resp.status_code})."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend API. Ensure the FastAPI server is running."
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"


def do_logout():
    """
    Clear session tokens — frontend-only logout.
    Optionally notifies the backend for audit logging.
    """
    try:
        if st.session_state.get("access_token"):
            requests.post(
                f"{API_BASE_URL}/v1/auth/logout",
                headers={"Authorization": f"Bearer {st.session_state['access_token']}"},
                timeout=5
            )
    except Exception:
        pass  # Logout is best-effort on the backend side

    # Clear all session tokens
    st.session_state["authenticated"] = False
    st.session_state["access_token"] = None
    st.session_state["refresh_token"] = None
    st.session_state["username"] = None
    st.session_state["role"] = None


# Define which roles can see which pages
ROLE_PAGE_ACCESS = {
    "Admin":         ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats"],
    "Finance User":  ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats"],
    "Fraud Analyst": ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats"],
    "Auditor":       ["📊 Dashboard", "📈 Model Stats"],  # Auditors can view but not create predictions
}


def get_accessible_pages() -> list:
    """Return pages accessible to the current user's role."""
    role = st.session_state.get("role", "")
    return ROLE_PAGE_ACCESS.get(role, [])


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state["authenticated"]:
    st.title("🔒 ERP Anomaly Detector — Login")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Sign In")
        st.caption("Authenticate to access the anomaly detection dashboard.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔐 Login", use_container_width=True)

            if submitted:
                if username and password:
                    success, error_msg = do_login(username, password)
                    if success:
                        st.success(f"Welcome, {st.session_state['username']}! Role: {st.session_state['role']}")
                        st.rerun()
                    else:
                        st.error(error_msg)
                else:
                    st.warning("Please enter both username and password.")

        st.markdown("---")
        st.caption("**Default test accounts:** admin, finance, analyst, auditor (password: password123)")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATED AREA — Load Model
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_cached_repository():
    repo = ModelRepository()
    repo.load_artifacts()
    return repo


try:
    repo = get_cached_repository()
    model = repo.model
    scaler = repo.scaler
    feature_cols = repo.feature_cols
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(f"Make sure model files are available at paths configured in settings: model={settings.MODEL_PATH}, scaler={settings.SCALER_PATH}, features={settings.FEATURE_COLS_PATH}")
    st.stop()


# ── Prediction Logic ──────────────────────────────────────────────────────────
def predict_df(df):
    """
    Predict anomaly scores for an uploaded DataFrame using the shared feature engineering logic.
    """
    df_feat = engineer_features_from_df(df)
    X = df_feat[feature_cols].values
    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[:, 1]
    df_feat['anomaly_score'] = proba
    df_feat['is_fraud'] = proba >= settings.FRAUD_THRESHOLD

    def get_risk_label(val):
        if val >= settings.HIGH_RISK_THRESHOLD:
            return RISK_HIGH
        elif val >= settings.FRAUD_THRESHOLD:
            return RISK_MEDIUM
        else:
            return RISK_LOW

    df_feat['risk_level'] = [get_risk_label(p) for p in proba]
    return df_feat


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 ERP Anomaly Detector")
st.sidebar.markdown("---")

# User info and logout
st.sidebar.markdown(f"👤 **{st.session_state['username']}**")
st.sidebar.markdown(f"🏷️ Role: **{st.session_state['role']}**")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    do_logout()
    st.rerun()

st.sidebar.markdown("---")

# Role-aware navigation
accessible_pages = get_accessible_pages()
page = st.sidebar.radio("Navigate", accessible_pages)

st.sidebar.markdown("---")
# Load model evaluation metrics dynamically
try:
    with open(settings.METADATA_PATH, "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = {
        "model_name": "Random Forest",
        "precision": 0.96,
        "recall": 0.76,
        "pr_auc": 0.88,
        "training_samples": 284807
    }

precision_val = metadata.get('precision', 0.96)
precision_str = f"{precision_val * 100:.0f}%" if isinstance(precision_val, float) and precision_val <= 1.0 else str(precision_val)

recall_val = metadata.get('recall', 0.76)
recall_str = f"{recall_val * 100:.0f}%" if isinstance(recall_val, float) and recall_val <= 1.0 else str(recall_val)

pr_auc_val = metadata.get('pr_auc', 0.88)
pr_auc_str = f"{pr_auc_val:.2f}" if isinstance(pr_auc_val, float) else str(pr_auc_val)

st.sidebar.markdown(f"**Model:** {metadata.get('model_name', 'Random Forest')}")
st.sidebar.markdown(f"**Precision:** {precision_str}")
st.sidebar.markdown(f"**Recall:** {recall_str}")
st.sidebar.markdown(f"**PR-AUC:** {pr_auc_str}")
st.sidebar.markdown(f"**Trained on:** {metadata.get('training_samples', 284807):,} transactions")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 ERP Anomaly Detection Dashboard")
    st.markdown("Upload your transaction CSV to detect anomalies in real time.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload transaction CSV (creditcard.csv format)", type=["csv"])

    if uploaded:
        with st.spinner("Analyzing transactions..."):
            df_raw = pd.read_csv(uploaded)
            df = predict_df(df_raw)

        total = len(df)
        flagged = int(df['is_fraud'].sum())
        high_risk = int((df['risk_level'] == RISK_HIGH).sum())
        avg_score = float(df['anomaly_score'].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Flagged", f"{flagged:,}", delta=f"{flagged/total*100:.2f}%" if total > 0 else "0.00%", delta_color="inverse")
        c3.metric("High Risk", f"{high_risk:,}", delta_color="inverse")
        c4.metric("Avg Anomaly Score", f"{avg_score:.3f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Distribution")
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig_pie = px.pie(
                risk_counts, names='Risk Level', values='Count',
                color='Risk Level',
                color_discrete_map=RISK_COLORS,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Anomaly Score Distribution")
            fig_hist = px.histogram(
                df, x='anomaly_score', nbins=50,
                color_discrete_sequence=['#7c3aed']
            )
            fig_hist.add_vline(x=settings.FRAUD_THRESHOLD, line_dash="dash", line_color="red",
                               annotation_text="Fraud threshold")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Transaction Amount vs Anomaly Score")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x='Amount', y='anomaly_score', color='risk_level',
            color_discrete_map=RISK_COLORS,
            opacity=0.6
        )
        fig_scatter.add_hline(y=settings.FRAUD_THRESHOLD, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("🚨 Flagged Transactions (Top 50)")
        flagged_df = df[df['is_fraud']][
            ['Time', 'Amount', 'anomaly_score', 'risk_level']
        ].sort_values('anomaly_score', ascending=False).head(50).copy()
        flagged_df.columns = ['Posting Time', 'Amount', 'Anomaly Score', 'Risk Level']
        flagged_df['Anomaly Score'] = flagged_df['Anomaly Score'].round(4)
        st.dataframe(flagged_df, use_container_width=True, height=400)

        csv = flagged_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Flagged Transactions", csv, "flagged_transactions.csv", "text/csv")

    else:
        st.info("👆 Upload creditcard.csv from Kaggle to get started.")
        st.markdown("**Expected columns:** `Time`, `V1–V28`, `Amount`, `Class`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Transaction":
    st.title("🔍 Analyze Single Transaction")
    st.markdown("Enter transaction details to get an instant fraud prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        vendor_id = st.text_input("Vendor ID", value="V00123")
        department = st.selectbox("Department", ["Finance", "HR", "Procurement"])
    with col2:
        approved_by = st.selectbox("Approved By", ["mgr_01", "mgr_02", "mgr_03"])
        posting_time = st.number_input("Posting Time (seconds)", value=3600.0)
    with col3:
        amount = st.number_input("Transaction Amount", value=250.0, min_value=0.0)

    st.markdown("#### PCA Features (V1–V28)")
    st.caption("Leave as 0 for a normal transaction, or paste real fraud values from the dataset.")

    v_cols = st.columns(7)
    v_vals = {}
    for i in range(1, 29):
        with v_cols[(i - 1) % 7]:
            v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, key=f"v{i}", format="%.3f")

    st.markdown("")
    if st.button("🔍 Analyze Transaction", type="primary"):
        row = {'Time': posting_time, 'Amount': amount, 'Class': 0, **v_vals}
        df_single = pd.DataFrame([row])
        result = predict_df(df_single).iloc[0]
        score = float(result['anomaly_score'])
        risk = str(result['risk_level'])

        st.markdown("---")
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Anomaly Score", f"{score:.4f}")
        r2.metric("Fraud Detected", "YES ⚠️" if result['is_fraud'] else "NO ✅")
        r3.metric("Risk Level", risk)

        if risk == RISK_HIGH:
            st.error(RISK_MESSAGES[RISK_HIGH])
        elif risk == RISK_MEDIUM:
            st.warning(RISK_MESSAGES[RISK_MEDIUM])
        else:
            st.success(RISK_MESSAGES[RISK_LOW])

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={'text': "Anomaly Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#7c3aed"},
                'steps': [
                    {'range': [0.0, settings.FRAUD_THRESHOLD], 'color': '#d1fae5'},
                    {'range': [settings.FRAUD_THRESHOLD, settings.HIGH_RISK_THRESHOLD], 'color': '#fef3c7'},
                    {'range': [settings.HIGH_RISK_THRESHOLD, 1.0], 'color': '#fee2e2'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': settings.FRAUD_THRESHOLD}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Top Risk Factors")
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        feat_df = pd.DataFrame({
            'Feature': [feature_cols[i] for i in top_idx],
            'Importance': [round(float(importances[i]), 4) for i in top_idx]
        })
        fig_bar = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Purples')
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'},
                              margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Transaction Analysis")
    st.markdown("Upload a CSV — anomaly scores will be appended to every row.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df_raw):,}** transactions. Running predictions...")

        with st.spinner("Analyzing..."):
            df_result = predict_df(df_raw)

        st.success("✅ Analysis complete!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{len(df_result):,}")
        c2.metric("Flagged", f"{int(df_result['is_fraud'].sum()):,}")
        c3.metric("High Risk", f"{int((df_result['risk_level'] == RISK_HIGH).sum()):,}")

        st.subheader("Anomaly Scores Over Time")
        fig_line = px.line(
            df_result.reset_index(), x='index', y='anomaly_score',
            labels={'index': 'Transaction Index', 'anomaly_score': 'Anomaly Score'},
            color_discrete_sequence=['#7c3aed']
        )
        fig_line.add_hline(y=settings.FRAUD_THRESHOLD, line_dash="dash", line_color="red",
                           annotation_text="Fraud threshold")
        st.plotly_chart(fig_line, use_container_width=True)

        out_cols = ['Time', 'Amount', 'anomaly_score', 'is_fraud', 'risk_level']
        st.dataframe(df_result[out_cols].head(100), use_container_width=True)

        csv_out = df_result[out_cols].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Results", csv_out, "batch_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL STATS (Read-only, accessible to all authenticated users)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Stats":
    st.title("📈 Model Statistics & Telemetry")
    st.markdown("View model metadata and configuration. *(Read-only)*")
    st.markdown("---")

    try:
        repo_info = get_cached_repository()
        repo_info.load_artifacts()

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Type", "RandomForestClassifier")
        n_est = repo_info.model.n_estimators if hasattr(repo_info.model, "n_estimators") else "N/A"
        c2.metric("Estimators", str(n_est))
        c3.metric("Features", str(len(repo_info.feature_cols)))

        st.subheader("Feature List")
        feat_table = pd.DataFrame({
            "Index": list(range(len(repo_info.feature_cols))),
            "Feature Name": repo_info.feature_cols
        })
        st.dataframe(feat_table, use_container_width=True, height=400)

        st.subheader("Feature Importance (Top 15)")
        importances = repo_info.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        imp_df = pd.DataFrame({
            'Feature': [repo_info.feature_cols[i] for i in top_idx],
            'Importance': [round(float(importances[i]), 4) for i in top_idx]
        })
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Purples')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          margin=dict(t=10, b=10), height=450)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading model stats: {e}")
