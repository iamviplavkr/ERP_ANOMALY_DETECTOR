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
from datetime import datetime, timedelta
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
    "Admin":         ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats", "🚨 Alert Management", "🤝 Vendor Directory", "📈 BI Analytics"],
    "Finance User":  ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats"],
    "Fraud Analyst": ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis", "📈 Model Stats", "🚨 Alert Management", "🤝 Vendor Directory", "📈 BI Analytics"],
    "Auditor":       ["📊 Dashboard", "📈 Model Stats", "🚨 Alert Management", "🤝 Vendor Directory", "📈 BI Analytics"],
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
        # Build the payload for the /v1/predict API endpoint
        payload = {
            "vendor_id": vendor_id,
            "department": department,
            "approved_by": approved_by,
            "posting_time": posting_time,
            "transaction_amount": amount,
            **{f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)},
        }
        try:
            api_resp = requests.post(
                f"{API_BASE_URL}/v1/predict",
                json=payload,
                headers={"Authorization": f"Bearer {st.session_state.get('access_token', '')}"},
                timeout=15,
            )
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the backend API server. Is uvicorn running?")
            st.stop()

        if api_resp.status_code == 401:
            st.error("🔒 Session expired. Please log in again.")
            st.stop()
        elif api_resp.status_code == 403:
            st.error("🚫 You do not have permission to submit predictions.")
            st.stop()
        elif api_resp.status_code != 200:
            st.error(f"❌ Prediction failed ({api_resp.status_code}): {api_resp.text}")
            st.stop()

        pred = api_resp.json()
        score = float(pred.get("anomaly_score", 0.0))
        risk  = str(pred.get("risk_level", "LOW"))
        is_fraud = bool(pred.get("is_fraud", False))

        st.markdown("---")
        st.subheader("Prediction Result")
        st.success("✅ Transaction analyzed and **saved to database** — BI Analytics will reflect this immediately.")

        r1, r2, r3 = st.columns(3)
        r1.metric("Anomaly Score", f"{score:.4f}")
        r2.metric("Fraud Detected", "YES ⚠️" if is_fraud else "NO ✅")
        r3.metric("Risk Level", risk)

        if risk == RISK_HIGH:
            st.error(RISK_MESSAGES[RISK_HIGH])
        elif risk == RISK_MEDIUM:
            st.warning(RISK_MESSAGES[RISK_MEDIUM])
        else:
            st.success(RISK_MESSAGES[RISK_LOW])

        # Alert message from risk engine
        alert_msg = pred.get("alert_message", "")
        if alert_msg:
            st.info(f"💬 {alert_msg}")

        # Anomaly Score Gauge
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

        # Top Risk Factors from API response
        top_factors = pred.get("top_risk_factors") or []
        if top_factors:
            st.subheader("Top Risk Factors")
            feat_df = pd.DataFrame(top_factors)
            if "feature" in feat_df.columns and "importance" in feat_df.columns:
                feat_df = feat_df.rename(columns={"feature": "Feature", "importance": "Importance"})
                feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
                fig_bar = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Purples")
                fig_bar.update_layout(yaxis={"categoryorder": "total ascending"},
                                      margin=dict(t=10, b=10), height=350)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Fallback: use local model feature importances
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ALERT MANAGEMENT (Admin, Fraud Analyst: full; Auditor: read-only)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alert Management":
    st.title("🚨 Alert Management")
    st.markdown("View and manage risk intelligence alerts generated by the Risk Engine.")
    st.markdown("---")

    role = st.session_state.get("role", "")
    is_write_role = role in ("Admin", "Fraud Analyst")

    if is_write_role:
        st.success(f"✅ **{role}** — Full access: you can list, view, and update alert statuses.")
    else:
        st.info(f"👁️ **{role}** — Read-only access: you can view alerts but cannot change their status.")

    # Helper: build auth header
    def _auth_headers() -> dict:
        token = st.session_state.get("access_token", "")
        return {"Authorization": f"Bearer {token}"}

    # Helper: colored status badge HTML
    BADGE_COLORS = {
        "OPEN":          ("#ef4444", "white"),
        "INVESTIGATING": ("#f59e0b", "white"),
        "RESOLVED":      ("#10b981", "white"),
        "DISMISSED":     ("#6b7280", "white"),
    }

    def _status_badge(status: str) -> str:
        bg, fg = BADGE_COLORS.get(status.upper(), ("#94a3b8", "white"))
        return (
            f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:600;">'
            f'{status}</span>'
        )

    RISK_BADGE_COLORS = {
        "CRITICAL": ("#7f1d1d", "white"),
        "HIGH":     ("#ef4444", "white"),
        "MEDIUM":   ("#f59e0b", "white"),
        "LOW":      ("#10b981", "white"),
    }

    def _risk_badge(risk: str) -> str:
        bg, fg = RISK_BADGE_COLORS.get(risk.upper(), ("#94a3b8", "white"))
        return (
            f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:700;">'
            f'{risk}</span>'
        )

    # ── Filter Controls ──────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED"],
            key="alert_filter_status",
        )
    with col_f2:
        filter_risk = st.selectbox(
            "Filter by Risk Level",
            ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            key="alert_filter_risk",
        )
    with col_f3:
        if st.button("🔄 Refresh Alerts", use_container_width=True, key="alert_refresh_btn"):
            st.rerun()

    # ── Fetch Alerts from API ─────────────────────────────────────────────────
    params = {}
    if filter_status != "All":
        params["status"] = filter_status
    if filter_risk != "All":
        params["risk_level"] = filter_risk

    try:
        resp = requests.get(
            f"{API_BASE_URL}/v1/alerts",
            headers=_auth_headers(),
            params=params,
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Ensure the FastAPI server is running.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    if resp.status_code == 403:
        st.error("🚫 Access denied. You do not have permission to view alerts.")
        st.stop()
    elif resp.status_code != 200:
        st.error(f"Backend API returned HTTP {resp.status_code}: {resp.text}")
        st.stop()

    data = resp.json()
    alerts = data.get("alerts", [])
    total = data.get("total", 0)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown(f"**{total} alert(s) found** with current filters.")
    if total > 0:
        open_count = sum(1 for a in alerts if a.get("status") == "OPEN")
        invest_count = sum(1 for a in alerts if a.get("status") == "INVESTIGATING")
        resolved_count = sum(1 for a in alerts if a.get("status") in ("RESOLVED", "DISMISSED"))
        critical_count = sum(1 for a in alerts if a.get("risk_level") == "CRITICAL")
        high_count = sum(1 for a in alerts if a.get("risk_level") == "HIGH")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Alerts", total)
        m2.metric("🔴 Open", open_count)
        m3.metric("🟠 Investigating", invest_count)
        m4.metric("🟢 Resolved/Dismissed", resolved_count)
        m5.metric("🆘 Critical", critical_count)

    st.markdown("---")

    # ── Alert Table ───────────────────────────────────────────────────────────
    if not alerts:
        st.info("✅ No alerts match the current filters.")
    else:
        # Valid transitions map for write roles
        NEXT_STATUS = {
            "OPEN":          ["INVESTIGATING"],
            "INVESTIGATING": ["RESOLVED", "DISMISSED"],
            "RESOLVED":      [],
            "DISMISSED":     [],
        }

        for i, alert in enumerate(alerts):
            alert_id = alert.get("id", "")
            status = alert.get("status", "OPEN")
            risk = alert.get("risk_level", "UNKNOWN")
            rules = alert.get("rules_triggered", [])
            mitigation = alert.get("mitigation_action", "")
            created_at = alert.get("created_at", "")
            updated_at = alert.get("updated_at") or "—"
            prediction = alert.get("prediction", {}) or {}

            # Build header row
            header_col1, header_col2, header_col3 = st.columns([3, 2, 2])
            with header_col1:
                st.markdown(
                    f"**Alert** `{alert_id[:8]}…`  "
                    f"{_risk_badge(risk)}  {_status_badge(status)}",
                    unsafe_allow_html=True,
                )
            with header_col2:
                st.caption(f"Created: {created_at[:19].replace('T', ' ') if created_at else '—'}")
            with header_col3:
                st.caption(f"Updated: {str(updated_at)[:19].replace('T', ' ') if updated_at != '—' else '—'}")

            # Expandable details
            with st.expander(f"Details — Alert {alert_id[:8]}…", expanded=(i == 0 and total <= 3)):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Rules Triggered**")
                    for rule in rules:
                        st.markdown(f"- {rule}")
                    st.markdown("**Mitigation Recommended**")
                    st.info(mitigation or "—")

                with d2:
                    st.markdown("**Linked Prediction**")
                    if prediction:
                        tx = prediction.get("transaction", {}) or {}
                        st.markdown(f"- **Score:** `{prediction.get('anomaly_score', 'N/A')}`")
                        st.markdown(f"- **Is Fraud:** `{prediction.get('is_fraud', 'N/A')}`")
                        st.markdown(f"- **Model Version:** `{prediction.get('model_version', 'N/A')}`")
                        if tx:
                            st.markdown(f"- **Vendor:** `{tx.get('vendor_id', 'N/A')}`")
                            st.markdown(f"- **Department:** `{tx.get('department', 'N/A')}`")
                            st.markdown(f"- **Amount:** `${tx.get('transaction_amount', 'N/A'):,.2f}`")
                    else:
                        st.caption("Prediction detail not available.")

                # ── Status Update (write roles only) ──────────────────────────
                if is_write_role:
                    next_options = NEXT_STATUS.get(status, [])
                    if next_options:
                        st.markdown("**Update Status**")
                        upd_col1, upd_col2 = st.columns([2, 1])
                        with upd_col1:
                            new_status = st.selectbox(
                                "Transition to",
                                next_options,
                                key=f"status_sel_{alert_id}",
                            )
                        with upd_col2:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(
                                "✅ Confirm",
                                key=f"status_btn_{alert_id}",
                                use_container_width=True,
                            ):
                                try:
                                    upd_resp = requests.put(
                                        f"{API_BASE_URL}/v1/alerts/{alert_id}/status",
                                        json={"status": new_status},
                                        headers=_auth_headers(),
                                        timeout=10,
                                    )
                                    if upd_resp.status_code == 200:
                                        st.success(
                                            f"✅ Alert status updated to **{new_status}** successfully."
                                        )
                                        st.rerun()
                                    elif upd_resp.status_code == 422:
                                        detail = upd_resp.json().get("detail", "Invalid transition.")
                                        st.error(f"❌ Invalid transition: {detail}")
                                    elif upd_resp.status_code == 403:
                                        st.error("🚫 You do not have permission to update alert status.")
                                    else:
                                        st.error(
                                            f"API error {upd_resp.status_code}: {upd_resp.text}"
                                        )
                                except requests.exceptions.ConnectionError:
                                    st.error("Cannot connect to backend API.")
                    else:
                        st.markdown(
                            f"⚠️ Alert is in terminal state **{status}** — no further transitions possible."
                        )
                else:
                    # Read-only badge for Auditor
                    st.caption("🔒 You have read-only access to alerts.")

            st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — VENDOR DIRECTORY (Admin, Fraud Analyst: full; Auditor: read-only)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤝 Vendor Directory":
    st.title("🤝 Vendor Directory")
    st.markdown("Manage vendor risk profiles, reputation scores, and blacklist/watchlist status.")
    st.markdown("---")

    role = st.session_state.get("role", "")
    is_write_role = role in ("Admin", "Fraud Analyst")

    if is_write_role:
        st.success(f"✅ **{role}** — Full access: create, view, update, and delete vendor profiles.")
    else:
        st.info(f"👁️ **{role}** — Read-only access.")

    def _auth_headers_v() -> dict:
        return {"Authorization": f"Bearer {st.session_state.get('access_token', '')}"}

    # ── Reputation badge helpers ──────────────────────────────────────────────
    def _rep_badge(score: float) -> str:
        if score >= 80:
            bg, fg, label = "#10b981", "white", "Good"
        elif score >= 50:
            bg, fg, label = "#f59e0b", "white", "Fair"
        elif score >= 30:
            bg, fg, label = "#ef4444", "white", "Poor"
        else:
            bg, fg, label = "#7f1d1d", "white", "Critical"
        return (
            f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:700;">'
            f'{score:.0f}/100 — {label}</span>'
        )

    def _flag_badge(label: str, color: str) -> str:
        return (
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:600;">{label}</span>'
        )

    # ── Filter Controls ───────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        filter_blacklisted = st.selectbox(
            "Blacklist Filter", ["All", "Blacklisted Only", "Not Blacklisted"], key="vnd_bl_filter"
        )
    with col_f2:
        filter_watchlist = st.selectbox(
            "Watchlist Filter", ["All", "Watchlisted Only", "Not Watchlisted"], key="vnd_wl_filter"
        )
    with col_f3:
        if st.button("🔄 Refresh", use_container_width=True, key="vnd_refresh_btn"):
            st.rerun()

    params = {}
    if filter_blacklisted == "Blacklisted Only":
        params["is_blacklisted"] = "true"
    elif filter_blacklisted == "Not Blacklisted":
        params["is_blacklisted"] = "false"
    if filter_watchlist == "Watchlisted Only":
        params["is_watchlist"] = "true"
    elif filter_watchlist == "Not Watchlisted":
        params["is_watchlist"] = "false"

    # ── Fetch Vendors ─────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{API_BASE_URL}/v1/vendors",
            headers=_auth_headers_v(),
            params=params,
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API.")
        st.stop()

    if resp.status_code == 403:
        st.error("🚫 Access denied.")
        st.stop()
    elif resp.status_code != 200:
        st.error(f"Backend API returned HTTP {resp.status_code}: {resp.text}")
        st.stop()

    data = resp.json()
    vendors = data.get("vendors", [])
    total = data.get("total", 0)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    if total > 0:
        bl_count = sum(1 for v in vendors if v.get("is_blacklisted"))
        wl_count = sum(1 for v in vendors if v.get("is_watchlist"))
        avg_rep = sum(v.get("reputation_score", 100) for v in vendors) / total
        total_alerts = sum(v.get("historical_alerts_count", 0) for v in vendors)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Vendors", total)
        m2.metric("🚫 Blacklisted", bl_count)
        m3.metric("⚠️ Watchlist", wl_count)
        m4.metric("⭐ Avg Reputation", f"{avg_rep:.1f}")
        m5.metric("🔔 Total Alerts", total_alerts)
    else:
        st.info("No vendors found with the current filters.")

    st.markdown("---")

    # ── Create Vendor Form (write roles only) ─────────────────────────────────
    if is_write_role:
        with st.expander("➕ Register New Vendor", expanded=False):
            with st.form("create_vendor_form"):
                c1, c2 = st.columns(2)
                with c1:
                    new_vid = st.text_input("Vendor ID*", key="new_vid")
                    new_name = st.text_input("Vendor Name*", key="new_name")
                with c2:
                    new_rep = st.slider("Reputation Score", 0.0, 100.0, 90.0, 1.0, key="new_rep")
                    col_bl, col_wl = st.columns(2)
                    new_bl = col_bl.checkbox("Blacklisted", key="new_bl")
                    new_wl = col_wl.checkbox("Watchlisted", key="new_wl")
                submitted = st.form_submit_button("✅ Register Vendor", use_container_width=True)
                if submitted:
                    if not new_vid or not new_name:
                        st.error("Vendor ID and Name are required.")
                    else:
                        try:
                            cr = requests.post(
                                f"{API_BASE_URL}/v1/vendors",
                                json={
                                    "vendor_id": new_vid,
                                    "name": new_name,
                                    "reputation_score": new_rep,
                                    "is_blacklisted": new_bl,
                                    "is_watchlist": new_wl,
                                },
                                headers=_auth_headers_v(),
                                timeout=10,
                            )
                            if cr.status_code == 201:
                                st.success(f"✅ Vendor `{new_vid}` registered successfully.")
                                st.rerun()
                            elif cr.status_code == 400:
                                st.error(f"❌ {cr.json().get('detail', 'Vendor already exists.')}")
                            else:
                                st.error(f"API error {cr.status_code}: {cr.text}")
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to backend API.")

        st.markdown("---")

    # ── Vendor Cards ──────────────────────────────────────────────────────────
    if not vendors:
        st.info("No vendors match the current filters.")
    else:
        for vendor in vendors:
            vid = vendor.get("vendor_id", "")
            name = vendor.get("name", "—")
            rep = float(vendor.get("reputation_score", 100))
            is_bl = vendor.get("is_blacklisted", False)
            is_wl = vendor.get("is_watchlist", False)
            alert_cnt = vendor.get("historical_alerts_count", 0)
            total_tx = vendor.get("total_transactions_count", 0)
            fraud_rate = float(vendor.get("historical_fraud_rate", 0.0))
            last_tx = vendor.get("last_transaction_at", None)
            last_al = vendor.get("last_alert_at", None)

            # Header row
            hc1, hc2, hc3 = st.columns([3, 2, 2])
            with hc1:
                flags = ""
                if is_bl:
                    flags += f" {_flag_badge('BLACKLISTED', '#7f1d1d')}"
                if is_wl:
                    flags += f" {_flag_badge('WATCHLIST', '#b45309')}"
                st.markdown(
                    f"**`{vid}`** — {name} &nbsp; {_rep_badge(rep)} {flags}",
                    unsafe_allow_html=True,
                )
            with hc2:
                st.caption(f"Last Tx: {str(last_tx)[:19].replace('T',' ') if last_tx else '—'}")
            with hc3:
                st.caption(f"Last Alert: {str(last_al)[:19].replace('T',' ') if last_al else '—'}")

            with st.expander(f"Details — {vid}", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Historical Metrics**")
                    st.markdown(f"- **Total Transactions:** `{total_tx}`")
                    st.markdown(f"- **Historical Alerts:** `{alert_cnt}`")
                    st.markdown(f"- **Fraud Rate:** `{fraud_rate*100:.1f}%`")
                with d2:
                    st.markdown("**Risk Status**")
                    st.markdown(f"- **Reputation Score:** `{rep:.1f}/100`")
                    st.markdown(f"- **Blacklisted:** `{'Yes ⛔' if is_bl else 'No ✅'}`")
                    st.markdown(f"- **Watchlisted:** `{'Yes ⚠️' if is_wl else 'No ✅'}`")

                # Update form (write roles only)
                if is_write_role:
                    st.markdown("**Update Vendor**")
                    uf1, uf2, uf3 = st.columns([2, 1, 1])
                    with uf1:
                        new_rep_upd = st.slider(
                            "Reputation Score", 0.0, 100.0, rep, 1.0,
                            key=f"rep_{vid}"
                        )
                    with uf2:
                        new_bl_upd = st.checkbox("Blacklisted", value=is_bl, key=f"bl_{vid}")
                    with uf3:
                        new_wl_upd = st.checkbox("Watchlisted", value=is_wl, key=f"wl_{vid}")

                    ub1, ub2 = st.columns([1, 1])
                    with ub1:
                        if st.button("💾 Save Changes", key=f"upd_{vid}", use_container_width=True):
                            try:
                                ur = requests.put(
                                    f"{API_BASE_URL}/v1/vendors/{vid}",
                                    json={
                                        "reputation_score": new_rep_upd,
                                        "is_blacklisted": new_bl_upd,
                                        "is_watchlist": new_wl_upd,
                                    },
                                    headers=_auth_headers_v(),
                                    timeout=10,
                                )
                                if ur.status_code == 200:
                                    st.success(f"✅ Vendor `{vid}` updated.")
                                    st.rerun()
                                else:
                                    st.error(f"Error {ur.status_code}: {ur.text}")
                            except requests.exceptions.ConnectionError:
                                st.error("Cannot connect to backend API.")
                    with ub2:
                        if st.button("🗑️ Delete Vendor", key=f"del_{vid}", use_container_width=True, type="secondary"):
                            try:
                                dr = requests.delete(
                                    f"{API_BASE_URL}/v1/vendors/{vid}",
                                    headers=_auth_headers_v(),
                                    timeout=10,
                                )
                                if dr.status_code == 204:
                                    st.success(f"✅ Vendor `{vid}` deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"Error {dr.status_code}: {dr.text}")
                            except requests.exceptions.ConnectionError:
                                st.error("Cannot connect to backend API.")

            st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — EXECUTIVE ANALYTICS AND BI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 BI Analytics":
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    st.title("📈 Executive Analytics & BI Dashboard")
    st.markdown("Comprehensive operational KPIs, fraud trends, departmental metrics, and risk distributions.")
    st.markdown("---")

    role = st.session_state.get("role", "")
    if role not in ("Admin", "Fraud Analyst", "Auditor"):
        st.error("🚫 Access denied. You do not have permission to view the BI dashboard.")
        st.stop()

    def _auth_headers_a() -> dict:
        return {"Authorization": f"Bearer {st.session_state.get('access_token', '')}"}

    # ── Controls ──────────────────────────────────────────────────────────────
    col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
    with col_c1:
        # Default to 30 days ago
        default_start = datetime.now() - timedelta(days=30)
        start_date = st.date_input("Start Date", default_start, key="bi_start_dt")
    with col_c2:
        end_date = st.date_input("End Date", datetime.now(), key="bi_end_dt")
    with col_c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Analytics", use_container_width=True, key="bi_refresh_btn"):
            st.rerun()

    # Convert to query format
    start_str = start_date.isoformat() + "T00:00:00Z"
    end_str = end_date.isoformat() + "T23:59:59Z"
    params = {"start_date": start_str, "end_date": end_str}

    # ── Fetch KPI overview data ───────────────────────────────────────────────
    try:
        overview_resp = requests.get(
            f"{API_BASE_URL}/v1/analytics/overview",
            headers=_auth_headers_a(),
            params=params,
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend API server.")
        st.stop()

    if overview_resp.status_code == 403:
        st.error("🚫 Access denied.")
        st.stop()
    elif overview_resp.status_code != 200:
        st.error(f"Failed to fetch overview: {overview_resp.text}")
        st.stop()

    kpis = overview_resp.json()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Total Transactions",
            value=f"{kpis.get('total_transactions', 0):,}",
        )
    with k2:
        st.metric(
            label="Total Volume",
            value=f"${kpis.get('total_amount', 0.0):,.2f}",
        )
    with k3:
        st.metric(
            label="Anomalies Flagged",
            value=f"{kpis.get('flagged_anomalies', 0):,}",
            delta=f"{kpis.get('anomaly_rate', 0.0)*100:.2f}% rate",
            delta_color="inverse",
        )
    with k4:
        st.metric(
            label="Open Alerts",
            value=f"{kpis.get('open_alerts', 0):,}",
            delta="Requires Action" if kpis.get("open_alerts", 0) > 0 else "Clear",
            delta_color="off" if kpis.get("open_alerts", 0) == 0 else "normal",
        )

    st.markdown("---")

    # ── Tabs for Visualizations ───────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Trends & Risk Distribution",
        "🏢 Departmental & Vendor Analysis",
        "🤖 Alert Lifecycle & Model Performance",
    ])

    # ── TAB 1: Trends & Risk Distribution ─────────────────────────────────────
    with tab1:
        st.subheader("Fraud & Transaction Trends")
        try:
            trends_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/trends",
                headers=_auth_headers_a(),
                params=params,
                timeout=10,
            )
            trends_data = trends_resp.json() if trends_resp.status_code == 200 else []
        except Exception:
            trends_data = []

        if trends_data:
            df_trends = pd.DataFrame(trends_data)
            df_trends["date"] = pd.to_datetime(df_trends["date"])
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=df_trends["date"],
                y=df_trends["count"],
                name="Total Transactions",
                yaxis="y1",
                marker_color="#3b82f6",
                opacity=0.75,
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_trends["date"],
                y=df_trends["average_anomaly_score"],
                name="Avg Anomaly Score",
                yaxis="y2",
                line=dict(color="#ef4444", width=3),
            ))

            fig_trend.update_layout(
                title="Daily Volume vs Average Anomaly Score",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Transaction Count", side="left"),
                yaxis2=dict(title="Avg Anomaly Score", side="right", overlaying="y", range=[0, 1]),
                legend=dict(x=0.01, y=0.99),
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No trend data available for this range.")

        st.markdown("---")

        st.subheader("Risk Distribution Analysis")
        col_rd1, col_rd2 = st.columns(2)
        try:
            risk_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/risk-distribution",
                headers=_auth_headers_a(),
                params=params,
                timeout=10,
            )
            risk_data = risk_resp.json() if risk_resp.status_code == 200 else []
        except Exception:
            risk_data = []

        if risk_data:
            df_risk = pd.DataFrame(risk_data)
            
            # Count pie chart
            with col_rd1:
                fig_pie = px.pie(
                    df_risk,
                    values="count",
                    names="risk_level",
                    title="Transaction Counts by Risk Level",
                    color="risk_level",
                    color_discrete_map={
                        "CRITICAL": "#7f1d1d",
                        "HIGH": "#ef4444",
                        "MEDIUM": "#f59e0b",
                        "LOW": "#10b981",
                    },
                    hole=0.4,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Amount pie chart
            with col_rd2:
                fig_pie_amt = px.pie(
                    df_risk,
                    values="total_amount",
                    names="risk_level",
                    title="Total Amount ($) by Risk Level",
                    color="risk_level",
                    color_discrete_map={
                        "CRITICAL": "#7f1d1d",
                        "HIGH": "#ef4444",
                        "MEDIUM": "#f59e0b",
                        "LOW": "#10b981",
                    },
                    hole=0.4,
                )
                st.plotly_chart(fig_pie_amt, use_container_width=True)
        else:
            st.info("No risk distribution data available.")

    # ── TAB 2: Departmental & Vendor Analysis ────────────────────────────────
    with tab2:
        st.subheader("Department-wise Activity & Risk")
        try:
            dept_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/departments",
                headers=_auth_headers_a(),
                params=params,
                timeout=10,
            )
            dept_data = dept_resp.json() if dept_resp.status_code == 200 else []
        except Exception:
            dept_data = []

        if dept_data:
            df_dept = pd.DataFrame(dept_data)
            
            # Grouped bar chart
            fig_dept = px.bar(
                df_dept,
                x="department",
                y="total_amount",
                color="average_anomaly_score",
                title="Total Amount and Average Anomaly Score by Department",
                labels={
                    "total_amount": "Total Amount ($)",
                    "average_anomaly_score": "Avg Anomaly Score",
                    "department": "Department",
                },
                color_continuous_scale=px.colors.sequential.Reds,
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            st.info("No departmental data available.")

        st.markdown("---")

        st.subheader("Vendor Risk Rankings")
        
        # Paginated rankings
        rank_sort_by = st.selectbox(
            "Sort Rankings By",
            [
                ("reputation_score", "Reputation Score (Lower = More risky)"),
                ("historical_alerts_count", "Historical Alerts Count (Higher = More risky)"),
                ("historical_fraud_rate", "Historical Fraud Rate (Higher = More risky)"),
                ("total_transactions_count", "Total Transactions Volume"),
            ],
            index=0,
            format_func=lambda x: x[1],
            key="bi_sort_select",
        )
        
        # Sorting order
        rank_sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, index=0 if rank_sort_by[0] == "reputation_score" else 1, key="bi_order_radio")
        order_str = "asc" if rank_sort_order == "Ascending" else "desc"

        # Pagination params
        v_limit = 5
        v_page = st.number_input("Page", min_value=1, value=1, step=1, key="bi_page_num")
        v_offset = (v_page - 1) * v_limit

        try:
            v_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/vendors",
                headers=_auth_headers_a(),
                params={
                    "start_date": start_str,
                    "end_date": end_str,
                    "limit": v_limit,
                    "offset": v_offset,
                    "sort_by": rank_sort_by[0],
                    "sort_order": order_str,
                },
                timeout=10,
            )
            v_rankings = v_resp.json() if v_resp.status_code == 200 else {"total": 0, "vendors": []}
        except Exception:
            v_rankings = {"total": 0, "vendors": []}

        v_list = v_rankings.get("vendors", [])
        v_total = v_rankings.get("total", 0)

        if v_list:
            df_v = pd.DataFrame(v_list)
            # Reorder columns beautifully
            df_v_show = df_v[[
                "vendor_id",
                "name",
                "reputation_score",
                "historical_alerts_count",
                "total_transactions_count",
                "historical_fraud_rate",
                "is_blacklisted",
                "is_watchlist",
            ]].copy()
            df_v_show.columns = [
                "Vendor ID",
                "Name",
                "Reputation (0-100)",
                "Alerts Count",
                "Tx Count",
                "Fraud Rate",
                "Blacklisted",
                "Watchlist",
            ]
            st.dataframe(df_v_show, use_container_width=True, hide_index=True)
            st.caption(f"Showing page {v_page} ({len(v_list)} records out of {v_total} total registered vendors).")
        else:
            st.info("No vendor metrics available.")

    # ── TAB 3: Alert Lifecycle & Model Performance ───────────────────────────
    with tab3:
        st.subheader("Alert Lifecycle Statuses")
        try:
            alerts_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/alerts-lifecycle",
                headers=_auth_headers_a(),
                params=params,
                timeout=10,
            )
            alerts_data = alerts_resp.json() if alerts_resp.status_code == 200 else []
        except Exception:
            alerts_data = []

        if alerts_data:
            df_alerts = pd.DataFrame(alerts_data)
            fig_al = px.bar(
                df_alerts,
                x="status",
                y="count",
                title="Active & Resolved Alert Distribution",
                color="status",
                color_discrete_map={
                    "OPEN": "#ef4444",
                    "INVESTIGATING": "#f59e0b",
                    "RESOLVED": "#10b981",
                    "DISMISSED": "#6b7280",
                },
            )
            st.plotly_chart(fig_al, use_container_width=True)
        else:
            st.info("No alert lifecycle data available for this range.")

        st.markdown("---")

        st.subheader("Model Performance Summary")
        try:
            perf_resp = requests.get(
                f"{API_BASE_URL}/v1/analytics/model-performance",
                headers=_auth_headers_a(),
                params=params,
                timeout=10,
            )
            perf_data = perf_resp.json() if perf_resp.status_code == 200 else {}
        except Exception:
            perf_data = {}

        if perf_data and perf_data.get("total_predictions", 0) > 0:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Avg Score", f"{perf_data.get('average_score', 0.0):.4f}")
            pc2.metric("Max Score", f"{perf_data.get('max_score', 0.0):.4f}")
            pc3.metric("Anomalous Transactions count (Score >= 0.5)", f"{perf_data.get('high_confidence_anomalies', 0):,}")

            # Plot distribution
            df_perf_dist = pd.DataFrame([
                {"Label": "High Risk (Score >= 0.5)", "Count": perf_data.get("high_confidence_anomalies", 0)},
                {"Label": "Low Risk (Score < 0.5)", "Count": perf_data.get("low_confidence_normal", 0)},
            ])
            fig_perf = px.pie(
                df_perf_dist,
                values="Count",
                names="Label",
                title="Prediction Confidence Score Distribution Split",
                hole=0.4,
                color="Label",
                color_discrete_map={
                    "High Risk (Score >= 0.5)": "#ef4444",
                    "Low Risk (Score < 0.5)": "#3b82f6",
                }
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No model performance metrics available.")

    st.markdown("---")

    # ── CSV Export Controls ───────────────────────────────────────────────────
    st.subheader("📥 Export Transaction Logs")
    st.markdown("Export all predictions and associated transaction details matching the current date filters to a CSV file.")
    
    # Sort choice for export
    exp_sort_by = st.selectbox(
        "Sort Log By",
        [
            ("created_at", "Date Created"),
            ("transaction_amount", "Transaction Amount"),
            ("anomaly_score", "Anomaly Score"),
        ],
        index=0,
        format_func=lambda x: x[1],
        key="bi_exp_sort_select",
    )
    exp_sort_order = st.radio("Export Order", ["Ascending", "Descending"], index=1, horizontal=True, key="bi_exp_order_radio")
    exp_order_str = "asc" if exp_sort_order == "Ascending" else "desc"

    export_url = f"{API_BASE_URL}/v1/analytics/export?start_date={start_str}&end_date={end_str}&sort_by={exp_sort_by[0]}&sort_order={exp_order_str}"
    
    try:
        exp_resp = requests.get(
            export_url,
            headers=_auth_headers_a(),
            timeout=15,
        )
        if exp_resp.status_code == 200:
            st.download_button(
                label="💾 Download CSV Report",
                data=exp_resp.content,
                file_name=f"erp_anomaly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="bi_download_btn",
            )
        else:
            st.error(f"Failed to prepare CSV download: HTTP {exp_resp.status_code}")
    except Exception as exc:
        st.error(f"Could not connect to fetch CSV: {exc}")



