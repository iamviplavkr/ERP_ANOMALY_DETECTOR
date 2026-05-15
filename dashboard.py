import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Make sure model.pkl, scaler.pkl, feature_cols.pkl are in the same folder as dashboard.py")
    st.stop()

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['log_amount']    = np.log1p(df['Amount'])
    df['hour_of_day']   = (df['Time'] % 86400) // 3600
    df['is_night']      = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-9)
    return df

def predict_df(df):
    df_feat = engineer_features(df)
    X       = df_feat[feature_cols].values
    X_sc    = scaler.transform(X)
    proba   = model.predict_proba(X_sc)[:, 1]
    df_feat['anomaly_score'] = proba
    df_feat['is_fraud']      = proba >= 0.5
    df_feat['risk_level']    = pd.cut(
        proba,
        bins=[-0.01, 0.5, 0.8, 1.01],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    return df_feat

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 ERP Anomaly Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Single Transaction", "📁 Batch Analysis"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Precision:** 96%")
st.sidebar.markdown("**Recall:** 76%")
st.sidebar.markdown("**PR-AUC:** 0.88")
st.sidebar.markdown("**Trained on:** 284,807 transactions")

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
            df     = predict_df(df_raw)

        total     = len(df)
        flagged   = int(df['is_fraud'].sum())
        high_risk = int((df['risk_level'] == 'HIGH').sum())
        avg_score = float(df['anomaly_score'].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Flagged", f"{flagged:,}", delta=f"{flagged/total*100:.2f}%", delta_color="inverse")
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
                color_discrete_map={'HIGH':'#ef4444','MEDIUM':'#f59e0b','LOW':'#10b981'},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Anomaly Score Distribution")
            fig_hist = px.histogram(
                df, x='anomaly_score', nbins=50,
                color_discrete_sequence=['#7c3aed']
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                               annotation_text="Fraud threshold")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Transaction Amount vs Anomaly Score")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x='Amount', y='anomaly_score', color='risk_level',
            color_discrete_map={'HIGH':'#ef4444','MEDIUM':'#f59e0b','LOW':'#10b981'},
            opacity=0.6
        )
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("🚨 Flagged Transactions (Top 50)")
        flagged_df = df[df['is_fraud']][
            ['Time','Amount','anomaly_score','risk_level']
        ].sort_values('anomaly_score', ascending=False).head(50).copy()
        flagged_df.columns = ['Posting Time','Amount','Anomaly Score','Risk Level']
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
        vendor_id    = st.text_input("Vendor ID", value="V00123")
        department   = st.selectbox("Department", ["Finance", "HR", "Procurement"])
    with col2:
        approved_by  = st.selectbox("Approved By", ["mgr_01", "mgr_02", "mgr_03"])
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
        row       = {'Time': posting_time, 'Amount': amount, 'Class': 0, **v_vals}
        df_single = pd.DataFrame([row])
        result    = predict_df(df_single).iloc[0]
        score     = float(result['anomaly_score'])
        risk      = str(result['risk_level'])

        st.markdown("---")
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Anomaly Score", f"{score:.4f}")
        r2.metric("Fraud Detected", "YES ⚠️" if result['is_fraud'] else "NO ✅")
        r3.metric("Risk Level", risk)

        if risk == 'HIGH':
            st.error("⚠️ HIGH RISK — Immediate review required.")
        elif risk == 'MEDIUM':
            st.warning("🔶 MEDIUM RISK — Manual verification recommended.")
        else:
            st.success("✅ LOW RISK — Transaction appears normal.")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={'text': "Anomaly Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar':  {'color': "#7c3aed"},
                'steps': [
                    {'range': [0.0, 0.5], 'color': '#d1fae5'},
                    {'range': [0.5, 0.8], 'color': '#fef3c7'},
                    {'range': [0.8, 1.0], 'color': '#fee2e2'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Top Risk Factors")
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        feat_df = pd.DataFrame({
            'Feature':    [feature_cols[i] for i in top_idx],
            'Importance': [round(float(importances[i]), 4) for i in top_idx]
        })
        fig_bar = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Purples')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'},
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
        c1.metric("Total",     f"{len(df_result):,}")
        c2.metric("Flagged",   f"{int(df_result['is_fraud'].sum()):,}")
        c3.metric("High Risk", f"{int((df_result['risk_level']=='HIGH').sum()):,}")

        st.subheader("Anomaly Scores Over Time")
        fig_line = px.line(
            df_result.reset_index(), x='index', y='anomaly_score',
            labels={'index': 'Transaction Index', 'anomaly_score': 'Anomaly Score'},
            color_discrete_sequence=['#7c3aed']
        )
        fig_line.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="Fraud threshold")
        st.plotly_chart(fig_line, use_container_width=True)

        out_cols = ['Time', 'Amount', 'anomaly_score', 'is_fraud', 'risk_level']
        st.dataframe(df_result[out_cols].head(100), use_container_width=True)

        csv_out = df_result[out_cols].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Results", csv_out, "batch_results.csv", "text/csv")