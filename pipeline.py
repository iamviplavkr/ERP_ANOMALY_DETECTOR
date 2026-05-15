import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import shap
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & rename to ERP context ──────────────────────────────────────────
df = pd.read_csv('creditcard.csv')
df = df.rename(columns={
    'Time':   'posting_time',
    'Amount': 'transaction_amount',
    'Class':  'is_fraud'
})
np.random.seed(42)
df['department']  = np.random.choice(['Finance', 'HR', 'Procurement'], len(df))
df['vendor_id']   = 'V' + df.index.astype(str).str.zfill(5)
df['approved_by'] = np.random.choice(['mgr_01', 'mgr_02', 'mgr_03'], len(df))

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df['log_amount']    = np.log1p(df['transaction_amount'])
df['hour_of_day']   = (df['posting_time'] % 86400) // 3600
df['is_night']      = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
df['amount_zscore'] = (df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std()

# ── 3. Prepare Features ───────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c.startswith('V')] + \
               ['log_amount', 'hour_of_day', 'is_night', 'amount_zscore']
X = df[feature_cols].values
y = df['is_fraud'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Isolation Forest (Unsupervised) ────────────────────────────────────────
print("\n===== Isolation Forest =====")
iso = IsolationForest(contamination=0.002, random_state=42, n_estimators=100)
iso.fit(X_train)
iso_preds = (iso.predict(X_test) == -1).astype(int)
print(classification_report(y_test, iso_preds, target_names=['Normal', 'Fraud']))

# ── 5. Random Forest (Supervised) ────────────────────────────────────────────
print("===== Random Forest =====")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, rf_preds, target_names=['Normal', 'Fraud']))

p, r, _ = precision_recall_curve(y_test, rf_proba)
print(f"PR-AUC: {auc(r, p):.4f}")

# ── 6. SHAP Explainability ────────────────────────────────────────────────────
print("\nComputing SHAP values...")
sample_idx = np.random.choice(len(X_test), 300, replace=False)
explainer  = shap.TreeExplainer(rf)
shap_out   = explainer.shap_values(X_test[sample_idx])
sv = shap_out[1] if isinstance(shap_out, list) else shap_out[:, :, 1]

importances = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols)
print("\nTop 10 features by SHAP importance:")
print(importances.sort_values(ascending=False).head(10))

# ── 7. Flagged Transactions ───────────────────────────────────────────────────
df_test = df.iloc[len(X_train):].reset_index(drop=True).copy()
df_test['anomaly_score'] = rf_proba
df_test['flagged']       = rf_preds

print("\nSample flagged transactions:")
flagged = df_test[df_test['flagged'] == 1][
    ['vendor_id', 'department', 'approved_by',
     'transaction_amount', 'anomaly_score', 'is_fraud']
].head(10)
print(flagged.to_string(index=False))

# Save flagged transactions to CSV
df_test[df_test['flagged'] == 1].to_csv('flagged_transactions.csv', index=False)
print("\nFlagged transactions saved to flagged_transactions.csv")