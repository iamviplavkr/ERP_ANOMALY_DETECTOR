import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.repositories.model_repository import ModelRepository
from ml.features.feature_engineering import engineer_features_from_request, engineer_features_from_df
from backend.schemas.transaction import TransactionRequest

def run_analysis():
    print("=" * 80)
    print("1. VERIFY MODEL INTEGRITY")
    print("=" * 80)
    
    # Absolute paths
    print(f"MODEL_PATH: {os.path.abspath(settings.MODEL_PATH)}")
    print(f"SCALER_PATH: {os.path.abspath(settings.SCALER_PATH)}")
    print(f"FEATURE_COLS_PATH: {os.path.abspath(settings.FEATURE_COLS_PATH)}")
    print(f"METADATA_PATH: {os.path.abspath(settings.METADATA_PATH)}")
    
    print("\nRoot directory artifact existence and sizes:")
    for f in ['model.pkl', 'scaler.pkl', 'feature_cols.pkl']:
        p = ROOT / f
        if p.exists():
            print(f"  {f}: {p.stat().st_size} bytes")
        else:
            print(f"  {f}: NOT FOUND")
            
    print("\nartifacts/ directory artifact existence and sizes:")
    for f in ['model.pkl', 'scaler.pkl', 'feature_cols.pkl', 'model_metadata.json']:
        p = ROOT / 'artifacts' / f
        if p.exists():
            print(f"  artifacts/{f}: {p.stat().st_size} bytes")
        else:
            print(f"  artifacts/{f}: NOT FOUND")

    repo = ModelRepository()
    repo.load_artifacts(force=True)
    
    model = repo.model
    scaler = repo.scaler
    feature_cols = repo.feature_cols
    
    print(f"\nLoaded Model Type: {type(model)}")
    print(f"Model Params: {model.get_params()}")
    print(f"Scaler Mean shape: {scaler.mean_.shape}")
    print(f"Scaler Mean: {scaler.mean_}")
    print(f"Scaler Scale: {scaler.scale_}")
    
    if os.path.exists(settings.METADATA_PATH):
        with open(settings.METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata: {json.dumps(metadata, indent=2)}")

    print("\n" + "=" * 80)
    print("2. VERIFY FEATURE CONSISTENCY")
    print("=" * 80)
    print(f"Loaded feature_cols: {feature_cols}")
    
    # Load dataset to test feature engineering consistency
    data_path = settings.DATA_PATH
    df_raw = pd.read_csv(data_path)
    
    # Pick a known fraud transaction (Class == 1)
    fraud_df = df_raw[df_raw['Class'] == 1].iloc[0]
    print(f"\nSelected Fraud Sample (index {fraud_df.name}):")
    print(f"  Time: {fraud_df['Time']}, Amount: {fraud_df['Amount']}, Class: {fraud_df['Class']}")
    
    # Feature engineering via single request
    txn_kwargs = {
        'vendor_id': 'V00001',
        'department': 'Finance',
        'approved_by': 'mgr_01',
        'posting_time': float(fraud_df['Time']),
        'transaction_amount': float(fraud_df['Amount'])
    }
    for i in range(1, 29):
        txn_kwargs[f'V{i}'] = float(fraud_df[f'V{i}'])
    
    txn_req = TransactionRequest(**txn_kwargs)
    feat_vector_req = engineer_features_from_request(txn_req)
    print(f"\nEngineered feature vector (Request API):\n{feat_vector_req}")
    
    # Feature engineering via DataFrame (Training / Batch)
    df_temp = df_raw.iloc[[fraud_df.name]].copy()
    df_feat = engineer_features_from_df(df_temp)
    feat_vector_df = df_feat[feature_cols].values
    print(f"\nEngineered feature vector (DataFrame batch):\n{feat_vector_df}")
    
    # Training pipeline feature engineering logic check:
    # In train.py:
    # df_feat = engineer_features_from_df(df_temp)
    # df['log_amount'] = df_feat['log_amount']
    # df['hour_of_day'] = df_feat['hour_of_day']
    # df['is_night'] = df_feat['is_night']
    # df['amount_zscore'] = df_feat['amount_zscore']
    
    print("\nComparing Feature Vector Differences (Request vs DataFrame single row):")
    diff = feat_vector_req - feat_vector_df
    print(f"Max absolute diff: {np.max(np.abs(diff))}")
    for col, v_req, v_df in zip(feature_cols, feat_vector_req[0], feat_vector_df[0]):
        print(f"  {col:15s}: API={v_req:12.6f} | DF={v_df:12.6f} | diff={abs(v_req - v_df):.6f}")

    # Now let's compare with scaling
    feat_sc_req = scaler.transform(feat_vector_req)
    feat_sc_df = scaler.transform(feat_vector_df)
    print(f"\nScaled feature vector (Request API):\n{feat_sc_req}")
    print(f"Scaled feature vector (DataFrame batch):\n{feat_sc_df}")

    print("\n" + "=" * 80)
    print("3. VERIFY SAVED MODEL PERFORMANCE ON FULL DATASET")
    print("=" * 80)
    
    # Engineer features on entire creditcard.csv using the dataset
    df_full = df_raw.copy()
    df_full_feat = engineer_features_from_df(df_full)
    X_full = df_full_feat[feature_cols].values
    y_full = df_full_feat['Class'].values
    
    X_full_sc = scaler.transform(X_full)
    
    y_pred_full = model.predict(X_full_sc)
    y_proba_full = model.predict_proba(X_full_sc)[:, 1]
    
    print("\nClassification Report (Full Dataset):")
    print(classification_report(y_full, y_pred_full, target_names=['Normal', 'Fraud'], digits=4))
    
    cm = confusion_matrix(y_full, y_pred_full)
    print("Confusion Matrix:")
    print(cm)
    
    prec = precision_score(y_full, y_pred_full)
    rec = recall_score(y_full, y_pred_full)
    precision_curve, recall_curve, _ = precision_recall_curve(y_full, y_proba_full)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    
    total_fraud = int(np.sum(y_full == 1))
    fraud_pred_as_fraud = int(np.sum((y_full == 1) & (y_pred_full == 1)))
    fraud_pred_as_normal = int(np.sum((y_full == 1) & (y_pred_full == 0)))
    print(f"Total fraud samples (Class = 1): {total_fraud}")
    print(f"Predicted as fraud (threshold >= 0.5): {fraud_pred_as_fraud}")
    print(f"Predicted as normal (threshold < 0.5): {fraud_pred_as_normal}")

    print("\n" + "=" * 80)
    print("4. INSPECT PROBABILITY DISTRIBUTION")
    print("=" * 80)
    
    fraud_probas = y_proba_full[y_full == 1]
    normal_probas = y_proba_full[y_full == 0]
    
    print(f"Fraud Probabilities:  min={fraud_probas.min():.6f}, max={fraud_probas.max():.6f}, mean={fraud_probas.mean():.6f}, median={np.median(fraud_probas):.6f}")
    print(f"Normal Probabilities: min={normal_probas.min():.6f}, max={normal_probas.max():.6f}, mean={normal_probas.mean():.6f}, median={np.median(normal_probas):.6f}")
    
    print("\nHistogram of Fraud Probabilities:")
    counts, bin_edges = np.histogram(fraud_probas, bins=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for i in range(len(counts)):
        print(f"  [{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]: {counts[i]}")

    print("\nHistogram of Normal Probabilities:")
    counts_n, bin_edges_n = np.histogram(normal_probas, bins=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for i in range(len(counts_n)):
        print(f"  [{bin_edges_n[i]:.2f} - {bin_edges_n[i+1]:.2f}]: {counts_n[i]}")

    print("\n" + "=" * 80)
    print("5. VERIFY PREDICTION LOGIC ON KNOWN FRAUD SAMPLES")
    print("=" * 80)
    
    fraud_indices = np.where(y_full == 1)[0][:10]
    for idx in fraud_indices:
        raw_x = X_full[idx]
        sc_x = X_full_sc[idx].reshape(1, -1)
        pred = model.predict(sc_x)[0]
        proba = model.predict_proba(sc_x)[0]
        anomaly_score = float(proba[1])
        
        # Risk level determination logic from backend/utils/helpers.py or prediction_service.py
        if anomaly_score >= 0.8:
            risk_level = "CRITICAL"
        elif anomaly_score >= 0.5:
            risk_level = "HIGH"
        elif anomaly_score >= 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        print(f"Index {idx:6d} | Amount: {df_raw.iloc[idx]['Amount']:8.2f} | predict: {pred} | proba: [0]={proba[0]:.4f}, [1]={proba[1]:.4f} | anomaly_score: {anomaly_score:.4f} | risk: {risk_level}")

    print("\n" + "=" * 80)
    print("6. INSPECT FLAGGED_TRANSACTIONS.CSV GENERATION & ALIGNMENT")
    print("=" * 80)
    
    from sklearn.model_selection import train_test_split
    X_train_check, X_test_check, y_train_check, y_test_check = train_test_split(
        X_full_sc, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print(f"Total rows in df: {len(df_full)}")
    print(f"len(X_train): {len(X_train_check)}, len(X_test): {len(X_test_check)}")
    
    df_test_incorrect = df_full.iloc[len(X_train_check):].reset_index(drop=True)
    test_indices_actual = df_full.index[len(X_train_check):] # wait, train_test_split returns arrays, not original indices unless split on indices
    
    # In run_pipeline.py:
    # df_test = df.iloc[len(X_train):].reset_index(drop=True).copy()
    # flagged_df = evaluator.build_flagged_df(df_test, rf_preds, rf_proba)
    # Let's check y of df.iloc[len(X_train):] vs y_test!
    y_df_test_incorrect = df_full.iloc[len(X_train_check):]['Class'].values
    print(f"Fraud count in y_test_check (actual test set): {np.sum(y_test_check)}")
    print(f"Fraud count in df.iloc[len(X_train):]['Class'] (sliced tail): {np.sum(y_df_test_incorrect)}")
    print(f"Do y_test_check and df_test_incorrect['Class'] match? {np.array_equal(y_test_check, y_df_test_incorrect)}")
    
    # Check overlap/matching between sliced tail and actual test indices
    # Let's train_test_split on indices
    indices = np.arange(len(df_full))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_full)
    
    match_count = np.sum(idx_test == indices[len(X_train_check):])
    print(f"Number of indices matching between actual test split and sliced tail: {match_count} out of {len(idx_test)}")

    # Check root model vs artifacts model differences!
    root_model_path = ROOT / 'model.pkl'
    art_model_path = ROOT / 'artifacts' / 'model.pkl'
    if root_model_path.exists() and art_model_path.exists():
        with open(root_model_path, 'rb') as f:
            m_root = pickle.load(f)
        with open(art_model_path, 'rb') as f:
            m_art = pickle.load(f)
        print(f"\nRoot model.pkl type: {type(m_root)}, n_estimators: {m_root.n_estimators}")
        print(f"Artifacts model.pkl type: {type(m_art)}, n_estimators: {m_art.n_estimators}")
        
        # Test fraud predictions on root model vs artifacts model!
        preds_root = m_root.predict_proba(X_full_sc[:100])[:, 1]
        preds_art = m_art.predict_proba(X_full_sc[:100])[:, 1]
        print(f"Sample 10 predictions root model: {preds_root[:10]}")
        print(f"Sample 10 predictions artifacts model: {preds_art[:10]}")

if __name__ == "__main__":
    run_analysis()
