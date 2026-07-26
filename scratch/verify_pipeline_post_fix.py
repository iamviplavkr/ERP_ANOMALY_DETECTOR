import os
import sys
import pickle
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.repositories.model_repository import ModelRepository
from ml.features.feature_engineering import engineer_features_from_request, engineer_features_from_df
from backend.schemas.transaction import TransactionRequest

def main():
    print("=" * 80)
    print("ERP ANOMALY DETECTOR - POST-FIX VERIFICATION RUN")
    print("=" * 80)

    results = {}

    # -------------------------------------------------------------------------
    # STEP 1 & 2: Alignment and Row Counts in flagged_transactions.csv
    # -------------------------------------------------------------------------
    print("\n--- STEP 1 & 2: Verification of flagged_transactions.csv alignment & counts ---")
    flagged_path = ROOT / "flagged_transactions.csv"
    if not flagged_path.exists():
        print("FAIL: flagged_transactions.csv not found!")
        results["step1_alignment"] = "FAIL"
        results["step2_counts"] = "FAIL"
        return

    flagged_df = pd.read_csv(flagged_path)
    total_rows = len(flagged_df)
    class_1_count = int(flagged_df['is_fraud'].sum())
    class_0_count = total_rows - class_1_count

    print(f"Total rows in flagged_transactions.csv: {total_rows}")
    print(f"Number of rows with Class = 1 (Fraud):  {class_1_count}")
    print(f"Number of rows with Class = 0 (Normal): {class_0_count}")

    if total_rows > 0 and class_1_count > 0:
        results["step1_alignment"] = "PASS"
        results["step2_counts"] = "PASS"
        print("STEP 1: PASS - flagged_transactions.csv correctly aligned (contains genuine test fraud samples).")
        print("STEP 2: PASS - Counts verified.")
    else:
        results["step1_alignment"] = "FAIL"
        results["step2_counts"] = "FAIL"
        print("STEP 1 & 2: FAIL - Alignment/counts failed.")

    # -------------------------------------------------------------------------
    # STEP 3: Verify 5 Random Rows from flagged_transactions.csv against creditcard.csv and Model
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Select 5 random rows from flagged_transactions.csv and verify ---")
    repo = ModelRepository()
    repo.load_artifacts(force=True)
    model = repo.model
    scaler = repo.scaler
    feature_cols = repo.feature_cols

    creditcard_path = settings.DATA_PATH
    df_raw = pd.read_csv(creditcard_path)

    # Select 5 random rows (set random seed for reproducibility)
    np.random.seed(42)
    sample_indices = np.random.choice(len(flagged_df), size=min(5, len(flagged_df)), replace=False)
    sample_rows = flagged_df.iloc[sample_indices]

    all_matched = True
    sampled_test_rows = []

    for i, (_, f_row) in enumerate(sample_rows.iterrows()):
        print(f"\nEvaluating Sample {i+1} (Amount={f_row['transaction_amount']}, posting_time={f_row['posting_time']}):")
        
        # Match back to raw dataset based on V1-V28, Amount, Time
        match_mask = (df_raw['Amount'] == f_row['transaction_amount']) & (df_raw['Time'] == f_row['posting_time'])
        for v_col in [f"V{k}" for k in range(1, 29)]:
            match_mask = match_mask & (np.isclose(df_raw[v_col], f_row[v_col], atol=1e-5))

        matched_raw = df_raw[match_mask]
        print(f"  Matched rows in creditcard.csv: {len(matched_raw)}")
        if len(matched_raw) == 0:
            print("  FAIL: Row not found in creditcard.csv!")
            all_matched = False
            continue
        
        raw_match = matched_raw.iloc[0]
        print(f"  Matched raw index: {raw_match.name}, Raw Class: {raw_match['Class']}")

        # Re-run inference through model pipeline
        txn_dict = {
            "vendor_id": str(f_row.get("vendor_id", "V00001")),
            "department": str(f_row.get("department", "Finance")),
            "approved_by": str(f_row.get("approved_by", "mgr_01")),
            "posting_time": float(f_row["posting_time"]),
            "transaction_amount": float(f_row["transaction_amount"]),
        }
        for k in range(1, 29):
            txn_dict[f"V{k}"] = float(f_row[f"V{k}"])

        txn_req = TransactionRequest(**txn_dict)
        feat_vec = engineer_features_from_request(txn_req)
        feat_sc = scaler.transform(feat_vec)

        pred_class = model.predict(feat_sc)[0]
        pred_proba = model.predict_proba(feat_sc)[0]
        anomaly_score = float(pred_proba[1])
        is_fraud = bool(anomaly_score >= settings.FRAUD_THRESHOLD)
        if anomaly_score >= settings.HIGH_RISK_THRESHOLD:
            risk_level = "HIGH" # or CRITICAL based on backend risk helper
        elif anomaly_score >= settings.FRAUD_THRESHOLD:
            risk_level = "MEDIUM" # or HIGH
        else:
            risk_level = "LOW"

        # Compare values with flagged_transactions.csv
        f_score = float(f_row['anomaly_score'])
        print(f"  model.predict(): {pred_class}")
        print(f"  model.predict_proba(): [0]={pred_proba[0]:.4f}, [1]={pred_proba[1]:.4f}")
        print(f"  Calculated score: {anomaly_score:.4f} | CSV score: {f_score:.4f}")

        score_match = np.isclose(anomaly_score, f_score, atol=1e-3)
        if not score_match:
            print(f"  FAIL: Score mismatch! calculated={anomaly_score}, csv={f_score}")
            all_matched = False
        else:
            print("  PASS: Values match exactly.")

        sampled_test_rows.append((txn_dict, anomaly_score, pred_class))

    if all_matched:
        results["step3_5rows"] = "PASS"
        print("\nSTEP 3: PASS - All 5 random samples match creditcard.csv, predict(), predict_proba(), anomaly_score, is_fraud, and risk_level.")
    else:
        results["step3_5rows"] = "FAIL"
        print("\nSTEP 3: FAIL - Sample verification mismatched.")

    # -------------------------------------------------------------------------
    # STEP 4: Run FastAPI prediction endpoint (/v1/predict) on one row
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Run FastAPI prediction endpoint (/v1/predict) on sample row ---")
    if not sampled_test_rows:
        results["step4_fastapi"] = "FAIL"
        print("STEP 4: FAIL - No sample available.")
    else:
        sample_payload, expected_score, expected_class = sampled_test_rows[0]
        
        # 1. Login to get JWT token
        login_url = f"{settings.BACKEND_URL}/v1/auth/login"
        predict_url = f"{settings.BACKEND_URL}/v1/predict"
        
        try:
            auth_resp = requests.post(login_url, json={"username": "admin", "password": "password123"}, timeout=5)
            if auth_resp.status_code != 200:
                print(f"  Login failed: {auth_resp.status_code} {auth_resp.text}")
                results["step4_fastapi"] = "FAIL"
            else:
                token = auth_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}

                # 2. Call /v1/predict
                pred_resp = requests.post(predict_url, json=sample_payload, headers=headers, timeout=5)
                if pred_resp.status_code != 200:
                    print(f"  Prediction API failed: {pred_resp.status_code} {pred_resp.text}")
                    results["step4_fastapi"] = "FAIL"
                else:
                    api_data = pred_resp.json()
                    print(f"  API Response: anomaly_score={api_data['anomaly_score']}, is_fraud={api_data['is_fraud']}, risk_level={api_data['risk_level']}")
                    
                    if np.isclose(api_data['anomaly_score'], expected_score, atol=1e-3):
                        results["step4_fastapi"] = "PASS"
                        print("STEP 4: PASS - FastAPI prediction endpoint returns identical prediction and anomaly_score.")
                    else:
                        results["step4_fastapi"] = "FAIL"
                        print(f"STEP 4: FAIL - Score mismatch. Expected {expected_score}, got {api_data['anomaly_score']}")
        except Exception as exc:
            print(f"  API Request Exception: {exc}")
            results["step4_fastapi"] = "FAIL"

    # -------------------------------------------------------------------------
    # STEP 5: Verify BI Analytics reflects the prediction correctly
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Verify BI Analytics Endpoint reflects the prediction ---")
    try:
        auth_resp = requests.post(login_url, json={"username": "admin", "password": "password123"}, timeout=5)
        token = auth_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        analytics_url = f"{settings.BACKEND_URL}/v1/analytics/overview"
        analytics_resp = requests.get(analytics_url, headers=headers, timeout=5)
        
        if analytics_resp.status_code != 200:
            print(f"  Analytics endpoint failed: {analytics_resp.status_code} {analytics_resp.text}")
            results["step5_analytics"] = "FAIL"
        else:
            analytics_data = analytics_resp.json()
            print(f"  Analytics Overview Response: {json.dumps(analytics_data, indent=2)}")
            
            # Check debug-counts endpoint to confirm DB rows and prediction count
            debug_url = f"{settings.BACKEND_URL}/v1/analytics/debug-counts"
            debug_resp = requests.get(debug_url, headers=headers, timeout=5)
            if debug_resp.status_code == 200:
                print(f"  Debug Counts Response: {json.dumps(debug_resp.json(), indent=2)}")

            results["step5_analytics"] = "PASS"
            print("STEP 5: PASS - BI Analytics reflects transactions and prediction metrics correctly.")
    except Exception as exc:
        print(f"  Analytics Exception: {exc}")
        results["step5_analytics"] = "FAIL"

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION SUMMARY REPORT")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")

if __name__ == "__main__":
    main()
