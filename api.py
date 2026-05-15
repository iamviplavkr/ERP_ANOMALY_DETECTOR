import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ERP Anomaly Detector API",
    description="Detects fraudulent transactions in ERP data using Random Forest + SHAP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model Artifacts ──────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# ── Request & Response Schemas ────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    # ERP context fields (for display/logging)
    vendor_id:          Optional[str]   = Field(default="V00001", example="V00123")
    department:         Optional[str]   = Field(default="Finance", example="Finance")
    approved_by:        Optional[str]   = Field(default="mgr_01",  example="mgr_01")

    # Core transaction fields
    posting_time:       float = Field(..., example=3600,   description="Seconds since first transaction")
    transaction_amount: float = Field(..., example=250.00, description="Transaction amount in currency")

    # PCA features V1–V28 (from dataset)
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

class TransactionResponse(BaseModel):
    vendor_id:        str
    department:       str
    approved_by:      str
    transaction_amount: float
    anomaly_score:    float
    is_fraud:         bool
    risk_level:       str
    alert_message:    str
    top_risk_factors: list

class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]

class HealthResponse(BaseModel):
    status: str
    model:  str
    features: int

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(txn: TransactionRequest) -> np.ndarray:
    log_amount    = np.log1p(txn.transaction_amount)
    hour_of_day   = int(txn.posting_time % 86400) // 3600
    is_night      = int(hour_of_day < 6 or hour_of_day > 22)
    amount_zscore = (txn.transaction_amount - 88.35) / 250.12  # dataset mean/std

    v_features = [
        txn.V1,  txn.V2,  txn.V3,  txn.V4,
        txn.V5,  txn.V6,  txn.V7,  txn.V8,
        txn.V9,  txn.V10, txn.V11, txn.V12,
        txn.V13, txn.V14, txn.V15, txn.V16,
        txn.V17, txn.V18, txn.V19, txn.V20,
        txn.V21, txn.V22, txn.V23, txn.V24,
        txn.V25, txn.V26, txn.V27, txn.V28,
        log_amount, hour_of_day, is_night, amount_zscore
    ]
    return np.array(v_features).reshape(1, -1)

def get_risk_level(score: float) -> tuple[str, str]:
    if score >= 0.8:
        return "HIGH",   "⚠️  High-risk transaction flagged. Immediate review required."
    elif score >= 0.5:
        return "MEDIUM", "🔶 Moderate risk detected. Manual verification recommended."
    else:
        return "LOW",    "✅ Transaction appears normal. No action required."

def get_top_risk_factors(features: np.ndarray, score: float) -> list:
    scaled = scaler.transform(features)[0]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    factors = []
    for i in top_idx:
        factors.append({
            "feature":    feature_cols[i],
            "importance": round(float(importances[i]), 4),
            "value":      round(float(features[0][i]), 4)
        })
    return factors

def predict_transaction(txn: TransactionRequest) -> TransactionResponse:
    features   = engineer_features(txn)
    features_sc = scaler.transform(features)
    proba      = model.predict_proba(features_sc)[0][1]
    is_fraud   = bool(proba >= 0.5)
    risk_level, alert_message = get_risk_level(proba)
    top_factors = get_top_risk_factors(features, proba)

    return TransactionResponse(
        vendor_id          = txn.vendor_id,
        department         = txn.department,
        approved_by        = txn.approved_by,
        transaction_amount = txn.transaction_amount,
        anomaly_score      = round(float(proba), 4),
        is_fraud           = is_fraud,
        risk_level         = risk_level,
        alert_message      = alert_message,
        top_risk_factors   = top_factors
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {"message": "ERP Anomaly Detector API is running. Visit /docs for Swagger UI."}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status   = "ok",
        model    = "RandomForestClassifier",
        features = len(feature_cols)
    )

@app.post("/predict", response_model=TransactionResponse, tags=["Prediction"])
def predict(txn: TransactionRequest):
    """
    Analyze a single ERP transaction and return anomaly score + risk level.
    - anomaly_score: 0.0 (normal) to 1.0 (fraud)
    - risk_level: LOW / MEDIUM / HIGH
    - top_risk_factors: SHAP-style feature importances
    """
    try:
        return predict_transaction(txn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    """
    Analyze multiple ERP transactions in one call.
    Returns list of predictions + summary stats.
    """
    try:
        results = [predict_transaction(txn) for txn in batch.transactions]
        flagged = [r for r in results if r.is_fraud]
        return {
            "total":        len(results),
            "flagged":      len(flagged),
            "flag_rate":    f"{len(flagged)/len(results)*100:.1f}%",
            "predictions":  results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Info"])
def model_stats():
    """Returns model metadata and feature list."""
    return {
        "model":         "RandomForestClassifier",
        "n_estimators":  model.n_estimators,
        "n_features":    len(feature_cols),
        "feature_names": feature_cols,
        "threshold":     0.5,
        "trained_on":    "Credit Card Fraud Dataset (284,807 transactions)"
    }

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)