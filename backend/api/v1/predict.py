"""
backend/api/v1/predict.py
─────────────────────────────────────────────────────────────────
Anomaly prediction endpoints supporting single and batch predictions.
Supports API-key authentication validation if configured.
"""

from fastapi import APIRouter, Depends
from backend.auth.auth_dependencies import RoleChecker
from backend.schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
    BatchRequest,
    BatchResponseSummary,
)
from backend.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

# RBAC Guard for predictions (Auditors are restricted from creating predictions)
prediction_rbac = RoleChecker(["Admin", "Finance User", "Fraud Analyst"])


@router.post(
    "/predict",
    response_model=TransactionResponse,
    tags=["Prediction"],
    dependencies=[Depends(prediction_rbac)]
)
def predict(txn: TransactionRequest):
    """
    Analyze a single ERP transaction and return anomaly score + risk level.
    - anomaly_score: 0.0 (normal) to 1.0 (fraud)
    - risk_level: LOW / MEDIUM / HIGH
    - top_risk_factors: SHAP-style feature importances
    """
    return prediction_service.predict_single(txn)


@router.post(
    "/predict/batch",
    response_model=BatchResponseSummary,
    tags=["Prediction"],
    dependencies=[Depends(prediction_rbac)]
)
def predict_batch(batch: BatchRequest):
    """
    Analyze multiple ERP transactions in one call.
    Returns list of predictions + summary stats.
    """
    res = prediction_service.predict_batch(batch.transactions)
    return BatchResponseSummary(
        total=res["total"],
        flagged=res["flagged"],
        flag_rate=res["flag_rate"],
        predictions=res["predictions"]
    )
