"""
backend/api/v1/predict.py
─────────────────────────────────────────────────────────────────
Anomaly prediction endpoints supporting single and batch predictions.
Supports API-key authentication validation if configured.
"""

from fastapi import APIRouter, Depends, Request

from backend.auth.auth_dependencies import (
    RoleChecker,
    get_transaction_repository,
    get_prediction_repository,
    get_audit_log_repository,
    get_alert_repository,
    get_vendor_repository,
)
from backend.repositories.transaction_repository import (
    TransactionRepositoryInterface,
    PostgresTransactionRepository,
)
from backend.repositories.prediction_repository import PredictionRepositoryInterface
from backend.repositories.audit_log_repository import AuditLogRepositoryInterface
from backend.repositories.alert_repository import AlertRepositoryInterface
from backend.repositories.vendor_repository import VendorRepositoryInterface
from backend.schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
    BatchRequest,
    BatchResponseSummary,
)
from backend.services.prediction_service import PredictionService
from backend.services.analytics_service import AnalyticsService

router = APIRouter()
prediction_service = PredictionService()


def _get_db_from_repo(repo: TransactionRepositoryInterface):
    """Extract the SQLAlchemy session from a Postgres repo, or None for InMemory.

    FastAPI's DI creates one Session per Depends(get_db) call.  All repositories
    receive the SAME Session object because get_transaction_repository (and the
    others) all depend on get_db — FastAPI deduplicates identical dependency
    instances within a single request.  We therefore pull the session from the
    transaction_repo and pass it as svc.db so that commit()/rollback() targets
    the exact session that staged the ORM objects.
    """
    if isinstance(repo, PostgresTransactionRepository):
        return repo.db
    return None

# RBAC Guard for predictions (Auditors are restricted from creating predictions)
prediction_rbac = RoleChecker(["Admin", "Finance User", "Fraud Analyst"])


@router.post(
    "/predict",
    response_model=TransactionResponse,
    tags=["Prediction"],
)
def predict(
    txn: TransactionRequest,
    request: Request,
    current_user: dict = Depends(prediction_rbac),
    transaction_repo: TransactionRepositoryInterface = Depends(get_transaction_repository),
    prediction_repo: PredictionRepositoryInterface = Depends(get_prediction_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    alert_repo: AlertRepositoryInterface = Depends(get_alert_repository),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
):
    """
    Analyze a single ERP transaction and return anomaly score + risk level.
    - anomaly_score: 0.0 (normal) to 1.0 (fraud)
    - risk_level: LOW / MEDIUM / HIGH
    - top_risk_factors: SHAP-style feature importances
    """
    service = PredictionService(
        transaction_repo=transaction_repo,
        prediction_repo=prediction_repo,
        audit_log_repo=audit_log_repo,
        alert_repo=alert_repo,
        vendor_repo=vendor_repo,
        db=_get_db_from_repo(transaction_repo),
    )
    result = service.predict_single(
        txn,
        user_id=current_user.get("id"),
        ip_address=request.client.host if request.client else None,
    )
    # Bust analytics cache so BI dashboard reflects the new record immediately
    AnalyticsService.invalidate_cache()
    return result


@router.post(
    "/predict/batch",
    response_model=BatchResponseSummary,
    tags=["Prediction"],
)
def predict_batch(
    batch: BatchRequest,
    request: Request,
    current_user: dict = Depends(prediction_rbac),
    transaction_repo: TransactionRepositoryInterface = Depends(get_transaction_repository),
    prediction_repo: PredictionRepositoryInterface = Depends(get_prediction_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    alert_repo: AlertRepositoryInterface = Depends(get_alert_repository),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
):
    """
    Analyze multiple ERP transactions in one call.
    Returns list of predictions + summary stats.
    """
    service = PredictionService(
        transaction_repo=transaction_repo,
        prediction_repo=prediction_repo,
        audit_log_repo=audit_log_repo,
        alert_repo=alert_repo,
        vendor_repo=vendor_repo,
        db=_get_db_from_repo(transaction_repo),
    )
    res = service.predict_batch(
        batch.transactions,
        user_id=current_user.get("id"),
        ip_address=request.client.host if request.client else None,
    )
    # Bust analytics cache so BI dashboard reflects all new records immediately
    AnalyticsService.invalidate_cache()
    return BatchResponseSummary(
        total=res["total"],
        flagged=res["flagged"],
        flag_rate=res["flag_rate"],
        predictions=res["predictions"]
    )
