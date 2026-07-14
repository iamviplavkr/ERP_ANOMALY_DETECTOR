"""
backend/api/v1/stats.py
─────────────────────────────────────────────────────────────────
Model telemetry and description endpoints.
Protected by JWT authentication — any authenticated user may access.
"""

from fastapi import APIRouter, Depends
from backend.auth.auth_dependencies import RoleChecker, get_current_user
from backend.repositories.model_repository import ModelRepository

router = APIRouter()
repo = ModelRepository()

# RBAC Guard for stats (Finance User is restricted from viewing stats)
stats_rbac = RoleChecker(["Admin", "Fraud Analyst", "Auditor"])


@router.get(
    "/stats",
    tags=["Info"],
    dependencies=[Depends(stats_rbac)]
)
def model_stats():
    """
    Returns model metadata and feature list.
    Requires a valid JWT access token.
    """
    repo.load_artifacts()
    return {
        "model":         "RandomForestClassifier",
        "n_estimators":  repo.model.n_estimators if hasattr(repo.model, "n_estimators") else 100,
        "n_features":    len(repo.feature_cols),
        "feature_names": repo.feature_cols,
        "threshold":     0.5,
        "trained_on":    "Credit Card Fraud Dataset (284,807 transactions)"
    }
