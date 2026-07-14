"""
backend/api/v1/health.py
─────────────────────────────────────────────────────────────────
Health check endpoints for service checking and monitoring.
"""

from fastapi import APIRouter
from backend.schemas.transaction import HealthResponse
from backend.repositories.model_repository import ModelRepository

router = APIRouter()
repo = ModelRepository()


@router.get("/", tags=["Root"])
def root():
    """
    Root endpoint for uptime validation.
    """
    return {"message": "ERP Anomaly Detector API is running. Visit /docs for Swagger UI."}


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Performs dependencies readiness check (model loaded check).
    """
    # Ensure artifacts are loaded
    repo.load_artifacts()

    return HealthResponse(
        status="ok",
        model="RandomForestClassifier",
        features=len(repo.feature_cols)
    )
