"""
backend/api/router.py
─────────────────────────────────────────────────────────────────
Root API router that registers all component sub-routers.
"""

from fastapi import APIRouter
from backend.api.v1.health import router as health_router
from backend.api.v1.predict import router as predict_router
from backend.api.v1.stats import router as stats_router
from backend.api.v1.auth import router as auth_router
from backend.api.v1.alerts import router as alerts_router
from backend.api.v1.vendors import router as vendors_router
from backend.api.v1.analytics import router as analytics_router

api_router = APIRouter()

# Register sub-routers under version prefixes
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/v1", tags=["Authentication"])
api_router.include_router(predict_router, prefix="/v1", tags=["Prediction"])
api_router.include_router(stats_router, prefix="/v1", tags=["Info"])
api_router.include_router(alerts_router, prefix="/v1", tags=["Alerts"])
api_router.include_router(vendors_router, prefix="/v1", tags=["Vendors"])
api_router.include_router(analytics_router, prefix="/v1", tags=["Analytics"])


