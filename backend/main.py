"""
backend/main.py
─────────────────────────────────────────────────────────────────
Main FastAPI application entry point.

Loads settings, configures logging, CORS, global exception handlers,
logging middleware, and launches Uvicorn server if run directly.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import api_router
from backend.core.config import settings
from backend.core.logging import get_logger, setup_logging
from backend.middleware.error_handler import register_exception_handlers
from backend.middleware.logging_middleware import RequestLoggingMiddleware
from backend.repositories.model_repository import ModelRepository

# Initialize logging before any loggers are instantiated
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Detects anomalous and fraudulent transactions in ERP financial data using Scikit-Learn Random Forest + SHAP.",
    version=settings.VERSION,
    debug=settings.DEBUG,
)

# Register CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Request Logging Middleware
app.add_middleware(RequestLoggingMiddleware)

# Register Global Exception Handlers
register_exception_handlers(app)

# Include Centralized APIRouter
app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    """
    Triggered when FastAPI server starts.
    Validates environment configurations and pre-loads model artifacts.
    """
    logger.info("Starting up ERP Anomaly Detector API server...")
    logger.info(f"Environment: {settings.ENVIRONMENT} | Debug mode: {settings.DEBUG}")

    # Pre-load model artifacts to speed up first inference and fail fast if files are missing
    repo = ModelRepository()
    try:
        repo.load_artifacts()
        logger.info("Ready to accept prediction requests.")
    except Exception as exc:
        logger.error(
            f"Failed to pre-load model artifacts during startup: {str(exc)}. "
            "Server will still start but predict requests will fail until artifacts are restored."
        )


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
