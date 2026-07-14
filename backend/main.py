"""
backend/main.py
─────────────────────────────────────────────────────────────────
Main FastAPI application entry point.

Loads settings, configures logging, CORS, global exception handlers,
logging middleware, and launches Uvicorn server if run directly.

Schema management is handled exclusively by Alembic — this file
never calls ``Base.metadata.create_all()``.
"""

from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager replacing the deprecated @app.on_event("startup").
    Validates environment configurations and pre-loads model artifacts on startup.
    Seeds default database users for non-testing environments.
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

    # Seed default development users (only for non-testing environments).
    # Alembic must have been run first to create the schema — this does NOT
    # call Base.metadata.create_all().
    if settings.ENVIRONMENT != "testing":
        try:
            from backend.database.seed import seed_default_users
            from backend.database.session import get_engine
            seed_default_users(get_engine())
            logger.info("Database seeding completed successfully.")
        except Exception as exc:
            logger.error(
                f"Database seeding failed: {str(exc)}. "
                "Ensure PostgreSQL is running and Alembic migrations have been applied."
            )

    yield  # Application runs here

    # Shutdown: dispose the database connection pool
    if settings.ENVIRONMENT != "testing":
        try:
            from backend.database.session import dispose_engine
            dispose_engine()
        except Exception:
            pass


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Detects anomalous and fraudulent transactions in ERP financial data using Scikit-Learn Random Forest + SHAP.",
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
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


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
