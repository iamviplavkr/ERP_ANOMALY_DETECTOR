"""
backend/middleware/error_handler.py
─────────────────────────────────────────────────────────────────
Global exception handler registration for FastAPI app.
Maps custom exception classes to clean, standardized JSON HTTP responses.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from backend.core.exceptions import (
    ERPBaseException,
    ArtifactNotFoundError,
    ModelNotLoadedError,
    ModelLoadError,
    FeatureEngineeringError,
    InvalidInputError,
    PredictionError,
    BatchPredictionError,
    ConfigurationError,
    AuthError,
    InvalidCredentialsError,
    TokenExpiredError,
    TokenInvalidError,
    InsufficientPermissionsError,
    DatabaseError,
)
from backend.core.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers exception handlers on the FastAPI app.
    """

    @app.exception_handler(ArtifactNotFoundError)
    async def artifact_not_found_handler(request: Request, exc: ArtifactNotFoundError):
        logger.error(f"Artifact Not Found: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "ModelServiceUnavailable", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        logger.error(f"Model Not Loaded: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "ModelNotLoaded", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError):
        logger.error(f"Model Load Failed: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "ModelLoadError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(FeatureEngineeringError)
    async def feature_engineering_handler(request: Request, exc: FeatureEngineeringError):
        logger.error(f"Feature Engineering Failed: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "FeatureEngineeringError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        logger.error(f"Invalid Input: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "InvalidInput", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(PredictionError)
    async def prediction_handler(request: Request, exc: PredictionError):
        logger.error(f"Prediction Error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "PredictionError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(BatchPredictionError)
    async def batch_prediction_handler(request: Request, exc: BatchPredictionError):
        logger.error(f"Batch Prediction Error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "BatchPredictionError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError):
        logger.error(f"Configuration Error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "ConfigurationError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(InvalidCredentialsError)
    async def invalid_credentials_handler(request: Request, exc: InvalidCredentialsError):
        logger.warning(f"Failed login attempt: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "InvalidCredentials", "message": exc.message}
        )

    @app.exception_handler(TokenExpiredError)
    async def token_expired_handler(request: Request, exc: TokenExpiredError):
        logger.warning(f"Expired JWT token: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "TokenExpired", "message": exc.message}
        )

    @app.exception_handler(TokenInvalidError)
    async def token_invalid_handler(request: Request, exc: TokenInvalidError):
        logger.warning(f"Invalid JWT token: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "TokenInvalid", "message": exc.message}
        )

    @app.exception_handler(InsufficientPermissionsError)
    async def insufficient_permissions_handler(request: Request, exc: InsufficientPermissionsError):
        logger.warning(f"Unauthorized RBAC access attempt: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Forbidden", "message": exc.message}
        )

    @app.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError):
        logger.error(f"Database Error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "DatabaseError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(ERPBaseException)
    async def generic_erp_handler(request: Request, exc: ERPBaseException):
        logger.error(f"ERP Application Error: {exc.message}", extra={"details": exc.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "ERPApplicationError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(Exception)
    async def global_fallback_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled system exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": "An unexpected error occurred. Please contact the administrator."}
        )
