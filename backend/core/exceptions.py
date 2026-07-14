"""
backend/core/exceptions.py
─────────────────────────────────────────────────────────────────
Custom exception hierarchy for the ERP Anomaly Detector.

Raising typed exceptions (rather than bare `Exception`) lets the
global error handlers in `backend/middleware/error_handler.py`
return precise HTTP status codes and structured error bodies.
"""


class ERPBaseException(Exception):
    """Root exception — never raised directly."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


# ── Model / Artifact Errors ───────────────────────────────────────────────────

class ArtifactNotFoundError(ERPBaseException):
    """Raised when a required model artifact (.pkl file) cannot be found."""


class ModelNotLoadedError(ERPBaseException):
    """Raised when inference is attempted before the model is initialised."""


class ModelLoadError(ERPBaseException):
    """Raised when a model artifact exists but cannot be deserialised."""


# ── Feature Engineering Errors ────────────────────────────────────────────────

class FeatureEngineeringError(ERPBaseException):
    """Raised when feature construction fails (bad input shape, NaN, etc.)."""


class InvalidInputError(ERPBaseException):
    """Raised when the request payload contains invalid or out-of-range values."""


# ── Prediction Errors ─────────────────────────────────────────────────────────

class PredictionError(ERPBaseException):
    """Raised when the model raises an error during inference."""


class BatchPredictionError(ERPBaseException):
    """Raised when one or more transactions in a batch fail."""


# ── Configuration Errors ──────────────────────────────────────────────────────

class ConfigurationError(ERPBaseException):
    """Raised when a required configuration value is missing or invalid."""


# ── Authentication & Authorization Errors ─────────────────────────────────────

class AuthError(ERPBaseException):
    """Base exception for authentication failures."""


class InvalidCredentialsError(AuthError):
    """Raised when user login validation fails."""


class TokenExpiredError(AuthError):
    """Raised when a JWT access or refresh token has expired."""


class TokenInvalidError(AuthError):
    """Raised when a JWT token signature or payload decoding fails."""


class InsufficientPermissionsError(ERPBaseException):
    """Raised when the user has valid authentication but fails RBAC check."""


# ── Database Errors ───────────────────────────────────────────────────────────

class DatabaseError(ERPBaseException):
    """Raised when a database operation fails (connection, query, migration)."""

