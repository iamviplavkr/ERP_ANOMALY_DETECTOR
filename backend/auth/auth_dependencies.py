"""
backend/auth/auth_dependencies.py
─────────────────────────────────────────────────────────────────
FastAPI dependencies for route protection and RBAC guards.

Repository selection:
  • ENVIRONMENT=testing  →  InMemoryUserRepository (no DB needed)
  • Any other value      →  PostgresUserRepository (requires DB)
  Never falls back silently — a missing DATABASE_URL in production
  raises ConfigurationError.
"""

from typing import List, Dict, Any, Optional
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.core.config import settings
from backend.core.exceptions import TokenInvalidError, InsufficientPermissionsError
from backend.database.session import get_db
from backend.repositories.postgres_user_repository import PostgresUserRepository
from backend.repositories.user_repository import UserRepositoryInterface, InMemoryUserRepository
from backend.repositories.transaction_repository import (
    TransactionRepositoryInterface,
    PostgresTransactionRepository,
    InMemoryTransactionRepository,
)
from backend.repositories.prediction_repository import (
    PredictionRepositoryInterface,
    PostgresPredictionRepository,
    InMemoryPredictionRepository,
)
from backend.repositories.audit_log_repository import (
    AuditLogRepositoryInterface,
    PostgresAuditLogRepository,
    InMemoryAuditLogRepository,
)
from backend.repositories.alert_repository import (
    AlertRepositoryInterface,
    PostgresAlertRepository,
    InMemoryAlertRepository,
)
from backend.repositories.vendor_repository import (
    VendorRepositoryInterface,
    PostgresVendorRepository,
    InMemoryVendorRepository,
)
from backend.repositories.analytics_repository import (
    AnalyticsRepositoryInterface,
    PostgresAnalyticsRepository,
    InMemoryAnalyticsRepository,
)
from backend.services.auth_service import AuthService

# Secure credentials extraction helper using standard HTTP Bearer schema
bearer_scheme = HTTPBearer(auto_error=False)


def get_transaction_repository(db=Depends(get_db)) -> TransactionRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryTransactionRepository()
    return PostgresTransactionRepository(db)


def get_prediction_repository(db=Depends(get_db)) -> PredictionRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryPredictionRepository()
    return PostgresPredictionRepository(db)


def get_audit_log_repository(db=Depends(get_db)) -> AuditLogRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryAuditLogRepository()
    return PostgresAuditLogRepository(db)


def get_alert_repository(db=Depends(get_db)) -> AlertRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryAlertRepository()
    return PostgresAlertRepository(db)


def get_vendor_repository(db=Depends(get_db)) -> VendorRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryVendorRepository()
    return PostgresVendorRepository(db)


def get_analytics_repository(
    db=Depends(get_db),
    transaction_repo=Depends(get_transaction_repository),
    prediction_repo=Depends(get_prediction_repository),
    alert_repo=Depends(get_alert_repository),
    vendor_repo=Depends(get_vendor_repository),
) -> AnalyticsRepositoryInterface:
    if settings.ENVIRONMENT == "testing":
        return InMemoryAnalyticsRepository(
            transaction_repo=transaction_repo,
            prediction_repo=prediction_repo,
            alert_repo=alert_repo,
            vendor_repo=vendor_repo,
        )
    return PostgresAnalyticsRepository(db)


def get_user_repository(db=Depends(get_db)) -> UserRepositoryInterface:
    """
    Provider dependency function for the user repository.

    Uses InMemoryUserRepository **only** when ENVIRONMENT=testing.
    All other environments receive a PostgresUserRepository backed
    by the real database session — no silent fallback.
    """
    if settings.ENVIRONMENT == "testing":
        return InMemoryUserRepository()
    return PostgresUserRepository(db)


def get_auth_service(
    user_repo: UserRepositoryInterface = Depends(get_user_repository)
) -> AuthService:
    """
    Provider dependency function for the AuthService.
    """
    return AuthService(user_repo=user_repo)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """
    Dependency checking JWT token authenticity in the HTTP headers.
    Returns the user data dict if validation succeeds.
    """
    if not credentials:
        raise TokenInvalidError("Missing authentication token header.")

    # Decode and validate token integrity
    payload = auth_service.decode_token(credentials.credentials, is_refresh=False)
    username = payload.get("sub")
    if not username:
        raise TokenInvalidError("Invalid token payload credentials.")

    # Fetch user data from repository
    user = auth_service.user_repo.get_by_username(username)
    if not user:
        raise TokenInvalidError("User account not found.")

    if not user.get("is_active", True):
        raise TokenInvalidError("User account is deactivated.")

    return user


class RoleChecker:
    """
    FastAPI dependency factory class enforcing Role-Based Access Control.
    """

    def __init__(self, allowed_roles: List[str]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """
        Validates if the user's role is permitted to execute this action.
        """
        user_role = current_user.get("role")
        if user_role not in self.allowed_roles:
            raise InsufficientPermissionsError(
                f"Role '{user_role}' is not authorized to access this resource."
            )
        return current_user
