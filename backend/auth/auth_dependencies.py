"""
backend/auth/auth_dependencies.py
─────────────────────────────────────────────────────────────────
FastAPI dependencies for route protection and RBAC guards.
"""

from typing import List, Dict, Any, Optional
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.core.exceptions import TokenInvalidError, InsufficientPermissionsError
from backend.services.auth_service import AuthService
from backend.repositories.user_repository import UserRepositoryInterface, InMemoryUserRepository

# Secure credentials extraction helper using standard HTTP Bearer schema
bearer_scheme = HTTPBearer(auto_error=False)


def get_user_repository() -> UserRepositoryInterface:
    """
    Provider dependency function for the user repository.
    """
    return InMemoryUserRepository()


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
