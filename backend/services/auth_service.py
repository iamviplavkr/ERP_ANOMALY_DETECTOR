"""
backend/services/auth_service.py
─────────────────────────────────────────────────────────────────
Authentication services handling security computations, password verification,
and JWT token operations.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import jwt
import bcrypt

from backend.core.config import settings
from backend.core.exceptions import (
    InvalidCredentialsError,
    TokenExpiredError,
    TokenInvalidError,
)
from backend.core.logging import get_logger
from backend.repositories.user_repository import UserRepositoryInterface, InMemoryUserRepository

logger = get_logger(__name__)


class AuthService:
    """
    Business service layer managing user logins and cryptographically signed JWT tokens.
    """

    def __init__(self, user_repo: UserRepositoryInterface = None) -> None:
        self.user_repo = user_repo or InMemoryUserRepository()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies if the plaintext password matches the stored bcrypt hash."""
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"),
                hashed_password.encode("utf-8")
            )
        except Exception as exc:
            logger.error(f"Error checking bcrypt password: {exc}")
            return False

    def hash_password(self, password: str) -> str:
        """Hashes a plaintext password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def create_access_token(self, username: str, role: str) -> str:
        """
        Creates a signed JWT access token.
        """
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": username,
            "role": role,
            "exp": expire,
            "type": "access"
        }
        encoded_jwt = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt

    def create_refresh_token(self, username: str) -> str:
        """
        Creates a signed JWT refresh token.
        """
        expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        payload = {
            "sub": username,
            "exp": expire,
            "type": "refresh"
        }
        encoded_jwt = jwt.encode(
            payload,
            settings.JWT_REFRESH_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt

    def decode_token(self, token: str, is_refresh: bool = False) -> Dict[str, Any]:
        """
        Decodes and validates a JWT token.
        Raises TokenExpiredError or TokenInvalidError on failures.
        """
        secret = settings.JWT_REFRESH_SECRET_KEY if is_refresh else settings.JWT_SECRET_KEY
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Check type safety
            expected_type = "refresh" if is_refresh else "access"
            if payload.get("type") != expected_type:
                raise TokenInvalidError(f"Invalid token type. Expected {expected_type} token.")
                
            return payload

        except jwt.ExpiredSignatureError as exc:
            logger.warning("JWT token signature validation expired.")
            raise TokenExpiredError("Token has expired. Please log in again.") from exc
        except jwt.InvalidTokenError as exc:
            logger.warning(f"JWT token decoding failed: {exc}")
            raise TokenInvalidError("Invalid token authentication credentials.") from exc

    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Validates credentials and returns user information.
        Raises InvalidCredentialsError if authentication fails.
        """
        user = self.user_repo.get_by_username(username)
        if not user:
            logger.warning(f"Failed login attempt: Username '{username}' not found.")
            raise InvalidCredentialsError("Invalid username or password.")

        if not user.get("is_active", True):
            logger.warning(f"Failed login attempt: Account '{username}' is disabled.")
            raise InvalidCredentialsError("This account is currently deactivated.")

        if not self.verify_password(password, user["password_hash"]):
            logger.warning(f"Failed login attempt: Invalid password for username '{username}'.")
            raise InvalidCredentialsError("Invalid username or password.")

        logger.info(f"User authenticated successfully: '{username}' with role '{user['role']}'.")
        return user

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Validates refresh token and issues a new access token.
        """
        payload = self.decode_token(refresh_token, is_refresh=True)
        username = payload.get("sub")
        
        user = self.user_repo.get_by_username(username)
        if not user or not user.get("is_active", True):
            raise InvalidCredentialsError("User account deactivated or missing.")

        new_access_token = self.create_access_token(user["username"], user["role"])
        return {
            "access_token": new_access_token,
            "role": user["role"]
        }
