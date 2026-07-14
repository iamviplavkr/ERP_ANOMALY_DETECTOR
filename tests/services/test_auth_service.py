"""
tests/services/test_auth_service.py
─────────────────────────────────────────────────────────────────
Unit tests for the AuthService layer.
Verifies password hashing, JWT creation/decoding, user authentication,
and token refresh — all through InMemoryUserRepository.
"""

import pytest
from backend.services.auth_service import AuthService
from backend.repositories.user_repository import InMemoryUserRepository
from backend.core.exceptions import (
    InvalidCredentialsError,
    TokenExpiredError,
    TokenInvalidError,
)


@pytest.fixture
def auth_service():
    """Creates an AuthService with the singleton InMemoryUserRepository."""
    return AuthService(user_repo=InMemoryUserRepository())


class TestPasswordHashing:
    """Tests for bcrypt password hashing and verification."""

    def test_hash_and_verify(self, auth_service):
        plain = "my_secure_password"
        hashed = auth_service.hash_password(plain)
        assert auth_service.verify_password(plain, hashed)

    def test_verify_wrong_password(self, auth_service):
        hashed = auth_service.hash_password("correct_password")
        assert not auth_service.verify_password("wrong_password", hashed)

    def test_hash_is_unique_per_call(self, auth_service):
        """Different salts produce different hashes for the same password."""
        h1 = auth_service.hash_password("same_password")
        h2 = auth_service.hash_password("same_password")
        assert h1 != h2  # bcrypt uses random salt


class TestJWTTokens:
    """Tests for JWT token creation and decoding."""

    def test_create_and_decode_access_token(self, auth_service):
        token = auth_service.create_access_token("admin", "Admin")
        payload = auth_service.decode_token(token, is_refresh=False)
        assert payload["sub"] == "admin"
        assert payload["role"] == "Admin"
        assert payload["type"] == "access"

    def test_create_and_decode_refresh_token(self, auth_service):
        token = auth_service.create_refresh_token("finance")
        payload = auth_service.decode_token(token, is_refresh=True)
        assert payload["sub"] == "finance"
        assert payload["type"] == "refresh"

    def test_access_token_rejected_as_refresh(self, auth_service):
        """Access token should fail if used as a refresh token."""
        access = auth_service.create_access_token("admin", "Admin")
        with pytest.raises((TokenInvalidError,)):
            auth_service.decode_token(access, is_refresh=True)

    def test_refresh_token_rejected_as_access(self, auth_service):
        """Refresh token should fail if used as an access token."""
        refresh = auth_service.create_refresh_token("admin")
        with pytest.raises((TokenInvalidError,)):
            auth_service.decode_token(refresh, is_refresh=False)

    def test_invalid_token_raises_error(self, auth_service):
        with pytest.raises(TokenInvalidError):
            auth_service.decode_token("invalid.jwt.garbage", is_refresh=False)


class TestUserAuthentication:
    """Tests for authenticate_user — all via InMemoryUserRepository."""

    def test_authenticate_success(self, auth_service):
        user = auth_service.authenticate_user("admin", "password123")
        assert user["username"] == "admin"
        assert user["role"] == "Admin"
        assert user["is_active"] is True

    def test_authenticate_all_roles(self, auth_service):
        """Verify all pre-seeded users can authenticate."""
        role_map = {
            "admin": "Admin",
            "finance": "Finance User",
            "analyst": "Fraud Analyst",
            "auditor": "Auditor",
        }
        for username, expected_role in role_map.items():
            user = auth_service.authenticate_user(username, "password123")
            assert user["role"] == expected_role

    def test_authenticate_wrong_password(self, auth_service):
        with pytest.raises(InvalidCredentialsError):
            auth_service.authenticate_user("admin", "wrong_password")

    def test_authenticate_nonexistent_user(self, auth_service):
        with pytest.raises(InvalidCredentialsError):
            auth_service.authenticate_user("ghost_user", "password123")


class TestTokenRefresh:
    """Tests for refresh_access_token — issues new access token."""

    def test_refresh_success(self, auth_service):
        refresh = auth_service.create_refresh_token("admin")
        result = auth_service.refresh_access_token(refresh)
        assert "access_token" in result
        assert result["role"] == "Admin"

    def test_refresh_with_invalid_token(self, auth_service):
        with pytest.raises(TokenInvalidError):
            auth_service.refresh_access_token("invalid.token.here")
