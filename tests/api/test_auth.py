"""
tests/api/test_auth.py
─────────────────────────────────────────────────────────────────
Comprehensive tests for authentication and RBAC endpoints.
Covers login, logout, token refresh, /me profile, and role access control.
"""

import pytest
from starlette.testclient import TestClient
from backend.main import app
from tests.conftest import auth_header


client = TestClient(app)


# ── Login Tests ───────────────────────────────────────────────────────────────

class TestLogin:
    """Tests for POST /v1/auth/login"""

    def test_login_success_admin(self):
        response = client.post("/v1/auth/login", json={
            "username": "admin",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == "admin"
        assert data["role"] == "Admin"

    def test_login_success_finance(self):
        response = client.post("/v1/auth/login", json={
            "username": "finance",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "finance"
        assert data["role"] == "Finance User"

    def test_login_success_analyst(self):
        response = client.post("/v1/auth/login", json={
            "username": "analyst",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "analyst"
        assert data["role"] == "Fraud Analyst"

    def test_login_success_auditor(self):
        response = client.post("/v1/auth/login", json={
            "username": "auditor",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "auditor"
        assert data["role"] == "Auditor"

    def test_login_wrong_password(self):
        response = client.post("/v1/auth/login", json={
            "username": "admin",
            "password": "wrong_password"
        })
        assert response.status_code == 401
        assert "InvalidCredentials" in response.json()["error"]

    def test_login_nonexistent_user(self):
        response = client.post("/v1/auth/login", json={
            "username": "nonexistent_user",
            "password": "password123"
        })
        assert response.status_code == 401
        assert "InvalidCredentials" in response.json()["error"]

    def test_login_empty_username(self):
        response = client.post("/v1/auth/login", json={
            "username": "",
            "password": "password123"
        })
        # Empty username won't match any user
        assert response.status_code == 401

    def test_login_missing_fields(self):
        response = client.post("/v1/auth/login", json={
            "username": "admin"
        })
        assert response.status_code == 422  # Pydantic validation error


# ── Token Refresh Tests ───────────────────────────────────────────────────────

class TestTokenRefresh:
    """Tests for POST /v1/auth/refresh"""

    def test_refresh_token_success(self):
        # First login to get tokens
        login_resp = client.post("/v1/auth/login", json={
            "username": "admin",
            "password": "password123"
        })
        tokens = login_resp.json()

        # Use refresh token to get new access token
        response = client.post("/v1/auth/refresh", json={
            "refresh_token": tokens["refresh_token"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["username"] == "admin"
        assert data["role"] == "Admin"
        # Verify access token is a valid non-empty string
        assert isinstance(data["access_token"], str) and len(data["access_token"]) > 0

    def test_refresh_with_invalid_token(self):
        response = client.post("/v1/auth/refresh", json={
            "refresh_token": "invalid.jwt.token"
        })
        assert response.status_code == 401

    def test_refresh_with_access_token_fails(self):
        """Refresh endpoint should reject access tokens (type mismatch)."""
        login_resp = client.post("/v1/auth/login", json={
            "username": "admin",
            "password": "password123"
        })
        tokens = login_resp.json()

        response = client.post("/v1/auth/refresh", json={
            "refresh_token": tokens["access_token"]  # Wrong token type
        })
        assert response.status_code == 401


# ── Profile (GET /me) Tests ──────────────────────────────────────────────────

class TestProfile:
    """Tests for GET /v1/auth/me"""

    def test_get_profile_authenticated(self):
        login_resp = client.post("/v1/auth/login", json={
            "username": "finance",
            "password": "password123"
        })
        tokens = login_resp.json()

        response = client.get("/v1/auth/me", headers=auth_header(tokens))
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "finance"
        assert data["email"] == "finance@company.com"
        assert data["role"] == "Finance User"
        assert data["is_active"] is True

    def test_get_profile_no_token(self):
        response = client.get("/v1/auth/me")
        assert response.status_code == 401

    def test_get_profile_invalid_token(self):
        response = client.get("/v1/auth/me", headers={
            "Authorization": "Bearer invalid.jwt.token"
        })
        assert response.status_code == 401


# ── Logout Tests ─────────────────────────────────────────────────────────────

class TestLogout:
    """Tests for POST /v1/auth/logout"""

    def test_logout_success(self):
        login_resp = client.post("/v1/auth/login", json={
            "username": "admin",
            "password": "password123"
        })
        tokens = login_resp.json()

        response = client.post("/v1/auth/logout", headers=auth_header(tokens))
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out."
        assert data["username"] == "admin"

    def test_logout_without_token(self):
        response = client.post("/v1/auth/logout")
        assert response.status_code == 401

    def test_logout_with_invalid_token(self):
        response = client.post("/v1/auth/logout", headers={
            "Authorization": "Bearer invalid.jwt.token"
        })
        assert response.status_code == 401


# ── RBAC / Protected Route Tests ─────────────────────────────────────────────

class TestRBAC:
    """Tests for Role-Based Access Control on protected endpoints."""

    def _get_tokens(self, username):
        resp = client.post("/v1/auth/login", json={
            "username": username,
            "password": "password123"
        })
        return resp.json()

    def test_predict_allowed_for_admin(self):
        tokens = self._get_tokens("admin")
        payload = {
            "vendor_id": "V00123",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 3600.0,
            "transaction_amount": 250.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        response = client.post("/v1/predict", json=payload, headers=auth_header(tokens))
        assert response.status_code == 200

    def test_predict_allowed_for_finance(self):
        tokens = self._get_tokens("finance")
        payload = {
            "vendor_id": "V00123",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 3600.0,
            "transaction_amount": 250.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        response = client.post("/v1/predict", json=payload, headers=auth_header(tokens))
        assert response.status_code == 200

    def test_predict_allowed_for_analyst(self):
        tokens = self._get_tokens("analyst")
        payload = {
            "vendor_id": "V00123",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 3600.0,
            "transaction_amount": 250.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        response = client.post("/v1/predict", json=payload, headers=auth_header(tokens))
        assert response.status_code == 200

    def test_predict_denied_for_auditor(self):
        """Auditors should NOT be able to create predictions."""
        tokens = self._get_tokens("auditor")
        payload = {
            "vendor_id": "V00123",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 3600.0,
            "transaction_amount": 250.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        response = client.post("/v1/predict", json=payload, headers=auth_header(tokens))
        assert response.status_code == 403
        assert "Forbidden" in response.json()["error"]

    def test_predict_denied_without_token(self):
        payload = {
            "vendor_id": "V00123",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 3600.0,
            "transaction_amount": 250.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 401

    def test_stats_requires_auth(self):
        response = client.get("/v1/stats")
        assert response.status_code == 401

    def test_stats_allowed_for_admin_analyst_auditor(self):
        """Admin, Fraud Analyst, and Auditor should be allowed to view model stats."""
        for username in ["admin", "analyst", "auditor"]:
            tokens = self._get_tokens(username)
            response = client.get("/v1/stats", headers=auth_header(tokens))
            assert response.status_code == 200, f"Stats denied for {username}"

    def test_stats_denied_for_finance(self):
        """Finance User should NOT be allowed to view model stats."""
        tokens = self._get_tokens("finance")
        response = client.get("/v1/stats", headers=auth_header(tokens))
        assert response.status_code == 403
        assert "Forbidden" in response.json()["error"]

    def test_batch_predict_denied_for_auditor(self):
        """Auditors should NOT be able to run batch predictions."""
        tokens = self._get_tokens("auditor")
        payload = {
            "transactions": [{
                "vendor_id": "V00001",
                "department": "HR",
                "approved_by": "mgr_02",
                "posting_time": 1000.0,
                "transaction_amount": 50.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            }]
        }
        response = client.post("/v1/predict/batch", json=payload, headers=auth_header(tokens))
        assert response.status_code == 403


# ── Health Endpoint Stays Public ─────────────────────────────────────────────

class TestHealthPublic:
    """Verify health endpoints remain accessible without authentication."""

    def test_root_is_public(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_is_public(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
