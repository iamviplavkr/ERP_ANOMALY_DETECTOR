"""
tests/api/test_stats.py
─────────────────────────────────────────────────────────────────
Tests the statistics and telemetry metadata endpoints.
Now requires JWT authentication.
"""

from fastapi.testclient import TestClient
from backend.main import app
from tests.conftest import auth_header

client = TestClient(app)


def _get_admin_tokens():
    """Helper to authenticate as admin for stats tests."""
    resp = client.post("/v1/auth/login", json={
        "username": "admin",
        "password": "password123"
    })
    return resp.json()


def test_stats_endpoint():
    tokens = _get_admin_tokens()
    response = client.get("/v1/stats", headers=auth_header(tokens))
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "RandomForestClassifier"
    assert "n_estimators" in data
    assert "n_features" in data
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) == data["n_features"]


def test_stats_requires_auth():
    """Stats without authentication should return 401."""
    response = client.get("/v1/stats")
    assert response.status_code == 401


def test_stats_allowed_for_analyst():
    resp = client.post("/v1/auth/login", json={"username": "analyst", "password": "password123"})
    tokens = resp.json()
    response = client.get("/v1/stats", headers=auth_header(tokens))
    assert response.status_code == 200


def test_stats_allowed_for_auditor():
    resp = client.post("/v1/auth/login", json={"username": "auditor", "password": "password123"})
    tokens = resp.json()
    response = client.get("/v1/stats", headers=auth_header(tokens))
    assert response.status_code == 200


def test_stats_denied_for_finance_user():
    resp = client.post("/v1/auth/login", json={"username": "finance", "password": "password123"})
    tokens = resp.json()
    response = client.get("/v1/stats", headers=auth_header(tokens))
    assert response.status_code == 403
    assert "Forbidden" in response.json()["error"]
