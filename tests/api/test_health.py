"""
tests/api/test_health.py
─────────────────────────────────────────────────────────────────
Tests the health check and root endpoints.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Swagger UI" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "RandomForestClassifier"
    assert isinstance(data["features"], int)
