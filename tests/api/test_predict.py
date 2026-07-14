"""
tests/api/test_predict.py
─────────────────────────────────────────────────────────────────
Tests the single and batch prediction endpoints.
All prediction endpoints require JWT authentication with an authorised role.
"""

from starlette.testclient import TestClient
from backend.main import app
from tests.conftest import auth_header

client = TestClient(app)


def _get_admin_tokens():
    """Helper to authenticate as admin for prediction tests."""
    resp = client.post("/v1/auth/login", json={
        "username": "admin",
        "password": "password123"
    })
    return resp.json()


def test_predict_single():
    tokens = _get_admin_tokens()
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
    data = response.json()
    assert data["vendor_id"] == "V00123"
    assert "anomaly_score" in data
    assert isinstance(data["is_fraud"], bool)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_risk_factors"], list)


def test_predict_batch():
    tokens = _get_admin_tokens()
    payload = {
        "transactions": [
            {
                "vendor_id": "V00001",
                "department": "HR",
                "approved_by": "mgr_02",
                "posting_time": 1000.0,
                "transaction_amount": 50.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            },
            {
                "vendor_id": "V00002",
                "department": "Procurement",
                "approved_by": "mgr_03",
                "posting_time": 2000.0,
                "transaction_amount": 5000.0,
                **{f"V{i}": 1.5 for i in range(1, 29)}
            }
        ]
    }
    response = client.post("/v1/predict/batch", json=payload, headers=auth_header(tokens))
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "flagged" in data
    assert "flag_rate" in data
    assert len(data["predictions"]) == 2


def test_predict_single_requires_auth():
    """Prediction without authentication should return 401."""
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
