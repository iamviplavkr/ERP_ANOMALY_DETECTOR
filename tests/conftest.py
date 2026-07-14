"""
tests/conftest.py
─────────────────────────────────────────────────────────────────
Test configuration and global fixtures.
Provides auth helper fixtures for JWT-protected endpoint tests.
"""

import os
import sys
import pickle
import pytest
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings

# Override settings for tests
settings.ENVIRONMENT = "testing"
settings.DEBUG = True
settings.REQUIRE_API_KEY = False  # Disable for ease of testing endpoints


@pytest.fixture(scope="session", autouse=True)
def mock_model_artifacts():
    """
    Creates temporary mock model artifacts if they don't exist.
    Guarantees test suite can run even if real models haven't been trained yet.
    """
    # Force loading a small mock model if files are not present in root or artifacts/
    model_path = Path(settings.MODEL_PATH)
    scaler_path = Path(settings.SCALER_PATH)
    feature_cols_path = Path(settings.FEATURE_COLS_PATH)

    # If any artifact is missing, let's create a minimal mock one
    if not (model_path.exists() and scaler_path.exists() and feature_cols_path.exists()):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        logger_log = []

        # Create dummy data
        X = np.random.randn(10, 32)
        y = np.random.randint(0, 2, 10)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=2, random_state=42)
        rf.fit(X_sc, y)

        from backend.constants.ml_constants import ALL_FEATURE_COLS
        feature_cols = ALL_FEATURE_COLS

        # Make sure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(feature_cols_path, "wb") as f:
            pickle.dump(feature_cols, f)

    yield

    # Clean up can be managed, but since they might be needed we leave them or let user manage.


# ── Auth Test Helpers ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client():
    """Session-scoped TestClient for all API tests."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def _login_user(test_client, username: str = "admin", password: str = "password123") -> dict:
    """
    Helper to perform a login and return the full token response dict.
    Returns: {"access_token": ..., "refresh_token": ..., "username": ..., "role": ...}
    """
    response = test_client.post("/v1/auth/login", json={
        "username": username,
        "password": password
    })
    assert response.status_code == 200, f"Login failed for {username}: {response.text}"
    return response.json()


@pytest.fixture(scope="session")
def admin_tokens(test_client):
    """Returns token response for the admin user."""
    return _login_user(test_client, "admin", "password123")


@pytest.fixture(scope="session")
def finance_tokens(test_client):
    """Returns token response for the finance user."""
    return _login_user(test_client, "finance", "password123")


@pytest.fixture(scope="session")
def analyst_tokens(test_client):
    """Returns token response for the fraud analyst user."""
    return _login_user(test_client, "analyst", "password123")


@pytest.fixture(scope="session")
def auditor_tokens(test_client):
    """Returns token response for the auditor user."""
    return _login_user(test_client, "auditor", "password123")


def auth_header(tokens: dict) -> dict:
    """Utility to create Authorization header from token response."""
    return {"Authorization": f"Bearer {tokens['access_token']}"}
