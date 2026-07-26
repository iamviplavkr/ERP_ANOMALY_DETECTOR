"""
tests/api/test_alerts_api.py
─────────────────────────────────────────────────────────────────
Tests for the /v1/alerts API endpoints covering:
  - RBAC enforcement (Admin, Fraud Analyst full access; Auditor read-only; Finance User 403)
  - List, get, and status update with valid/invalid lifecycle transitions
  - Audit log entry for status changes
"""

import pytest
from starlette.testclient import TestClient

from backend.main import app
from tests.conftest import auth_header
from backend.repositories.alert_repository import InMemoryAlertRepository
from backend.auth.auth_dependencies import get_alert_repository, get_audit_log_repository
from backend.repositories.audit_log_repository import InMemoryAuditLogRepository


client = TestClient(app)


def _seed_alert(repo: InMemoryAlertRepository) -> str:
    """Create a test alert in the repo and return its id."""
    alert = repo.create({
        "prediction_id": "00000000-0000-0000-0000-000000000001",
        "risk_level": "HIGH",
        "rules_triggered": ["Large Transaction Amount (High)"],
        "mitigation_action": "Request executive sign-off.",
    })
    return alert["id"]


def _login(username: str) -> dict:
    resp = client.post("/v1/auth/login", json={"username": username, "password": "password123"})
    assert resp.status_code == 200
    return resp.json()


# Override alert repo and audit log repo for all tests
@pytest.fixture(autouse=True)
def inject_test_repos():
    alert_repo = InMemoryAlertRepository()
    audit_repo = InMemoryAuditLogRepository()

    app.dependency_overrides[get_alert_repository] = lambda: alert_repo
    app.dependency_overrides[get_audit_log_repository] = lambda: audit_repo

    yield alert_repo, audit_repo

    app.dependency_overrides.pop(get_alert_repository, None)
    app.dependency_overrides.pop(get_audit_log_repository, None)


class TestAlertsRBAC:
    """Verify Finance User gets 403 and unauthenticated gets 403."""

    def test_finance_user_cannot_list_alerts(self, inject_test_repos):
        resp = client.get("/v1/alerts", headers=auth_header(_login("finance")))
        assert resp.status_code == 403

    def test_unauthenticated_cannot_list_alerts(self):
        resp = client.get("/v1/alerts")
        # Missing token → TokenInvalidError → 401 Unauthorized (not 403 Forbidden)
        assert resp.status_code == 401

    def test_finance_user_cannot_update_status(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "INVESTIGATING"},
            headers=auth_header(_login("finance")),
        )
        assert resp.status_code == 403


class TestAlertsList:
    """Test listing alerts for authorized roles."""

    def test_admin_can_list_empty_alerts(self, inject_test_repos):
        resp = client.get("/v1/alerts", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["alerts"] == []

    def test_admin_can_list_seeded_alerts(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_alert(repo)
        _seed_alert(repo)
        resp = client.get("/v1/alerts", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_analyst_can_list_alerts(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_alert(repo)
        resp = client.get("/v1/alerts", headers=auth_header(_login("analyst")))
        assert resp.status_code == 200

    def test_auditor_can_list_alerts(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_alert(repo)
        resp = client.get("/v1/alerts", headers=auth_header(_login("auditor")))
        assert resp.status_code == 200

    def test_filter_by_status(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_alert(repo)
        resp = client.get("/v1/alerts?status=OPEN", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["status"] == "OPEN" for a in data["alerts"])

    def test_filter_by_risk_level(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_alert(repo)
        resp = client.get("/v1/alerts?risk_level=HIGH", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["risk_level"] == "HIGH" for a in data["alerts"])


class TestAlertGet:
    """Test getting a specific alert."""

    def test_admin_can_get_alert(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.get(f"/v1/alerts/{alert_id}", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        assert resp.json()["id"] == alert_id

    def test_auditor_can_get_alert(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.get(f"/v1/alerts/{alert_id}", headers=auth_header(_login("auditor")))
        assert resp.status_code == 200

    def test_get_nonexistent_alert_returns_404(self, inject_test_repos):
        resp = client.get(
            "/v1/alerts/00000000-0000-0000-0000-deadbeef1234",
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 404


class TestAlertStatusUpdate:
    """Test alert lifecycle transitions."""

    def test_admin_open_to_investigating(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "INVESTIGATING"},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "INVESTIGATING"

    def test_analyst_investigating_to_resolved(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        repo.update_status(alert_id, "INVESTIGATING")
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "RESOLVED"},
            headers=auth_header(_login("analyst")),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "RESOLVED"

    def test_investigating_to_dismissed(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        repo.update_status(alert_id, "INVESTIGATING")
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "DISMISSED"},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "DISMISSED"

    def test_invalid_transition_open_to_resolved_returns_422(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "RESOLVED"},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 422

    def test_auditor_cannot_update_status(self, inject_test_repos):
        repo, _ = inject_test_repos
        alert_id = _seed_alert(repo)
        resp = client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "INVESTIGATING"},
            headers=auth_header(_login("auditor")),
        )
        assert resp.status_code == 403

    def test_status_change_creates_audit_log(self, inject_test_repos):
        repo, audit_repo = inject_test_repos
        alert_id = _seed_alert(repo)
        client.put(
            f"/v1/alerts/{alert_id}/status",
            json={"status": "INVESTIGATING"},
            headers=auth_header(_login("admin")),
        )
        logs = audit_repo.get_all()
        # Filter to only ALERT_STATUS_CHANGE logs (the LOGIN audit is also captured)
        status_change_logs = [l for l in logs if l["action"] == "ALERT_STATUS_CHANGE"]
        assert len(status_change_logs) == 1
        assert status_change_logs[0]["action"] == "ALERT_STATUS_CHANGE"
        assert status_change_logs[0]["details"]["new_status"] == "INVESTIGATING"

