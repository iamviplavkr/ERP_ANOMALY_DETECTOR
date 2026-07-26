"""
tests/api/test_analytics_api.py
─────────────────────────────────────────────────────────────────
Integration tests for the /v1/analytics API endpoints.
Covers:
  - RBAC checks (Admin, Analyst, Auditor allowed; Finance 403; unauth 401)
  - Date parsing and validation
  - Correct aggregation stats for overview, trends, risk-distribution, depts
  - AUDIT_LOG entry (ANALYTICS_EXPORT) generated during CSV export
  - Cache response verification (re-fetching uses cached values)
"""

from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from backend.main import app
from tests.conftest import auth_header
from backend.core.config import settings
from backend.repositories.transaction_repository import InMemoryTransactionRepository
from backend.repositories.prediction_repository import InMemoryPredictionRepository
from backend.repositories.alert_repository import InMemoryAlertRepository
from backend.repositories.vendor_repository import InMemoryVendorRepository
from backend.repositories.audit_log_repository import InMemoryAuditLogRepository
from backend.auth.auth_dependencies import (
    get_transaction_repository,
    get_prediction_repository,
    get_alert_repository,
    get_vendor_repository,
    get_audit_log_repository,
    get_db,
)
from backend.database.session import get_db as actual_get_db

client = TestClient(app)


def _login(username: str) -> dict:
    resp = client.post("/v1/auth/login", json={"username": username, "password": "password123"})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture(autouse=True)
def inject_analytics_test_repos():
    tx_repo = InMemoryTransactionRepository()
    pred_repo = InMemoryPredictionRepository()
    alert_repo = InMemoryAlertRepository()
    vendor_repo = InMemoryVendorRepository()
    audit_repo = InMemoryAuditLogRepository()

    # Pre-seed some test metrics
    tx1 = tx_repo.create({
        "id": "11111111-1111-1111-1111-111111111111",
        "vendor_id": "V001",
        "department": "Finance",
        "approved_by": "mgr_01",
        "posting_time": 100.0,
        "transaction_amount": 1000.0,
        "created_at": datetime.now(timezone.utc),
    })
    pred_repo.create({
        "transaction_id": tx1["id"],
        "anomaly_score": 0.25,
        "is_fraud": False,
        "risk_level": "LOW",
        "created_at": datetime.now(timezone.utc),
    })

    tx2 = tx_repo.create({
        "id": "22222222-2222-2222-2222-222222222222",
        "vendor_id": "V002",
        "department": "Procurement",
        "approved_by": "mgr_02",
        "posting_time": 200.0,
        "transaction_amount": 60000.0,
        "created_at": datetime.now(timezone.utc),
    })
    pred2 = pred_repo.create({
        "id": "88888888-8888-8888-8888-888888888888",
        "transaction_id": tx2["id"],
        "anomaly_score": 0.85,
        "is_fraud": True,
        "risk_level": "CRITICAL",
        "created_at": datetime.now(timezone.utc),
    })
    alert_repo.create({
        "prediction_id": pred2["id"],
        "risk_level": "CRITICAL",
        "status": "OPEN",
        "rules_triggered": ["Procurement Anomaly"],
        "mitigation_action": "Freeze vendor",
        "created_at": datetime.now(timezone.utc),
    })

    # Override dependencies
    app.dependency_overrides[get_transaction_repository] = lambda: tx_repo
    app.dependency_overrides[get_prediction_repository] = lambda: pred_repo
    app.dependency_overrides[get_alert_repository] = lambda: alert_repo
    app.dependency_overrides[get_vendor_repository] = lambda: vendor_repo
    app.dependency_overrides[get_audit_log_repository] = lambda: audit_repo
    app.dependency_overrides[actual_get_db] = lambda: None

    yield tx_repo, pred_repo, alert_repo, vendor_repo, audit_repo

    app.dependency_overrides.pop(get_transaction_repository, None)
    app.dependency_overrides.pop(get_prediction_repository, None)
    app.dependency_overrides.pop(get_alert_repository, None)
    app.dependency_overrides.pop(get_vendor_repository, None)
    app.dependency_overrides.pop(get_audit_log_repository, None)
    app.dependency_overrides.pop(actual_get_db, None)


# ── RBAC Checks ───────────────────────────────────────────────────────────────

class TestAnalyticsRBAC:
    def test_unauthenticated_returns_401(self):
        resp = client.get("/v1/analytics/overview")
        assert resp.status_code == 401

    def test_finance_user_is_forbidden(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/overview", headers=auth_header(_login("finance")))
        assert resp.status_code == 403

    def test_admin_is_allowed(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/overview", headers=auth_header(_login("admin")))
        assert resp.status_code == 200

    def test_analyst_is_allowed(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/overview", headers=auth_header(_login("analyst")))
        assert resp.status_code == 200

    def test_auditor_is_allowed(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/overview", headers=auth_header(_login("auditor")))
        assert resp.status_code == 200


# ── Query Analytics Tests ─────────────────────────────────────────────────────

class TestAnalyticsAggregations:
    def test_overview_kpis(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/overview", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_transactions"] == 2
        assert data["total_amount"] == 61000.0
        assert data["flagged_anomalies"] == 1
        assert data["open_alerts"] == 1

    def test_trends_data(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/trends", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["count"] == 2
        assert "average_anomaly_score" in data[0]

    def test_risk_distribution(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/risk-distribution", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        levels = [item["risk_level"] for item in data]
        assert "LOW" in levels
        assert "CRITICAL" in levels

    def test_department_metrics(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/departments", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        depts = [item["department"] for item in data]
        assert "Finance" in depts
        assert "Procurement" in depts

    def test_invalid_date_format_returns_400(self, inject_analytics_test_repos):
        resp = client.get(
            "/v1/analytics/overview?start_date=invalid-date",
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 400
        assert "Invalid date format" in resp.json()["detail"]


# ── CSV Export & Audit Log ────────────────────────────────────────────────────

class TestAnalyticsExport:
    def test_export_returns_csv(self, inject_analytics_test_repos):
        resp = client.get("/v1/analytics/export", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment; filename=analytics_export_" in resp.headers["content-disposition"]
        content = resp.text
        assert "prediction_id,transaction_id,vendor_id,department" in content
        assert "Finance" in content
        assert "Procurement" in content

    def test_export_generates_audit_log(self, inject_analytics_test_repos):
        _, _, _, _, audit_repo = inject_analytics_test_repos
        resp = client.get("/v1/analytics/export", headers=auth_header(_login("admin")))
        assert resp.status_code == 200

        logs = audit_repo.get_all()
        export_logs = [l for l in logs if l["action"] == "ANALYTICS_EXPORT"]
        assert len(export_logs) == 1
        details = export_logs[0]["details"]
        assert details["records_count"] == 2
        assert details["exported_by"] == "admin"


# ── Configurable Response Caching ─────────────────────────────────────────────

class TestAnalyticsCaching:
    def test_responses_are_cached(self, inject_analytics_test_repos, monkeypatch):
        monkeypatch.setattr(settings, "ANALYTICS_CACHE_TTL_SECONDS", 300)
        tx_repo, _, _, _, _ = inject_analytics_test_repos
        headers = auth_header(_login("admin"))

        # Step 1: Initial query
        resp1 = client.get("/v1/analytics/overview", headers=headers)
        val1 = resp1.json()

        # Step 2: Seed another transaction directly in repository (bypassing predict API)
        tx_repo.create({
            "id": "33333333-3333-3333-3333-333333333333",
            "vendor_id": "V003",
            "department": "Finance",
            "approved_by": "mgr_03",
            "posting_time": 150.0,
            "transaction_amount": 500.0,
            "created_at": datetime.now(timezone.utc),
        })

        # Step 3: Fetch overview again. Since it is cached, it should still return the old stats.
        resp2 = client.get("/v1/analytics/overview", headers=headers)
        val2 = resp2.json()

        assert val2["total_transactions"] == val1["total_transactions"]
        assert val2["total_transactions"] == 2  # Not 3
