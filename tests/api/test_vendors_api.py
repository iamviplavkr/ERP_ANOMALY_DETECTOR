"""
tests/api/test_vendors_api.py
─────────────────────────────────────────────────────────────────
Integration tests for the /v1/vendors REST API.
Covers:
  - RBAC enforcement (Admin/Analyst full; Auditor read-only; Finance 403; unauth 401)
  - Full CRUD lifecycle: create, list, get, update, delete
  - Duplicate creation rejection (400)
  - Missing vendor 404 handling
  - Audit log entries for VENDOR_CREATE, VENDOR_UPDATE, VENDOR_DELETE
"""

import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient

from backend.main import app
from tests.conftest import auth_header
from backend.repositories.vendor_repository import InMemoryVendorRepository
from backend.repositories.audit_log_repository import InMemoryAuditLogRepository
from backend.auth.auth_dependencies import get_vendor_repository, get_audit_log_repository
from backend.database.session import get_db


client = TestClient(app)


def _login(username: str) -> dict:
    resp = client.post("/v1/auth/login", json={"username": username, "password": "password123"})
    assert resp.status_code == 200
    return resp.json()


_VENDOR_PAYLOAD = {
    "vendor_id": "TEST_V001",
    "name": "Test Vendor Inc.",
    "reputation_score": 75.0,
    "is_blacklisted": False,
    "is_watchlist": False,
}


@pytest.fixture(autouse=True)
def inject_test_repos():
    vendor_repo = InMemoryVendorRepository()
    audit_repo = InMemoryAuditLogRepository()
    mock_db = MagicMock()  # Prevents real DB connections

    app.dependency_overrides[get_vendor_repository] = lambda: vendor_repo
    app.dependency_overrides[get_audit_log_repository] = lambda: audit_repo
    app.dependency_overrides[get_db] = lambda: mock_db

    yield vendor_repo, audit_repo

    app.dependency_overrides.pop(get_vendor_repository, None)
    app.dependency_overrides.pop(get_audit_log_repository, None)
    app.dependency_overrides.pop(get_db, None)


def _seed_vendor(repo: InMemoryVendorRepository, vendor_id: str = "TEST_V001") -> dict:
    return repo.create({
        "vendor_id": vendor_id,
        "name": "Seeded Vendor",
        "reputation_score": 80.0,
        "is_blacklisted": False,
        "is_watchlist": False,
    })


# ── RBAC Tests ────────────────────────────────────────────────────────────────

class TestVendorRBAC:

    def test_unauthenticated_list_returns_401(self):
        resp = client.get("/v1/vendors")
        assert resp.status_code == 401

    def test_finance_user_cannot_list_vendors(self, inject_test_repos):
        resp = client.get("/v1/vendors", headers=auth_header(_login("finance")))
        assert resp.status_code == 403

    def test_finance_user_cannot_create_vendor(self, inject_test_repos):
        resp = client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=auth_header(_login("finance")))
        assert resp.status_code == 403

    def test_auditor_can_list_vendors(self, inject_test_repos):
        resp = client.get("/v1/vendors", headers=auth_header(_login("auditor")))
        assert resp.status_code == 200

    def test_auditor_cannot_create_vendor(self, inject_test_repos):
        resp = client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=auth_header(_login("auditor")))
        assert resp.status_code == 403

    def test_auditor_cannot_update_vendor(self, inject_test_repos):
        resp = client.put(
            "/v1/vendors/SOME_VENDOR",
            json={"reputation_score": 50.0},
            headers=auth_header(_login("auditor")),
        )
        assert resp.status_code == 403

    def test_auditor_cannot_delete_vendor(self, inject_test_repos):
        resp = client.delete("/v1/vendors/SOME_VENDOR", headers=auth_header(_login("auditor")))
        assert resp.status_code == 403

    def test_admin_can_list_vendors(self, inject_test_repos):
        resp = client.get("/v1/vendors", headers=auth_header(_login("admin")))
        assert resp.status_code == 200

    def test_analyst_can_create_vendor(self, inject_test_repos):
        resp = client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=auth_header(_login("analyst")))
        assert resp.status_code == 201


# ── CRUD Tests ────────────────────────────────────────────────────────────────

class TestVendorCRUD:

    def test_create_vendor_returns_201(self, inject_test_repos):
        resp = client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=auth_header(_login("admin")))
        assert resp.status_code == 201
        data = resp.json()
        assert data["vendor_id"] == "TEST_V001"
        assert data["name"] == "Test Vendor Inc."
        assert data["reputation_score"] == 75.0
        assert data["is_blacklisted"] is False
        assert data["is_watchlist"] is False
        assert "id" in data

    def test_create_duplicate_vendor_returns_400(self, inject_test_repos):
        hdrs = auth_header(_login("admin"))
        client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=hdrs)
        resp = client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=hdrs)
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"].lower()

    def test_list_vendors_returns_all(self, inject_test_repos):
        repo, _ = inject_test_repos
        _seed_vendor(repo, "V_A")
        _seed_vendor(repo, "V_B")
        resp = client.get("/v1/vendors", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["vendors"]) == 2

    def test_list_vendors_filter_blacklisted(self, inject_test_repos):
        repo, _ = inject_test_repos
        repo.create({"vendor_id": "BL1", "name": "Blacklisted", "reputation_score": 10.0, "is_blacklisted": True, "is_watchlist": False})
        repo.create({"vendor_id": "OK1", "name": "Clean", "reputation_score": 90.0, "is_blacklisted": False, "is_watchlist": False})
        resp = client.get("/v1/vendors?is_blacklisted=true", headers=auth_header(_login("admin")))
        data = resp.json()
        assert data["total"] == 1
        assert data["vendors"][0]["vendor_id"] == "BL1"

    def test_get_vendor_by_id(self, inject_test_repos):
        repo, _ = inject_test_repos
        v = _seed_vendor(repo)
        resp = client.get(f"/v1/vendors/{v['vendor_id']}", headers=auth_header(_login("admin")))
        assert resp.status_code == 200
        assert resp.json()["vendor_id"] == v["vendor_id"]

    def test_get_nonexistent_vendor_returns_404(self, inject_test_repos):
        resp = client.get("/v1/vendors/NONEXISTENT_XYZ", headers=auth_header(_login("admin")))
        assert resp.status_code == 404

    def test_update_vendor_reputation(self, inject_test_repos):
        repo, _ = inject_test_repos
        v = _seed_vendor(repo)
        resp = client.put(
            f"/v1/vendors/{v['vendor_id']}",
            json={"reputation_score": 25.0},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 200
        assert resp.json()["reputation_score"] == 25.0

    def test_update_vendor_blacklist_flag(self, inject_test_repos):
        repo, _ = inject_test_repos
        v = _seed_vendor(repo)
        resp = client.put(
            f"/v1/vendors/{v['vendor_id']}",
            json={"is_blacklisted": True},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 200
        assert resp.json()["is_blacklisted"] is True

    def test_update_nonexistent_vendor_returns_404(self, inject_test_repos):
        resp = client.put(
            "/v1/vendors/GHOST_VENDOR",
            json={"reputation_score": 50.0},
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 404

    def test_delete_vendor_returns_204(self, inject_test_repos):
        repo, _ = inject_test_repos
        v = _seed_vendor(repo)
        resp = client.delete(
            f"/v1/vendors/{v['vendor_id']}",
            headers=auth_header(_login("admin")),
        )
        assert resp.status_code == 204

    def test_delete_nonexistent_vendor_returns_404(self, inject_test_repos):
        resp = client.delete("/v1/vendors/GHOST_VENDOR", headers=auth_header(_login("admin")))
        assert resp.status_code == 404


# ── Audit Log Tests ────────────────────────────────────────────────────────────

class TestVendorAuditLogs:

    def test_create_vendor_generates_audit_log(self, inject_test_repos):
        _, audit_repo = inject_test_repos
        client.post("/v1/vendors", json=_VENDOR_PAYLOAD, headers=auth_header(_login("admin")))
        logs = audit_repo.get_all()
        vendor_logs = [l for l in logs if l["action"] == "VENDOR_CREATE"]
        assert len(vendor_logs) == 1
        assert vendor_logs[0]["details"]["vendor_id"] == "TEST_V001"

    def test_update_vendor_generates_audit_log(self, inject_test_repos):
        repo, audit_repo = inject_test_repos
        v = _seed_vendor(repo)
        client.put(
            f"/v1/vendors/{v['vendor_id']}",
            json={"is_blacklisted": True},
            headers=auth_header(_login("admin")),
        )
        logs = audit_repo.get_all()
        update_logs = [l for l in logs if l["action"] == "VENDOR_UPDATE"]
        assert len(update_logs) == 1
        assert update_logs[0]["details"]["vendor_id"] == v["vendor_id"]

    def test_delete_vendor_generates_audit_log(self, inject_test_repos):
        repo, audit_repo = inject_test_repos
        v = _seed_vendor(repo)
        client.delete(
            f"/v1/vendors/{v['vendor_id']}",
            headers=auth_header(_login("admin")),
        )
        logs = audit_repo.get_all()
        delete_logs = [l for l in logs if l["action"] == "VENDOR_DELETE"]
        assert len(delete_logs) == 1
        assert delete_logs[0]["details"]["vendor_id"] == v["vendor_id"]
