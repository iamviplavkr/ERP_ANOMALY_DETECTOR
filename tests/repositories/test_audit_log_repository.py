"""
tests/repositories/test_audit_log_repository.py
─────────────────────────────────────────────────────────────────
Unit tests for PostgresAuditLogRepository (via SQLite) and
InMemoryAuditLogRepository.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.repositories.audit_log_repository import (
    PostgresAuditLogRepository,
    InMemoryAuditLogRepository,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


class TestAuditLogRepositories:
    """Verify audit log repository implementations."""

    def test_postgres_repo_create_and_get(self, db_session):
        repo = PostgresAuditLogRepository(db_session)
        log_data = {
            "user_id": "00000000-0000-0000-0000-000000000001",
            "action": "LOGIN",
            "resource": "/v1/auth/login",
            "details": {"username": "admin"},
            "ip_address": "127.0.0.1",
        }
        res = repo.create(log_data)
        assert res["action"] == "LOGIN"
        assert res["ip_address"] == "127.0.0.1"
        assert res["user_id"] == "00000000-0000-0000-0000-000000000001"
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["action"] == "LOGIN"

        all_logs = repo.get_all()
        assert len(all_logs) == 1
        assert all_logs[0]["id"] == res["id"]

    def test_inmemory_repo_create_and_get(self):
        repo = InMemoryAuditLogRepository()
        log_data = {
            "user_id": "00000000-0000-0000-0000-000000000002",
            "action": "LOGOUT",
            "resource": "/v1/auth/logout",
            "details": {"username": "finance"},
            "ip_address": "192.168.1.1",
        }
        res = repo.create(log_data)
        assert res["action"] == "LOGOUT"
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["action"] == "LOGOUT"

        all_logs = repo.get_all()
        assert len(all_logs) == 1
        assert all_logs[0]["id"] == res["id"]
