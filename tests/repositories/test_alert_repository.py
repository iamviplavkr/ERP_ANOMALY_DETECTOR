"""
tests/repositories/test_alert_repository.py
─────────────────────────────────────────────────────────────────
Unit tests for PostgresAlertRepository (via SQLite) and InMemoryAlertRepository,
including lifecycle transition enforcement.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.repositories.alert_repository import (
    PostgresAlertRepository,
    InMemoryAlertRepository,
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


def _alert_data(**kwargs):
    defaults = {
        "prediction_id": "00000000-0000-0000-0000-000000000099",
        "risk_level": "HIGH",
        "status": "OPEN",
        "rules_triggered": ["High Anomaly Score (Critical)"],
        "mitigation_action": "Freeze transaction immediately.",
    }
    defaults.update(kwargs)
    return defaults


class TestInMemoryAlertRepository:
    def test_create_and_get(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        assert alert["risk_level"] == "HIGH"
        assert alert["status"] == "OPEN"
        fetched = repo.get_by_id(alert["id"])
        assert fetched is not None

    def test_list_all_unfiltered(self):
        repo = InMemoryAlertRepository()
        repo.create(_alert_data())
        repo.create(_alert_data(risk_level="CRITICAL"))
        assert len(repo.list_all()) == 2

    def test_list_filter_by_status(self):
        repo = InMemoryAlertRepository()
        repo.create(_alert_data())
        repo.create(_alert_data())
        all_alerts = repo.list_all()
        first_id = all_alerts[0]["id"]
        repo.update_status(first_id, "INVESTIGATING")
        assert len(repo.list_all(status="OPEN")) == 1
        assert len(repo.list_all(status="INVESTIGATING")) == 1

    def test_list_filter_by_risk_level(self):
        repo = InMemoryAlertRepository()
        repo.create(_alert_data(risk_level="HIGH"))
        repo.create(_alert_data(risk_level="CRITICAL"))
        assert len(repo.list_all(risk_level="HIGH")) == 1
        assert len(repo.list_all(risk_level="CRITICAL")) == 1

    def test_valid_transition_open_to_investigating(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        updated = repo.update_status(alert["id"], "INVESTIGATING")
        assert updated["status"] == "INVESTIGATING"

    def test_valid_transition_investigating_to_resolved(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        repo.update_status(alert["id"], "INVESTIGATING")
        resolved = repo.update_status(alert["id"], "RESOLVED")
        assert resolved["status"] == "RESOLVED"

    def test_valid_transition_investigating_to_dismissed(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        repo.update_status(alert["id"], "INVESTIGATING")
        dismissed = repo.update_status(alert["id"], "DISMISSED")
        assert dismissed["status"] == "DISMISSED"

    def test_invalid_transition_open_to_resolved_raises(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        with pytest.raises(ValueError, match="Invalid status transition"):
            repo.update_status(alert["id"], "RESOLVED")

    def test_invalid_transition_open_to_dismissed_raises(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        with pytest.raises(ValueError, match="Invalid status transition"):
            repo.update_status(alert["id"], "DISMISSED")

    def test_terminal_state_resolved_cannot_transition(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        repo.update_status(alert["id"], "INVESTIGATING")
        repo.update_status(alert["id"], "RESOLVED")
        with pytest.raises(ValueError, match="Invalid status transition"):
            repo.update_status(alert["id"], "INVESTIGATING")

    def test_unknown_status_raises(self):
        repo = InMemoryAlertRepository()
        alert = repo.create(_alert_data())
        with pytest.raises(ValueError, match="Unknown status"):
            repo.update_status(alert["id"], "BOGUS")

    def test_get_by_id_not_found(self):
        repo = InMemoryAlertRepository()
        assert repo.get_by_id("00000000-0000-0000-0000-999999999999") is None
