"""
tests/repositories/test_prediction_repository.py
─────────────────────────────────────────────────────────────────
Unit tests for PostgresPredictionRepository (via SQLite) and
InMemoryPredictionRepository.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.repositories.prediction_repository import (
    PostgresPredictionRepository,
    InMemoryPredictionRepository,
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


class TestPredictionRepositories:
    """Verify prediction repository implementations."""

    def test_postgres_repo_create_and_get(self, db_session):
        repo = PostgresPredictionRepository(db_session)
        pred_data = {
            "transaction_id": "00000000-0000-0000-0000-000000000099",
            "anomaly_score": 0.75,
            "is_fraud": True,
            "risk_level": "HIGH",
            "alert_message": "Flagged",
            "top_risk_factors": [{"feature": "amount", "influence": 0.5}],
            "model_version": "2.0.0",
        }
        res = repo.create(pred_data)
        assert res["transaction_id"] == "00000000-0000-0000-0000-000000000099"
        assert res["anomaly_score"] == 0.75
        assert res["is_fraud"] is True
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["anomaly_score"] == 0.75

        fetched_by_tx = repo.get_by_transaction_id("00000000-0000-0000-0000-000000000099")
        assert fetched_by_tx is not None
        assert fetched_by_tx["id"] == res["id"]

    def test_inmemory_repo_create_and_get(self):
        repo = InMemoryPredictionRepository()
        pred_data = {
            "transaction_id": "00000000-0000-0000-0000-000000000088",
            "anomaly_score": 0.22,
            "is_fraud": False,
            "risk_level": "LOW",
            "alert_message": "Safe",
            "top_risk_factors": [],
            "model_version": "2.0.0",
        }
        res = repo.create(pred_data)
        assert res["transaction_id"] == "00000000-0000-0000-0000-000000000088"
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["anomaly_score"] == 0.22

        fetched_by_tx = repo.get_by_transaction_id("00000000-0000-0000-0000-000000000088")
        assert fetched_by_tx is not None
        assert fetched_by_tx["id"] == res["id"]
