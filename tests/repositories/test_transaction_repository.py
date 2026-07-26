"""
tests/repositories/test_transaction_repository.py
─────────────────────────────────────────────────────────────────
Unit tests for PostgresTransactionRepository (via SQLite) and
InMemoryTransactionRepository.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.repositories.transaction_repository import (
    PostgresTransactionRepository,
    InMemoryTransactionRepository,
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


class TestTransactionRepositories:
    """Verify transaction repository implementations."""

    def test_postgres_repo_create_and_get(self, db_session):
        repo = PostgresTransactionRepository(db_session)
        tx_data = {
            "vendor_id": "V001",
            "department": "Finance",
            "approved_by": "mgr_01",
            "posting_time": 100.0,
            "transaction_amount": 250.0,
            "pca_features": {"V1": 0.5, "V2": -0.2},
            "submitted_by": "00000000-0000-0000-0000-000000000001",
        }
        res = repo.create(tx_data)
        assert res["vendor_id"] == "V001"
        assert res["transaction_amount"] == 250.0
        assert res["pca_features"] == {"V1": 0.5, "V2": -0.2}
        assert res["submitted_by"] == "00000000-0000-0000-0000-000000000001"
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["vendor_id"] == "V001"

    def test_inmemory_repo_create_and_get(self):
        repo = InMemoryTransactionRepository()
        tx_data = {
            "vendor_id": "V002",
            "department": "Procurement",
            "approved_by": "mgr_02",
            "posting_time": 200.0,
            "transaction_amount": 5000.0,
            "pca_features": {"V1": 1.5, "V2": 0.3},
            "submitted_by": "00000000-0000-0000-0000-000000000002",
        }
        res = repo.create(tx_data)
        assert res["vendor_id"] == "V002"
        assert "id" in res

        fetched = repo.get_by_id(res["id"])
        assert fetched is not None
        assert fetched["vendor_id"] == "V002"
