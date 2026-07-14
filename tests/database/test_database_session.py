"""
tests/database/test_database_session.py
─────────────────────────────────────────────────────────────────
Unit tests for the database session module.
"""

import pytest

from backend.core.config import settings
from backend.database.session import Base, get_db


class TestBase:
    """Verify the declarative Base is usable."""

    def test_base_has_metadata(self):
        assert Base.metadata is not None

    def test_base_metadata_contains_tables(self):
        # After importing backend.models, all tables should be registered
        import backend.models  # noqa: F401
        table_names = set(Base.metadata.tables.keys())
        assert "users" in table_names
        assert "roles" in table_names
        assert "transactions" in table_names
        assert "predictions" in table_names
        assert "audit_logs" in table_names


class TestGetDb:
    """Verify get_db dependency behaviour in testing mode."""

    def test_get_db_yields_none_in_testing(self):
        """In testing mode, get_db should yield None (no real DB)."""
        assert settings.ENVIRONMENT == "testing"
        gen = get_db()
        value = next(gen)
        assert value is None
        # Exhaust the generator
        with pytest.raises(StopIteration):
            next(gen)
