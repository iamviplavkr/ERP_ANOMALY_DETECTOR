"""
tests/services/test_vendor_persistence.py
─────────────────────────────────────────────────────────────────
Integration tests verifying vendor stat updates during prediction.
Uses all in-memory repositories to confirm that total_transactions_count,
historical_alerts_count, historical_fraud_rate, last_transaction_at, and
last_alert_at are correctly updated after a prediction is scored.
"""

import pytest
from backend.repositories.vendor_repository import InMemoryVendorRepository
from backend.repositories.alert_repository import InMemoryAlertRepository
from backend.repositories.prediction_repository import InMemoryPredictionRepository
from backend.repositories.transaction_repository import InMemoryTransactionRepository
from backend.services.vendor_service import VendorService
from backend.services.prediction_service import PredictionService
from backend.schemas.transaction import TransactionRequest


def _make_service(**kwargs) -> PredictionService:
    """Build a PredictionService with all in-memory repos."""
    defaults = {
        "transaction_repo": InMemoryTransactionRepository(),
        "prediction_repo": InMemoryPredictionRepository(),
        "alert_repo": InMemoryAlertRepository(),
        "vendor_repo": InMemoryVendorRepository(),
    }
    defaults.update(kwargs)
    return PredictionService(**defaults)


def _make_txn(vendor_id: str = "V_STATS", amount: float = 100.0) -> TransactionRequest:
    return TransactionRequest(
        vendor_id=vendor_id,
        department="Finance",
        approved_by="mgr_01",
        posting_time=100.0,
        transaction_amount=amount,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    )


def _seed_vendor(repo: InMemoryVendorRepository, vendor_id: str = "V_STATS", **kwargs) -> dict:
    data = {
        "vendor_id": vendor_id,
        "name": "Stats Test Vendor",
        "reputation_score": 90.0,
        "is_blacklisted": False,
        "is_watchlist": False,
        "historical_alerts_count": 0,
        "total_transactions_count": 0,
        "historical_fraud_rate": 0.0,
        **kwargs,
    }
    return repo.create(data)


class TestVendorStatTracking:
    """Verify that running predictions updates vendor stats correctly."""

    def test_prediction_increments_transaction_count(self, mock_model_artifacts):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo)
        service = _make_service(vendor_repo=vendor_repo)
        service.predict_single(_make_txn())

        updated = vendor_repo.get_by_vendor_id("V_STATS")
        assert updated["total_transactions_count"] == 1

    def test_prediction_updates_last_transaction_at(self, mock_model_artifacts):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo)
        service = _make_service(vendor_repo=vendor_repo)
        service.predict_single(_make_txn())

        updated = vendor_repo.get_by_vendor_id("V_STATS")
        assert updated["last_transaction_at"] is not None

    def test_auto_creates_default_vendor_profile(self, mock_model_artifacts):
        """Vendor not in DB before prediction — should be auto-created."""
        vendor_repo = InMemoryVendorRepository()
        assert vendor_repo.get_by_vendor_id("BRAND_NEW") is None
        service = _make_service(vendor_repo=vendor_repo)
        service.predict_single(_make_txn(vendor_id="BRAND_NEW"))

        profile = vendor_repo.get_by_vendor_id("BRAND_NEW")
        assert profile is not None
        assert profile["total_transactions_count"] == 1

    def test_blacklisted_vendor_increments_alert_count(self, mock_model_artifacts):
        """Blacklisted vendor → CRITICAL → alert fires → historical_alerts_count increments."""
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo, is_blacklisted=True)
        service = _make_service(vendor_repo=vendor_repo)
        service.predict_single(_make_txn())

        updated = vendor_repo.get_by_vendor_id("V_STATS")
        assert updated["total_transactions_count"] == 1
        # Blacklisted triggers CRITICAL → alert created → count should increment
        assert updated["historical_alerts_count"] >= 1
        assert updated["last_alert_at"] is not None

    def test_multiple_predictions_accumulate_stats(self, mock_model_artifacts):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo)
        service = _make_service(vendor_repo=vendor_repo)
        service.predict_single(_make_txn())
        service.predict_single(_make_txn())

        updated = vendor_repo.get_by_vendor_id("V_STATS")
        assert updated["total_transactions_count"] == 2


class TestVendorService:
    """Basic VendorService unit tests."""

    def test_get_or_create_returns_existing(self):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo, reputation_score=55.0)
        svc = VendorService(vendor_repo)
        profile = svc.get_or_create_default("V_STATS")
        assert profile["reputation_score"] == 55.0  # Preserved, not overwritten

    def test_get_or_create_creates_default_when_missing(self):
        vendor_repo = InMemoryVendorRepository()
        svc = VendorService(vendor_repo)
        profile = svc.get_or_create_default("MISSING_VENDOR")
        assert profile is not None
        assert profile["vendor_id"] == "MISSING_VENDOR"
        assert profile["reputation_score"] == 90.0
        assert profile["is_blacklisted"] is False

    def test_list_vendors_with_blacklist_filter(self):
        vendor_repo = InMemoryVendorRepository()
        svc = VendorService(vendor_repo)
        vendor_repo.create({"vendor_id": "V_BL", "name": "Blacklisted", "reputation_score": 5.0,
                            "is_blacklisted": True, "is_watchlist": False})
        vendor_repo.create({"vendor_id": "V_OK", "name": "Clean", "reputation_score": 90.0,
                            "is_blacklisted": False, "is_watchlist": False})
        blacklisted = svc.list_vendors(is_blacklisted=True)
        assert len(blacklisted) == 1
        assert blacklisted[0]["vendor_id"] == "V_BL"

    def test_update_vendor_reputation(self):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo)
        svc = VendorService(vendor_repo)
        updated = svc.update_vendor("V_STATS", {"reputation_score": 30.0})
        assert updated["reputation_score"] == 30.0

    def test_delete_vendor(self):
        vendor_repo = InMemoryVendorRepository()
        _seed_vendor(vendor_repo)
        svc = VendorService(vendor_repo)
        assert svc.delete_vendor("V_STATS") is True
        assert svc.get_by_vendor_id("V_STATS") is None


