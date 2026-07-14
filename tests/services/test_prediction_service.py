"""
tests/services/test_prediction_service.py
─────────────────────────────────────────────────────────────────
Unit tests for the PredictionService layer.
"""

import pytest
from backend.services.prediction_service import PredictionService
from backend.schemas.transaction import TransactionRequest
from backend.core.exceptions import PredictionError


def test_prediction_service_single():
    service = PredictionService()
    txn = TransactionRequest(
        vendor_id="V007",
        department="HR",
        approved_by="mgr_02",
        posting_time=120.0,
        transaction_amount=999.0,
        **{f"V{i}": 0.1 for i in range(1, 29)}
    )
    result = service.predict_single(txn)
    assert result.vendor_id == "V007"
    assert result.transaction_amount == 999.0
    assert 0.0 <= result.anomaly_score <= 1.0


def test_prediction_service_batch():
    service = PredictionService()
    txns = [
        TransactionRequest(
            vendor_id=f"V{idx}",
            department="Finance",
            approved_by="mgr_01",
            posting_time=100.0,
            transaction_amount=100.0 + idx,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        )
        for idx in range(3)
    ]
    res = service.predict_batch(txns)
    assert res["total"] == 3
    assert len(res["predictions"]) == 3
    assert "flagged" in res
