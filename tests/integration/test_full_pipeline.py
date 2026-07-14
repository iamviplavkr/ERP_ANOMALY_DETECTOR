"""
tests/integration/test_full_pipeline.py
─────────────────────────────────────────────────────────────────
Integration tests running the full prediction flow:
Request -> Feature Engineering -> Model Inference -> Explanation -> Response.
"""

from backend.services.prediction_service import PredictionService
from backend.schemas.transaction import TransactionRequest
from backend.repositories.model_repository import ModelRepository


def test_full_pipeline_flow():
    repo = ModelRepository()
    repo.load_artifacts()

    service = PredictionService(repository=repo)

    txn = TransactionRequest(
        vendor_id="V_INT_TEST",
        department="Procurement",
        approved_by="mgr_03",
        posting_time=86400.0,
        transaction_amount=1500.0,
        **{f"V{i}": -0.5 for i in range(1, 29)}
    )

    result = service.predict_single(txn)

    assert result.vendor_id == "V_INT_TEST"
    assert result.transaction_amount == 1500.0
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_fraud, bool)
    assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert len(result.top_risk_factors) > 0
    assert result.top_risk_factors[0]["feature"] in repo.feature_cols
