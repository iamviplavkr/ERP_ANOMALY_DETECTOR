"""
backend/services/prediction_service.py
─────────────────────────────────────────────────────────────────
Service layer implementation for managing single and batch predictions.
Decouples FastAPI handlers from business logic and database/repository access.
"""

from typing import Dict, Any, List
from backend.core.exceptions import PredictionError, BatchPredictionError
from backend.core.logging import get_logger
from backend.repositories.model_repository import ModelRepository
from backend.schemas.transaction import TransactionRequest, TransactionResponse
from ml.features.feature_engineering import engineer_features_from_request
from backend.utils.helpers import get_risk_level, get_top_risk_factors

logger = get_logger(__name__)


class PredictionService:
    """
    Main business service class for running ERP anomaly detection.
    """

    def __init__(self, repository: ModelRepository = None) -> None:
        self.repo = repository or ModelRepository()

    def predict_single(self, txn: TransactionRequest) -> TransactionResponse:
        """
        Processes and scores a single ERP transaction.
        """
        try:
            logger.debug(f"Processing transaction for vendor: {txn.vendor_id}")
            # Ensure model and scaler are loaded
            self.repo.load_artifacts()

            # 1. Feature Engineering
            features = engineer_features_from_request(txn)

            # 2. Standardize/Scale features
            features_sc = self.repo.scaler.transform(features)

            # 3. Model Inference
            proba = self.repo.model.predict_proba(features_sc)[0][1]
            is_fraud = bool(proba >= 0.5)

            # 4. Human-readable Risk Levels & SHAP-style importances
            risk_level, alert_message = get_risk_level(proba)
            top_factors = get_top_risk_factors(
                features, proba, self.repo.scaler, self.repo.model, self.repo.feature_cols
            )

            response = TransactionResponse(
                vendor_id=txn.vendor_id or "UNKNOWN",
                department=txn.department or "Finance",
                approved_by=txn.approved_by or "mgr_01",
                transaction_amount=txn.transaction_amount,
                anomaly_score=round(float(proba), 4),
                is_fraud=is_fraud,
                risk_level=risk_level,
                alert_message=alert_message,
                top_risk_factors=top_factors
            )
            logger.info(f"Transaction scored: vendor={txn.vendor_id}, score={response.anomaly_score}, is_fraud={is_fraud}")
            return response

        except Exception as e:
            logger.exception(f"Error predicting single transaction: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}") from e

    def predict_batch(self, transactions: List[TransactionRequest]) -> Dict[str, Any]:
        """
        Processes and scores a batch of ERP transactions.
        """
        try:
            logger.info(f"Processing batch of {len(transactions)} transactions.")
            results = []
            for txn in transactions:
                # We reuse the single prediction logic to capture risk level, importances, and exception handling
                results.append(self.predict_single(txn))

            flagged = [r for r in results if r.is_fraud]
            total = len(results)
            flagged_count = len(flagged)
            flag_rate_pct = (flagged_count / total * 100) if total > 0 else 0.0

            return {
                "total": total,
                "flagged": flagged_count,
                "flag_rate": f"{flag_rate_pct:.1f}%",
                "predictions": results
            }
        except Exception as e:
            logger.exception(f"Error predicting batch: {e}")
            raise BatchPredictionError(f"Batch prediction failed: {str(e)}") from e
