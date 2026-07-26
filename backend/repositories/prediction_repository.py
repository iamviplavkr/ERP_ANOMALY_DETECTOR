"""
backend/repositories/prediction_repository.py
─────────────────────────────────────────────────────────────────
Prediction storage repository interface, in-memory stub, and Postgres.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid

from sqlalchemy.orm import Session
from backend.models.prediction import PredictionModel


class PredictionRepositoryInterface(ABC):
    """
    Abstract interface for managing prediction data access.
    """

    @abstractmethod
    def create(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and store a new prediction."""
        pass

    @abstractmethod
    def get_by_id(self, pred_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a prediction by its ID."""
        pass

    @abstractmethod
    def get_by_transaction_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a prediction linked to a specific transaction."""
        pass


class PostgresPredictionRepository(PredictionRepositoryInterface):
    """
    PostgreSQL-backed prediction repository.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def _to_dict(self, pred: PredictionModel) -> Dict[str, Any]:
        return {
            "id": str(pred.id),
            "transaction_id": str(pred.transaction_id),
            "anomaly_score": pred.anomaly_score,
            "is_fraud": pred.is_fraud,
            "risk_level": pred.risk_level,
            "alert_message": pred.alert_message,
            "top_risk_factors": pred.top_risk_factors,
            "model_version": pred.model_version,
            "created_at": pred.created_at.isoformat() if pred.created_at else None,
        }

    def create(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        pred_id = prediction_data.get("id")
        if isinstance(pred_id, str):
            pred_id = uuid.UUID(pred_id)
        elif not pred_id:
            pred_id = uuid.uuid4()

        tx_id = prediction_data.get("transaction_id")
        if isinstance(tx_id, str):
            tx_id = uuid.UUID(tx_id)

        pred = PredictionModel(
            id=pred_id,
            transaction_id=tx_id,
            anomaly_score=prediction_data.get("anomaly_score"),
            is_fraud=prediction_data.get("is_fraud"),
            risk_level=prediction_data.get("risk_level"),
            alert_message=prediction_data.get("alert_message"),
            top_risk_factors=prediction_data.get("top_risk_factors"),
            model_version=prediction_data.get("model_version"),
        )
        self.db.add(pred)
        # Flush to make ID available without committing transaction
        self.db.flush()
        return self._to_dict(pred)

    def get_by_id(self, pred_id: str) -> Optional[Dict[str, Any]]:
        pred = self.db.query(PredictionModel).filter(PredictionModel.id == uuid.UUID(pred_id)).first()
        return self._to_dict(pred) if pred else None

    def get_by_transaction_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        pred = self.db.query(PredictionModel).filter(PredictionModel.transaction_id == uuid.UUID(tx_id)).first()
        return self._to_dict(pred) if pred else None


class InMemoryPredictionRepository(PredictionRepositoryInterface):
    """
    In-memory list-backed stub database for tests.
    """

    def __init__(self) -> None:
        self._predictions: Dict[str, Dict[str, Any]] = {}

    def create(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        pred_id = prediction_data.get("id") or str(uuid.uuid4())
        new_pred = {
            "id": pred_id,
            "transaction_id": prediction_data.get("transaction_id"),
            "anomaly_score": prediction_data.get("anomaly_score"),
            "is_fraud": prediction_data.get("is_fraud"),
            "risk_level": prediction_data.get("risk_level"),
            "alert_message": prediction_data.get("alert_message"),
            "top_risk_factors": prediction_data.get("top_risk_factors"),
            "model_version": prediction_data.get("model_version"),
            "created_at": prediction_data.get("created_at"),
        }
        self._predictions[pred_id] = new_pred
        return new_pred

    def get_by_id(self, pred_id: str) -> Optional[Dict[str, Any]]:
        return self._predictions.get(pred_id)

    def get_by_transaction_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        for pred in self._predictions.values():
            if pred["transaction_id"] == tx_id:
                return pred
        return None
