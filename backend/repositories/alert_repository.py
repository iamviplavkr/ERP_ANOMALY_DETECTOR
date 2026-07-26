"""
backend/repositories/alert_repository.py
─────────────────────────────────────────────────────────────────
Alert repository interface, PostgreSQL implementation, and in-memory stub.

Enforces lifecycle transitions: OPEN → INVESTIGATING → RESOLVED or DISMISSED
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy.orm import Session

from backend.models.alert import AlertModel

# Valid alert status transitions
VALID_TRANSITIONS: Dict[str, List[str]] = {
    "OPEN": ["INVESTIGATING"],
    "INVESTIGATING": ["RESOLVED", "DISMISSED"],
    "RESOLVED": [],
    "DISMISSED": [],
}
VALID_STATUSES = {"OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED"}


def _validate_transition(current: str, target: str) -> None:
    """Raise ValueError if the status transition is invalid."""
    if target not in VALID_STATUSES:
        raise ValueError(f"Unknown status '{target}'. Must be one of {sorted(VALID_STATUSES)}")
    allowed = VALID_TRANSITIONS.get(current, [])
    if target not in allowed:
        raise ValueError(
            f"Invalid status transition: {current!r} → {target!r}. "
            f"Allowed from {current!r}: {allowed or ['(none — terminal state)']}"
        )


class AlertRepositoryInterface(ABC):
    @abstractmethod
    def create(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_all(self, status: Optional[str] = None, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_status(self, alert_id: str, new_status: str) -> Dict[str, Any]:
        pass


class PostgresAlertRepository(AlertRepositoryInterface):
    def __init__(self, db: Session) -> None:
        self.db = db

    def _to_dict(self, alert: AlertModel) -> Dict[str, Any]:
        result = {
            "id": str(alert.id),
            "prediction_id": str(alert.prediction_id),
            "risk_level": alert.risk_level,
            "status": alert.status,
            "rules_triggered": alert.rules_triggered,
            "mitigation_action": alert.mitigation_action,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "updated_at": alert.updated_at.isoformat() if alert.updated_at else None,
        }
        # Include linked prediction snapshot if loaded
        if alert.prediction_rel:
            pred = alert.prediction_rel
            result["prediction"] = {
                "id": str(pred.id),
                "transaction_id": str(pred.transaction_id),
                "anomaly_score": pred.anomaly_score,
                "is_fraud": pred.is_fraud,
                "risk_level": pred.risk_level,
                "alert_message": pred.alert_message,
                "top_risk_factors": pred.top_risk_factors,
                "model_version": pred.model_version,
            }
            if pred.transaction_rel:
                tx = pred.transaction_rel
                result["prediction"]["transaction"] = {
                    "id": str(tx.id),
                    "vendor_id": tx.vendor_id,
                    "department": tx.department,
                    "approved_by": tx.approved_by,
                    "transaction_amount": tx.transaction_amount,
                }
        return result

    def create(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        pred_id = alert_data["prediction_id"]
        if isinstance(pred_id, str):
            pred_id = uuid.UUID(pred_id)

        alert = AlertModel(
            id=uuid.uuid4(),
            prediction_id=pred_id,
            risk_level=alert_data["risk_level"],
            status=alert_data.get("status", "OPEN"),
            rules_triggered=alert_data["rules_triggered"],
            mitigation_action=alert_data["mitigation_action"],
        )
        self.db.add(alert)
        self.db.flush()
        self.db.refresh(alert)
        return self._to_dict(alert)

    def get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        alert = self.db.query(AlertModel).filter(AlertModel.id == uuid.UUID(alert_id)).first()
        return self._to_dict(alert) if alert else None

    def list_all(self, status: Optional[str] = None, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        q = self.db.query(AlertModel)
        if status:
            q = q.filter(AlertModel.status == status.upper())
        if risk_level:
            q = q.filter(AlertModel.risk_level == risk_level.upper())
        q = q.order_by(AlertModel.created_at.desc())
        return [self._to_dict(a) for a in q.all()]

    def update_status(self, alert_id: str, new_status: str) -> Dict[str, Any]:
        alert = self.db.query(AlertModel).filter(AlertModel.id == uuid.UUID(alert_id)).first()
        if not alert:
            raise ValueError(f"Alert '{alert_id}' not found.")
        _validate_transition(alert.status, new_status.upper())
        alert.status = new_status.upper()
        # Use Python-side datetime for SQLite/PostgreSQL compatibility
        alert.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        self.db.refresh(alert)
        return self._to_dict(alert)


class InMemoryAlertRepository(AlertRepositoryInterface):
    def __init__(self) -> None:
        self._alerts: Dict[str, Dict[str, Any]] = {}

    def create(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        alert_id = str(uuid.uuid4())
        record = {
            "id": alert_id,
            "prediction_id": alert_data["prediction_id"],
            "risk_level": alert_data["risk_level"],
            "status": alert_data.get("status", "OPEN"),
            "rules_triggered": alert_data["rules_triggered"],
            "mitigation_action": alert_data["mitigation_action"],
            "created_at": alert_data.get("created_at"),
            "updated_at": alert_data.get("updated_at"),
        }
        self._alerts[alert_id] = record
        return record

    def get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        return self._alerts.get(alert_id)

    def list_all(self, status: Optional[str] = None, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self._alerts.values())
        if status:
            results = [a for a in results if a["status"] == status.upper()]
        if risk_level:
            results = [a for a in results if a["risk_level"] == risk_level.upper()]
        return results

    def update_status(self, alert_id: str, new_status: str) -> Dict[str, Any]:
        alert = self._alerts.get(alert_id)
        if not alert:
            raise ValueError(f"Alert '{alert_id}' not found.")
        _validate_transition(alert["status"], new_status.upper())
        alert["status"] = new_status.upper()
        return alert
