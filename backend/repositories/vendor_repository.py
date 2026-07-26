"""
backend/repositories/vendor_repository.py
─────────────────────────────────────────────────────────────────
Vendor repository interface, PostgreSQL implementation, and in-memory stub.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.vendor import VendorRiskModel


class VendorRepositoryInterface(ABC):
    @abstractmethod
    def create(self, vendor_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_by_vendor_id(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def update(self, vendor_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def delete(self, vendor_id: str) -> bool:
        pass

    @abstractmethod
    def list_all(
        self,
        is_blacklisted: Optional[bool] = None,
        is_watchlist: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        pass


class PostgresVendorRepository(VendorRepositoryInterface):
    def __init__(self, db: Session) -> None:
        self.db = db

    def _to_dict(self, vendor: VendorRiskModel) -> Dict[str, Any]:
        return {
            "id": str(vendor.id),
            "vendor_id": vendor.vendor_id,
            "name": vendor.name,
            "reputation_score": vendor.reputation_score,
            "is_blacklisted": vendor.is_blacklisted,
            "is_watchlist": vendor.is_watchlist,
            "historical_alerts_count": vendor.historical_alerts_count,
            "total_transactions_count": vendor.total_transactions_count,
            "historical_fraud_rate": vendor.historical_fraud_rate,
            "last_transaction_at": vendor.last_transaction_at.isoformat() if vendor.last_transaction_at else None,
            "last_alert_at": vendor.last_alert_at.isoformat() if vendor.last_alert_at else None,
            "created_at": vendor.created_at.isoformat() if vendor.created_at else None,
            "updated_at": vendor.updated_at.isoformat() if vendor.updated_at else None,
        }

    def create(self, vendor_data: Dict[str, Any]) -> Dict[str, Any]:
        vendor = VendorRiskModel(
            id=uuid.uuid4(),
            vendor_id=vendor_data["vendor_id"],
            name=vendor_data["name"],
            reputation_score=vendor_data.get("reputation_score", 100.0),
            is_blacklisted=vendor_data.get("is_blacklisted", False),
            is_watchlist=vendor_data.get("is_watchlist", False),
            historical_alerts_count=vendor_data.get("historical_alerts_count", 0),
            total_transactions_count=vendor_data.get("total_transactions_count", 0),
            historical_fraud_rate=vendor_data.get("historical_fraud_rate", 0.0),
            last_transaction_at=vendor_data.get("last_transaction_at"),
            last_alert_at=vendor_data.get("last_alert_at"),
        )
        self.db.add(vendor)
        self.db.flush()
        self.db.refresh(vendor)
        return self._to_dict(vendor)

    def get_by_vendor_id(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        vendor = self.db.query(VendorRiskModel).filter(
            func.lower(VendorRiskModel.vendor_id) == vendor_id.lower()
        ).first()
        return self._to_dict(vendor) if vendor else None

    def update(self, vendor_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        vendor = self.db.query(VendorRiskModel).filter(
            func.lower(VendorRiskModel.vendor_id) == vendor_id.lower()
        ).first()
        if not vendor:
            raise ValueError(f"Vendor '{vendor_id}' not found.")

        for key, val in updates.items():
            if hasattr(vendor, key):
                setattr(vendor, key, val)

        vendor.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        self.db.refresh(vendor)
        return self._to_dict(vendor)

    def delete(self, vendor_id: str) -> bool:
        vendor = self.db.query(VendorRiskModel).filter(
            func.lower(VendorRiskModel.vendor_id) == vendor_id.lower()
        ).first()
        if not vendor:
            return False
        self.db.delete(vendor)
        self.db.flush()
        return True

    def list_all(
        self,
        is_blacklisted: Optional[bool] = None,
        is_watchlist: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        q = self.db.query(VendorRiskModel)
        if is_blacklisted is not None:
            q = q.filter(VendorRiskModel.is_blacklisted == is_blacklisted)
        if is_watchlist is not None:
            q = q.filter(VendorRiskModel.is_watchlist == is_watchlist)
        q = q.order_by(VendorRiskModel.vendor_id.asc())
        return [self._to_dict(v) for v in q.all()]


class InMemoryVendorRepository(VendorRepositoryInterface):
    def __init__(self) -> None:
        self._vendors: Dict[str, Dict[str, Any]] = {}

    def _clone(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def create(self, vendor_data: Dict[str, Any]) -> Dict[str, Any]:
        vendor_id = vendor_data["vendor_id"]
        for k in self._vendors:
            if k.lower() == vendor_id.lower():
                raise ValueError(f"Vendor '{vendor_id}' already exists.")

        record = {
            "id": str(uuid.uuid4()),
            "vendor_id": vendor_id,
            "name": vendor_data["name"],
            "reputation_score": float(vendor_data.get("reputation_score", 100.0)),
            "is_blacklisted": bool(vendor_data.get("is_blacklisted", False)),
            "is_watchlist": bool(vendor_data.get("is_watchlist", False)),
            "historical_alerts_count": int(vendor_data.get("historical_alerts_count", 0)),
            "total_transactions_count": int(vendor_data.get("total_transactions_count", 0)),
            "historical_fraud_rate": float(vendor_data.get("historical_fraud_rate", 0.0)),
            "last_transaction_at": vendor_data.get("last_transaction_at").isoformat()
            if isinstance(vendor_data.get("last_transaction_at"), datetime)
            else vendor_data.get("last_transaction_at"),
            "last_alert_at": vendor_data.get("last_alert_at").isoformat()
            if isinstance(vendor_data.get("last_alert_at"), datetime)
            else vendor_data.get("last_alert_at"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
        }
        self._vendors[vendor_id.lower()] = record
        return self._clone(record)

    def get_by_vendor_id(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        record = self._vendors.get(vendor_id.lower())
        return self._clone(record) if record else None

    def update(self, vendor_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        record = self._vendors.get(vendor_id.lower())
        if not record:
            raise ValueError(f"Vendor '{vendor_id}' not found.")

        for key, val in updates.items():
            if key in ["reputation_score", "historical_fraud_rate"]:
                record[key] = float(val)
            elif key in ["is_blacklisted", "is_watchlist"]:
                record[key] = bool(val)
            elif key in ["historical_alerts_count", "total_transactions_count"]:
                record[key] = int(val)
            elif key in ["last_transaction_at", "last_alert_at"]:
                record[key] = val.isoformat() if isinstance(val, datetime) else val
            else:
                record[key] = val

        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._clone(record)

    def delete(self, vendor_id: str) -> bool:
        if vendor_id.lower() in self._vendors:
            del self._vendors[vendor_id.lower()]
            return True
        return False

    def list_all(
        self,
        is_blacklisted: Optional[bool] = None,
        is_watchlist: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        results = list(self._vendors.values())
        if is_blacklisted is not None:
            results = [r for r in results if r["is_blacklisted"] == is_blacklisted]
        if is_watchlist is not None:
            results = [r for r in results if r["is_watchlist"] == is_watchlist]
        results.sort(key=lambda x: x["vendor_id"])
        return [self._clone(r) for r in results]
