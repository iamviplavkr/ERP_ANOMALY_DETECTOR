"""
backend/repositories/transaction_repository.py
─────────────────────────────────────────────────────────────────
Transaction storage repository interface, in-memory stub, and Postgres.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid

from sqlalchemy.orm import Session
from backend.models.transaction import TransactionModel


class TransactionRepositoryInterface(ABC):
    """
    Abstract interface for managing transaction data access.
    """

    @abstractmethod
    def create(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and store a new transaction."""
        pass

    @abstractmethod
    def get_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a transaction by its ID."""
        pass


class PostgresTransactionRepository(TransactionRepositoryInterface):
    """
    PostgreSQL-backed transaction repository.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def _to_dict(self, tx: TransactionModel) -> Dict[str, Any]:
        return {
            "id": str(tx.id),
            "vendor_id": tx.vendor_id,
            "department": tx.department,
            "approved_by": tx.approved_by,
            "posting_time": tx.posting_time,
            "transaction_amount": tx.transaction_amount,
            "pca_features": tx.pca_features,
            "submitted_by": str(tx.submitted_by) if tx.submitted_by else None,
            "created_at": tx.created_at.isoformat() if tx.created_at else None,
        }

    def create(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        tx_id = transaction_data.get("id")
        if isinstance(tx_id, str):
            tx_id = uuid.UUID(tx_id)
        elif not tx_id:
            tx_id = uuid.uuid4()

        sub_by = transaction_data.get("submitted_by")
        if isinstance(sub_by, str):
            sub_by = uuid.UUID(sub_by)

        tx = TransactionModel(
            id=tx_id,
            vendor_id=transaction_data.get("vendor_id"),
            department=transaction_data.get("department"),
            approved_by=transaction_data.get("approved_by"),
            posting_time=transaction_data.get("posting_time"),
            transaction_amount=transaction_data.get("transaction_amount"),
            pca_features=transaction_data.get("pca_features"),
            submitted_by=sub_by,
        )
        self.db.add(tx)
        # Flush to make ID available without committing transaction
        self.db.flush()
        return self._to_dict(tx)

    def get_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        tx = self.db.query(TransactionModel).filter(TransactionModel.id == uuid.UUID(tx_id)).first()
        return self._to_dict(tx) if tx else None


class InMemoryTransactionRepository(TransactionRepositoryInterface):
    """
    In-memory list-backed stub database for tests.
    """

    def __init__(self) -> None:
        self._transactions: Dict[str, Dict[str, Any]] = {}

    def create(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        tx_id = transaction_data.get("id") or str(uuid.uuid4())
        new_tx = {
            "id": tx_id,
            "vendor_id": transaction_data.get("vendor_id"),
            "department": transaction_data.get("department"),
            "approved_by": transaction_data.get("approved_by"),
            "posting_time": transaction_data.get("posting_time"),
            "transaction_amount": transaction_data.get("transaction_amount"),
            "pca_features": transaction_data.get("pca_features"),
            "submitted_by": transaction_data.get("submitted_by"),
            "created_at": transaction_data.get("created_at"),
        }
        self._transactions[tx_id] = new_tx
        return new_tx

    def get_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        return self._transactions.get(tx_id)
