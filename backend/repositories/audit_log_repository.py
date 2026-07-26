"""
backend/repositories/audit_log_repository.py
─────────────────────────────────────────────────────────────────
Audit log storage repository interface, in-memory stub, and Postgres.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid

from sqlalchemy.orm import Session
from backend.models.audit_log import AuditLogModel


class AuditLogRepositoryInterface(ABC):
    """
    Abstract interface for managing audit log data access.
    """

    @abstractmethod
    def create(self, audit_log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and store a new audit log."""
        pass

    @abstractmethod
    def get_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an audit log by its ID."""
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all audit logs."""
        pass


class PostgresAuditLogRepository(AuditLogRepositoryInterface):
    """
    PostgreSQL-backed audit log repository.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def _to_dict(self, log: AuditLogModel) -> Dict[str, Any]:
        return {
            "id": str(log.id),
            "user_id": str(log.user_id) if log.user_id else None,
            "action": log.action,
            "resource": log.resource,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }

    def create(self, audit_log_data: Dict[str, Any]) -> Dict[str, Any]:
        log_id = audit_log_data.get("id")
        if isinstance(log_id, str):
            log_id = uuid.UUID(log_id)
        elif not log_id:
            log_id = uuid.uuid4()

        user_id = audit_log_data.get("user_id")
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)

        log = AuditLogModel(
            id=log_id,
            user_id=user_id,
            action=audit_log_data.get("action"),
            resource=audit_log_data.get("resource"),
            details=audit_log_data.get("details"),
            ip_address=audit_log_data.get("ip_address"),
        )
        self.db.add(log)
        # Flush to make ID available without committing transaction
        self.db.flush()
        return self._to_dict(log)

    def get_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        log = self.db.query(AuditLogModel).filter(AuditLogModel.id == uuid.UUID(log_id)).first()
        return self._to_dict(log) if log else None

    def get_all(self) -> List[Dict[str, Any]]:
        logs = self.db.query(AuditLogModel).order_by(AuditLogModel.created_at.desc()).all()
        return [self._to_dict(log) for log in logs]


class InMemoryAuditLogRepository(AuditLogRepositoryInterface):
    """
    In-memory list-backed stub database for tests.
    """

    def __init__(self) -> None:
        self._logs: Dict[str, Dict[str, Any]] = {}

    def create(self, audit_log_data: Dict[str, Any]) -> Dict[str, Any]:
        log_id = audit_log_data.get("id") or str(uuid.uuid4())
        new_log = {
            "id": log_id,
            "user_id": audit_log_data.get("user_id"),
            "action": audit_log_data.get("action"),
            "resource": audit_log_data.get("resource"),
            "details": audit_log_data.get("details"),
            "ip_address": audit_log_data.get("ip_address"),
            "created_at": audit_log_data.get("created_at"),
        }
        self._logs[log_id] = new_log
        return new_log

    def get_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        return self._logs.get(log_id)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._logs.values())
