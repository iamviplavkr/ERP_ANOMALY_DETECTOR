"""
backend/repositories/postgres_user_repository.py
─────────────────────────────────────────────────────────────────
PostgreSQL-backed implementation of ``UserRepositoryInterface``.

Uses SQLAlchemy ORM queries against the ``users`` and ``roles``
tables.  The ``_to_dict`` helper guarantees the same
``Dict[str, Any]`` contract (keys: username, email, role,
is_active, password_hash) that ``InMemoryUserRepository`` returns,
so all existing JWT / RBAC / service code works unchanged.
"""

from typing import Any, Dict, Optional

import bcrypt
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.role import RoleModel
from backend.models.user import UserModel
from backend.repositories.user_repository import UserRepositoryInterface
from backend.schemas.user import UserCreate


class PostgresUserRepository(UserRepositoryInterface):
    """Concrete user repository backed by PostgreSQL via SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(user: UserModel) -> Dict[str, Any]:
        """Convert an ORM instance to the canonical user dict."""
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role_rel.name,  # join-loaded from roles table
            "is_active": user.is_active,
            "password_hash": user.password_hash,
        }

    # ── interface methods ────────────────────────────────────────────────

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        user = (
            self.db.query(UserModel)
            .filter(func.lower(UserModel.username) == username.lower())
            .first()
        )
        return self._to_dict(user) if user else None

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        user = (
            self.db.query(UserModel)
            .filter(func.lower(UserModel.email) == email.lower())
            .first()
        )
        return self._to_dict(user) if user else None

    def create(self, user: UserCreate) -> Dict[str, Any]:
        # Reject duplicates
        existing = (
            self.db.query(UserModel)
            .filter(func.lower(UserModel.username) == user.username.lower())
            .first()
        )
        if existing:
            raise ValueError(f"User '{user.username}' already exists.")

        # Look up normalised role
        role = self.db.query(RoleModel).filter(RoleModel.name == user.role).first()
        if not role:
            raise ValueError(f"Role '{user.role}' not found.")

        # Hash password
        salt = bcrypt.gensalt()
        pwd_hash = bcrypt.hashpw(
            user.password.encode("utf-8"), salt
        ).decode("utf-8")

        db_user = UserModel(
            username=user.username.lower(),
            email=user.email,
            password_hash=pwd_hash,
            role_id=role.id,
            is_active=user.is_active if user.is_active is not None else True,
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return self._to_dict(db_user)
