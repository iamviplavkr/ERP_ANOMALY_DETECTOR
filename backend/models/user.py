"""
backend/models/user.py
─────────────────────────────────────────────────────────────────
ORM model for the ``users`` table.

Roles are normalized: ``role_id`` is a foreign key to ``roles.id``.
The ``role_rel`` relationship eager-loads the role so that
repository methods can return the role *name* string expected by
the existing JWT / RBAC contracts.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.session import Base


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("roles.id"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # Eager-load the role for every query so ``role_rel.name`` is
    # always available without an extra round-trip.
    role_rel: Mapped["RoleModel"] = relationship(  # noqa: F821
        "RoleModel", lazy="joined"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"
