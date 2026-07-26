"""
backend/models/alert.py
─────────────────────────────────────────────────────────────────
ORM model for the ``alerts`` table.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.session import Base


class AlertModel(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("predictions.id"), nullable=False
    )
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="OPEN", nullable=False)
    rules_triggered: Mapped[list] = mapped_column(JSON, nullable=False)
    mitigation_action: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # Relationship to prediction (which links to transaction)
    prediction_rel: Mapped["PredictionModel"] = relationship(  # noqa: F821
        "PredictionModel", lazy="joined"
    )

    def __repr__(self) -> str:
        return f"<Alert id={self.id} status={self.status} risk={self.risk_level}>"
