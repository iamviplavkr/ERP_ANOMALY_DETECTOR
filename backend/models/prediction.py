"""
backend/models/prediction.py
─────────────────────────────────────────────────────────────────
ORM model for the ``predictions`` table.
Stores inference results linked to a transaction.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.session import Base


class PredictionModel(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("transactions.id"), nullable=False
    )
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_fraud: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    alert_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    top_risk_factors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} score={self.anomaly_score}>"
