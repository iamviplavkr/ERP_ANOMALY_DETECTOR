"""
backend/models/vendor.py
─────────────────────────────────────────────────────────────────
ORM model for the ``vendors`` table.
Stores vendor profiles, reputation scores, blacklist/watchlist status,
and historical fraud metrics.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.session import Base


class VendorRiskModel(Base):
    __tablename__ = "vendors"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    vendor_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    reputation_score: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    is_blacklisted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_watchlist: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    historical_alerts_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_transactions_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    historical_fraud_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_transaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_alert_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    def __repr__(self) -> str:
        return f"<VendorRisk vendor_id={self.vendor_id!r} score={self.reputation_score}>"
