"""
backend/models/transaction.py
─────────────────────────────────────────────────────────────────
ORM model for the ``transactions`` table.
Stores raw ERP transaction data including PCA features as JSON.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.session import Base


class TransactionModel(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    vendor_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    department: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    posting_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    transaction_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    pca_features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    submitted_by: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Transaction id={self.id} vendor={self.vendor_id!r}>"
