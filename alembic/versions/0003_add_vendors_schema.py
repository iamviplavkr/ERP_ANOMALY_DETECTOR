"""Add vendors schema

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-25
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vendors",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("vendor_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("reputation_score", sa.Float(), nullable=False, server_default="100.0"),
        sa.Column("is_blacklisted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_watchlist", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("historical_alerts_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_transactions_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("historical_fraud_rate", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("last_transaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_alert_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_vendors_vendor_id", "vendors", ["vendor_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_vendors_vendor_id", table_name="vendors")
    op.drop_table("vendors")
