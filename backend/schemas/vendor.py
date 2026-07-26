"""
backend/schemas/vendor.py
─────────────────────────────────────────────────────────────────
Pydantic schemas for vendor profile request/response payloads.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class VendorResponse(BaseModel):
    """Response schema for a single vendor profile."""
    id: str
    vendor_id: str
    name: str
    reputation_score: float
    is_blacklisted: bool
    is_watchlist: bool
    historical_alerts_count: int
    total_transactions_count: int
    historical_fraud_rate: float
    last_transaction_at: Optional[str] = None
    last_alert_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VendorCreateRequest(BaseModel):
    """Request schema for creating a new vendor profile."""
    vendor_id: str = Field(..., description="Unique alphanumeric vendor identifier")
    name: str = Field(..., description="Human-readable business name of the vendor")
    reputation_score: float = Field(100.0, ge=0.0, le=100.0, description="Reputation score (0 to 100)")
    is_blacklisted: bool = Field(False, description="Flag indicating if the vendor is blacklisted")
    is_watchlist: bool = Field(False, description="Flag indicating if the vendor is on the watchlist")


class VendorUpdateRequest(BaseModel):
    """Request schema for updating an existing vendor profile."""
    name: Optional[str] = Field(None, description="Human-readable business name of the vendor")
    reputation_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Reputation score (0 to 100)")
    is_blacklisted: Optional[bool] = Field(None, description="Flag indicating if the vendor is blacklisted")
    is_watchlist: Optional[bool] = Field(None, description="Flag indicating if the vendor is on the watchlist")


class VendorListResponse(BaseModel):
    """Response schema for listing vendor profiles."""
    total: int
    vendors: List[VendorResponse]
