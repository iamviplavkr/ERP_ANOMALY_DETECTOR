"""
backend/schemas/alert.py
─────────────────────────────────────────────────────────────────
Pydantic schemas for alert request/response payloads.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlertResponse(BaseModel):
    """Response schema for a single alert."""
    id: str
    prediction_id: str
    risk_level: str
    status: str
    rules_triggered: List[str]
    mitigation_action: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    prediction: Optional[Dict[str, Any]] = None


class AlertListResponse(BaseModel):
    """Response schema for paginated alert list."""
    total: int
    alerts: List[AlertResponse]


class AlertStatusUpdateRequest(BaseModel):
    """Request schema for updating alert status."""
    status: str = Field(
        ...,
        description="New status. Valid transitions: OPEN→INVESTIGATING, INVESTIGATING→RESOLVED or DISMISSED"
    )
