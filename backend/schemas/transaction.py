"""
backend/schemas/transaction.py
─────────────────────────────────────────────────────────────────
FastAPI request and response schemas using Pydantic.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    # ERP context fields (for display/logging)
    vendor_id: Optional[str] = Field(default="V00001", json_schema_extra={"example": "V00123"})
    department: Optional[str] = Field(default="Finance", json_schema_extra={"example": "Finance"})
    approved_by: Optional[str] = Field(default="mgr_01", json_schema_extra={"example": "mgr_01"})

    # Core transaction fields
    posting_time: float = Field(..., json_schema_extra={"example": 3600.0}, description="Seconds since first transaction")
    transaction_amount: float = Field(..., json_schema_extra={"example": 250.00}, description="Transaction amount in currency")

    # PCA features V1–V28 (from dataset)
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float


class TransactionResponse(BaseModel):
    vendor_id: str
    department: str
    approved_by: str
    transaction_amount: float
    anomaly_score: float
    is_fraud: bool
    risk_level: str
    alert_message: str
    top_risk_factors: list


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchResponseSummary(BaseModel):
    total: int
    flagged: int
    flag_rate: str
    predictions: List[TransactionResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    features: int
