"""
backend/api/v1/alerts.py
─────────────────────────────────────────────────────────────────
Protected alert management endpoints.

RBAC:
  Admin, Fraud Analyst — full access (list, view, update status)
  Auditor             — read-only (list, view)
  Finance User        — no access (403)
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import status as http_status
from sqlalchemy.orm import Session
from typing import Optional

from backend.auth.auth_dependencies import (
    RoleChecker,
    get_current_user,
    get_alert_repository,
    get_audit_log_repository,
    get_db,
)
from backend.repositories.alert_repository import AlertRepositoryInterface
from backend.repositories.audit_log_repository import AuditLogRepositoryInterface
from backend.schemas.alert import AlertResponse, AlertListResponse, AlertStatusUpdateRequest

router = APIRouter(prefix="/alerts")

# RBAC guards
_read_roles = RoleChecker(["Admin", "Fraud Analyst", "Auditor"])
_write_roles = RoleChecker(["Admin", "Fraud Analyst"])


@router.get("", response_model=AlertListResponse, tags=["Alerts"])
def list_alerts(
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    _user: dict = Depends(_read_roles),
    alert_repo: AlertRepositoryInterface = Depends(get_alert_repository),
):
    """
    List all alerts with optional filters.
    Accessible by Admin, Fraud Analyst, and Auditor.
    """
    alerts = alert_repo.list_all(status=status, risk_level=risk_level)
    return AlertListResponse(total=len(alerts), alerts=alerts)


@router.get("/{alert_id}", response_model=AlertResponse, tags=["Alerts"])
def get_alert(
    alert_id: str,
    _user: dict = Depends(_read_roles),
    alert_repo: AlertRepositoryInterface = Depends(get_alert_repository),
):
    """
    Get a specific alert by ID including linked prediction and transaction details.
    Accessible by Admin, Fraud Analyst, and Auditor.
    """
    alert = alert_repo.get_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return alert


@router.put("/{alert_id}/status", response_model=AlertResponse, tags=["Alerts"])
def update_alert_status(
    alert_id: str,
    payload: AlertStatusUpdateRequest,
    request: Request,
    current_user: dict = Depends(_write_roles),
    alert_repo: AlertRepositoryInterface = Depends(get_alert_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    db: Session = Depends(get_db),
):
    """
    Update alert lifecycle status.
    Valid transitions: OPEN → INVESTIGATING → RESOLVED or DISMISSED.
    Accessible by Admin and Fraud Analyst only.
    """
    try:
        updated = alert_repo.update_status(alert_id, payload.status)
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    # Audit the status change
    if audit_log_repo:
        audit_log_repo.create({
            "user_id": current_user.get("id"),
            "action": "ALERT_STATUS_CHANGE",
            "resource": f"/v1/alerts/{alert_id}/status",
            "details": {
                "alert_id": alert_id,
                "new_status": payload.status,
                "username": current_user.get("username"),
            },
            "ip_address": request.client.host if request.client else None,
        })
        if db:
            try:
                db.commit()
            except Exception:
                pass

    return updated
