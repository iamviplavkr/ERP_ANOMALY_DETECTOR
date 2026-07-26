"""
backend/api/v1/analytics.py
─────────────────────────────────────────────────────────────────
Protected analytics and BI endpoints.

RBAC:
  Admin, Fraud Analyst, Auditor — read-only access (200)
  Finance User                  — no access (403)
"""

import csv
from datetime import datetime, timedelta, timezone
import io
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import status as http_status
from sqlalchemy.orm import Session

from backend.auth.auth_dependencies import (
    RoleChecker,
    get_analytics_repository,
    get_audit_log_repository,
    get_db,
)
from backend.repositories.analytics_repository import AnalyticsRepositoryInterface
from backend.repositories.audit_log_repository import AuditLogRepositoryInterface
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics")

# RBAC Guard for Analytics (accessible to Admin, Fraud Analyst, and Auditor)
analytics_rbac = RoleChecker(["Admin", "Fraud Analyst", "Auditor"])
# Debug counts endpoint: Admin only
admin_rbac = RoleChecker(["Admin"])


def _parse_dates(start_date: Optional[str], end_date: Optional[str]) -> tuple[datetime, datetime]:
    """Helper to parse query dates or fall back to last 30 days."""
    try:
        if end_date:
            parsed_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        else:
            parsed_end = datetime.now(timezone.utc).replace(second=59, microsecond=999999)

        if start_date:
            parsed_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        else:
            parsed_start = (parsed_end - timedelta(days=30)).replace(second=0, microsecond=0)

        # Ensure start <= end
        if parsed_start > parsed_end:
            parsed_start = parsed_end - timedelta(days=30)

        return parsed_start, parsed_end
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use ISO format (e.g. YYYY-MM-DDThh:mm:ss): {exc}",
        )


@router.get("/debug-counts", tags=["Analytics"], include_in_schema=True)
def get_debug_counts(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(admin_rbac),
    db: Session = Depends(get_db),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """
    Admin-only diagnostic endpoint.
    Returns raw PostgreSQL row counts (bypassing analytics cache) and the
    analytics service KPI result for the same date range so you can confirm
    whether any cache staleness is causing discrepancies.
    """
    from sqlalchemy import text
    from backend.services.analytics_service import _ANALYTICS_CACHE

    s_date, e_date = _parse_dates(start_date, end_date)

    # Raw counts directly from DB (no cache)
    raw = {}
    for tbl in ("transactions", "predictions", "alerts", "vendors", "audit_logs"):
        try:
            raw[tbl] = db.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        except Exception as exc:
            raw[tbl] = f"error: {exc}"

    # KPI via service (uses cache if TTL > 0)
    service = AnalyticsService(analytics_repo)
    kpis = service.get_kpis(s_date, e_date)

    return {
        "raw_db_counts_all_time": raw,
        "analytics_kpis_for_range": kpis,
        "cache_entries_active": len(_ANALYTICS_CACHE),
        "date_range": {"start": s_date.isoformat(), "end": e_date.isoformat()},
    }


@router.get("/overview", tags=["Analytics"])
def get_overview(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get high-level summary KPIs for transactions, anomalies, alerts, and vendors."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_kpis(s_date, e_date)


@router.get("/trends", tags=["Analytics"])
def get_trends(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get daily transaction volumes, averages, and fraud flags over time."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_fraud_trends(s_date, e_date)


@router.get("/risk-distribution", tags=["Analytics"])
def get_risk_distribution(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get transaction counts and volume sums by risk severity level."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_risk_distribution(s_date, e_date)


@router.get("/departments", tags=["Analytics"])
def get_departments(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get activity, amount, and average risk metrics per department."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_department_metrics(s_date, e_date)


@router.get("/vendors", tags=["Analytics"])
def get_vendors(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    sort_by: str = "reputation_score",
    sort_order: str = "asc",
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get paginated vendor threat rankings sorted by key metrics."""
    s_date, e_date = _parse_dates(start_date, end_date)
    if sort_by not in [
        "reputation_score",
        "historical_alerts_count",
        "total_transactions_count",
        "historical_fraud_rate",
        "name",
        "vendor_id",
    ]:
        sort_by = "reputation_score"
    if sort_order.lower() not in ["asc", "desc"]:
        sort_order = "asc"

    service = AnalyticsService(analytics_repo)
    return service.get_vendor_rankings(
        s_date, e_date, limit=limit, offset=offset, sort_by=sort_by, sort_order=sort_order
    )


@router.get("/alerts-lifecycle", tags=["Analytics"])
def get_alerts_lifecycle(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get active vs resolved/dismissed alert distribution counts."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_alert_lifecycle(s_date, e_date)


@router.get("/model-performance", tags=["Analytics"])
def get_model_performance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
):
    """Get prediction distributions and max scores for ML performance checks."""
    s_date, e_date = _parse_dates(start_date, end_date)
    service = AnalyticsService(analytics_repo)
    return service.get_model_performance(s_date, e_date)


@router.get("/export", tags=["Analytics"])
def export_analytics(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    format: str = "csv",
    current_user: dict = Depends(analytics_rbac),
    analytics_repo: AnalyticsRepositoryInterface = Depends(get_analytics_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    db: Session = Depends(get_db),
):
    """
    Export combined predictions and transaction data in CSV format.
    Generates an audit log entry for security audit compliance.
    """
    s_date, e_date = _parse_dates(start_date, end_date)
    if sort_by not in ["created_at", "transaction_amount", "anomaly_score"]:
        sort_by = "created_at"
    if sort_order.lower() not in ["asc", "desc"]:
        sort_order = "desc"

    service = AnalyticsService(analytics_repo)
    records = service.get_predictions_export(s_date, e_date, sort_by=sort_by, sort_order=sort_order)

    # ── Write Audit Log (Mandatory entry for security) ───────────────────────
    if audit_log_repo:
        audit_log_repo.create(
            {
                "user_id": current_user.get("id"),
                "action": "ANALYTICS_EXPORT",
                "resource": "/v1/analytics/export",
                "details": {
                    "start_date": s_date.isoformat(),
                    "end_date": e_date.isoformat(),
                    "records_count": len(records),
                    "exported_by": current_user.get("username"),
                    "format": format,
                },
                "ip_address": request.client.host if request.client else None,
            }
        )
        if db:
            try:
                db.commit()
            except Exception:
                pass

    if format.lower() == "json":
        return records

    # Generate CSV stream response
    output = io.StringIO()
    writer = csv.writer(output)

    # Header columns
    headers = [
        "prediction_id",
        "transaction_id",
        "vendor_id",
        "department",
        "approved_by",
        "transaction_amount",
        "anomaly_score",
        "is_fraud",
        "risk_level",
        "alert_message",
        "model_version",
        "created_at",
    ]
    writer.writerow(headers)

    for r in records:
        writer.writerow([r.get(col) for col in headers])

    content = output.getvalue()
    output.close()

    filename = f"analytics_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
