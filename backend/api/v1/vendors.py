"""
backend/api/v1/vendors.py
─────────────────────────────────────────────────────────────────
Protected vendor directory management endpoints.

RBAC:
  Admin, Fraud Analyst — full access (list, view, create, update, delete)
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
    get_vendor_repository,
    get_audit_log_repository,
    get_db,
)
from backend.repositories.vendor_repository import VendorRepositoryInterface
from backend.repositories.audit_log_repository import AuditLogRepositoryInterface
from backend.schemas.vendor import (
    VendorResponse,
    VendorCreateRequest,
    VendorUpdateRequest,
    VendorListResponse,
)

router = APIRouter(prefix="/vendors")

# RBAC guards
_read_roles = RoleChecker(["Admin", "Fraud Analyst", "Auditor"])
_write_roles = RoleChecker(["Admin", "Fraud Analyst"])


@router.get("", response_model=VendorListResponse, tags=["Vendors"])
def list_vendors(
    is_blacklisted: Optional[bool] = None,
    is_watchlist: Optional[bool] = None,
    _user: dict = Depends(_read_roles),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
):
    """
    List all vendor profiles with optional filters.
    Accessible by Admin, Fraud Analyst, and Auditor.
    """
    vendors = vendor_repo.list_all(is_blacklisted=is_blacklisted, is_watchlist=is_watchlist)
    return VendorListResponse(total=len(vendors), vendors=vendors)


@router.get("/{vendor_id}", response_model=VendorResponse, tags=["Vendors"])
def get_vendor(
    vendor_id: str,
    _user: dict = Depends(_read_roles),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
):
    """
    Get a specific vendor profile by vendor_id.
    Accessible by Admin, Fraud Analyst, and Auditor.
    """
    vendor = vendor_repo.get_by_vendor_id(vendor_id)
    if not vendor:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Vendor '{vendor_id}' not found.",
        )
    return vendor


@router.post("", response_model=VendorResponse, status_code=http_status.HTTP_201_CREATED, tags=["Vendors"])
def create_vendor(
    payload: VendorCreateRequest,
    request: Request,
    current_user: dict = Depends(_write_roles),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    db: Session = Depends(get_db),
):
    """
    Create a new vendor profile.
    Accessible by Admin and Fraud Analyst only.
    """
    try:
        # Check if already exists
        existing = vendor_repo.get_by_vendor_id(payload.vendor_id)
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Vendor '{payload.vendor_id}' already exists.",
            )

        vendor = vendor_repo.create(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # Audit log creation
    if audit_log_repo:
        audit_log_repo.create({
            "user_id": current_user.get("id"),
            "action": "VENDOR_CREATE",
            "resource": "/v1/vendors",
            "details": {
                "vendor_id": payload.vendor_id,
                "name": payload.name,
                "created_by": current_user.get("username"),
            },
            "ip_address": request.client.host if request.client else None,
        })
        if db:
            try:
                db.commit()
            except Exception:
                pass

    return vendor


@router.put("/{vendor_id}", response_model=VendorResponse, tags=["Vendors"])
def update_vendor(
    vendor_id: str,
    payload: VendorUpdateRequest,
    request: Request,
    current_user: dict = Depends(_write_roles),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    db: Session = Depends(get_db),
):
    """
    Update vendor risk configuration, name, or reputation score.
    Accessible by Admin and Fraud Analyst only.
    """
    # Exclude unset fields from the updates dictionary
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        # If no updates provided, just return existing
        existing = vendor_repo.get_by_vendor_id(vendor_id)
        if not existing:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Vendor '{vendor_id}' not found.",
            )
        return existing

    try:
        updated = vendor_repo.update(vendor_id, updates)
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    # Audit log update
    if audit_log_repo:
        audit_log_repo.create({
            "user_id": current_user.get("id"),
            "action": "VENDOR_UPDATE",
            "resource": f"/v1/vendors/{vendor_id}",
            "details": {
                "vendor_id": vendor_id,
                "updates": updates,
                "updated_by": current_user.get("username"),
            },
            "ip_address": request.client.host if request.client else None,
        })
        if db:
            try:
                db.commit()
            except Exception:
                pass

    return updated


@router.delete("/{vendor_id}", status_code=http_status.HTTP_204_NO_CONTENT, tags=["Vendors"])
def delete_vendor(
    vendor_id: str,
    request: Request,
    current_user: dict = Depends(_write_roles),
    vendor_repo: VendorRepositoryInterface = Depends(get_vendor_repository),
    audit_log_repo: AuditLogRepositoryInterface = Depends(get_audit_log_repository),
    db: Session = Depends(get_db),
):
    """
    Delete a vendor profile.
    Accessible by Admin and Fraud Analyst only.
    """
    deleted = vendor_repo.delete(vendor_id)
    if not deleted:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Vendor '{vendor_id}' not found.",
        )

    # Audit log deletion
    if audit_log_repo:
        audit_log_repo.create({
            "user_id": current_user.get("id"),
            "action": "VENDOR_DELETE",
            "resource": f"/v1/vendors/{vendor_id}",
            "details": {
                "vendor_id": vendor_id,
                "deleted_by": current_user.get("username"),
            },
            "ip_address": request.client.host if request.client else None,
        })
        if db:
            try:
                db.commit()
            except Exception:
                pass

    return
