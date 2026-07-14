"""
backend/api/v1/auth.py
─────────────────────────────────────────────────────────────────
Authentication routes for logging in, token refreshing, logout,
and profile verification.
"""

from fastapi import APIRouter, Depends

from backend.schemas.user import (
    UserLoginRequest,
    TokenResponse,
    TokenRefreshRequest,
    UserResponse,
)
from backend.services.auth_service import AuthService
from backend.auth.auth_dependencies import get_current_user, get_auth_service

router = APIRouter(prefix="/auth")


@router.post("/login", response_model=TokenResponse, tags=["Authentication"])
def login(payload: UserLoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    """
    Authenticate username and password, issuing access and refresh JWT tokens.
    """
    user = auth_service.authenticate_user(payload.username, payload.password)
    access_token = auth_service.create_access_token(user["username"], user["role"])
    refresh_token = auth_service.create_refresh_token(user["username"])

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        username=user["username"],
        role=user["role"]
    )


@router.post("/refresh", response_model=TokenResponse, tags=["Authentication"])
def refresh(payload: TokenRefreshRequest, auth_service: AuthService = Depends(get_auth_service)):
    """
    Renew a short-lived access token using a valid refresh token.
    """
    res = auth_service.refresh_access_token(payload.refresh_token)
    # Re-sign a fresh refresh token as well for sliding expiration security
    user_payload = auth_service.decode_token(payload.refresh_token, is_refresh=True)
    username = user_payload.get("sub")
    new_refresh = auth_service.create_refresh_token(username)

    return TokenResponse(
        access_token=res["access_token"],
        refresh_token=new_refresh,
        username=username,
        role=res["role"]
    )


@router.post("/logout", tags=["Authentication"])
def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout endpoint. Since we use stateless JWT tokens without a server-side
    blacklist, actual token invalidation is handled client-side by discarding
    stored tokens. This endpoint confirms the user was authenticated and
    signals a successful logout for frontend session cleanup.
    """
    return {
        "message": "Successfully logged out.",
        "username": current_user["username"]
    }


@router.get("/me", response_model=UserResponse, tags=["Authentication"])
def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Retrieve details of the currently authenticated user.
    """
    return UserResponse(
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        is_active=current_user.get("is_active", True)
    )
