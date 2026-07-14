"""
backend/auth/api_key.py
─────────────────────────────────────────────────────────────────
Optional API key authentication dependency for endpoints.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from backend.core.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Checks if API key authentication is enabled, and if so, verifies the provided key.
    """
    if not settings.REQUIRE_API_KEY:
        return api_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API Key."
        )

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key."
        )

    return api_key
