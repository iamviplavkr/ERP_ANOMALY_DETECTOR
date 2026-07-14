"""
backend/schemas/user.py
─────────────────────────────────────────────────────────────────
Pydantic schemas for User representations and token exchanges.
"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    username: str = Field(..., json_schema_extra={"example": "john_doe"})
    email: EmailStr = Field(..., json_schema_extra={"example": "john@example.com"})
    role: str = Field(..., json_schema_extra={"example": "Finance User"})
    is_active: Optional[bool] = True


class UserCreate(UserBase):
    password: str = Field(..., json_schema_extra={"example": "secure_password"})


class UserResponse(UserBase):
    pass


class UserLoginRequest(BaseModel):
    username: str = Field(..., json_schema_extra={"example": "admin"})
    password: str = Field(..., json_schema_extra={"example": "password123"})


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    username: str
    role: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str
