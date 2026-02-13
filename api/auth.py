# api/auth.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.user import User
from models.refresh_token import RefreshToken
from core.database import get_async_session
from core.auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    new_refresh_token_raw,
    hash_refresh_token,
    refresh_expires_at,
)
from core.rate_limit import TokenBucketLimiter

login_limiter = TokenBucketLimiter(rate=5, per_seconds=60, capacity=10)      # 5/min, burst 10
refresh_limiter = TokenBucketLimiter(rate=10, per_seconds=60, capacity=20)   # 10/min, burst 20

router = APIRouter(prefix="/auth", tags=["auth"])


# -------------------------------
# Schemas
# -------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class RefreshRequest(BaseModel):
    refreshToken: str = Field(min_length=20, max_length=300)

class LogoutRequest(BaseModel):
    refreshToken: str = Field(min_length=20, max_length=300)

class AuthResponse(BaseModel):
    userId: int
    accessToken: str
    refreshToken: str


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


async def _issue_refresh_token(session: AsyncSession, user_id: int) -> str:
    raw = new_refresh_token_raw()
    token_hash = hash_refresh_token(raw)

    rt = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=refresh_expires_at(),
    )
    session.add(rt)
    await session.commit()
    return raw


# -------------------------------
# Signup
# -------------------------------
@router.post("/signup", response_model=AuthResponse)

async def signup(req: SignupRequest, session: AsyncSession = Depends(get_async_session)):
    email = _norm_email(req.email)
    print("BACKEND emial repr:", repr(req.email))
    print("BACKEND email:", req.email)
    
    result = await session.execute(select(User).where(User.email == email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(email=email, hashed_password=hash_password(req.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)

    access = create_access_token(str(user.id))
    refresh = await _issue_refresh_token(session, user.id)
    return {"userId": user.id, "accessToken": access, "refreshToken": refresh}
   


# -------------------------------
# Login (FIXED)
# -------------------------------
@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest, request: Request, session: AsyncSession = Depends(get_async_session)):
    ip = request.client.host if request.client else "unknown"
    key = f"login:{ip}"
    if not login_limiter.allow(key):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again shortly.")

    email = _norm_email(req.email)
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalars().first()

    # Generic failure message (avoid account enumeration)
    if (not user) or (not verify_password(req.password, user.hashed_password)):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(str(user.id))
    refresh = await _issue_refresh_token(session, user.id)
    return {"userId": user.id, "accessToken": access, "refreshToken": refresh}


# -------------------------------
# Refresh (ROTATION)
# -------------------------------
@router.post("/refresh", response_model=AuthResponse)
async def refresh(req: RefreshRequest, request: Request, session: AsyncSession = Depends(get_async_session)):
    ip = request.client.host if request.client else "unknown"
    key = f"refresh:{ip}"
    if not refresh_limiter.allow(key):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again shortly.")

    token_hash = hash_refresh_token(req.refreshToken)

    result = await session.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    rt = result.scalars().first()

    # Use aware UTC to match refresh_expires_at() in auth_utils.py
    now = datetime.now(timezone.utc)

    if (not rt) or (rt.revoked_at is not None) or (rt.expires_at <= now):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Rotate: revoke old token, issue a new one
    rt.revoked_at = now
    rt.last_used_at = now
    await session.commit()

    new_refresh = await _issue_refresh_token(session, rt.user_id)
    new_access = create_access_token(str(rt.user_id))
    return {"userId": rt.user_id, "accessToken": new_access, "refreshToken": new_refresh}


# -------------------------------
# Logout (revoke refresh token)
# -------------------------------
@router.post("/logout")
async def logout(req: LogoutRequest, session: AsyncSession = Depends(get_async_session)):
    token_hash = hash_refresh_token(req.refreshToken)

    result = await session.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    rt = result.scalars().first()

    if rt and rt.revoked_at is None:
        rt.revoked_at = datetime.now(timezone.utc)
        await session.commit()

    return {"ok": True}

