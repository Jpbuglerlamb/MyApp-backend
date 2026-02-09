# core/auth_utils.py
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWTError
from passlib.context import CryptContext

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))

# Refresh token settings
REFRESH_TOKEN_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "30"))
# Pepper is a server secret mixed into refresh token hashing
REFRESH_TOKEN_PEPPER = os.getenv("REFRESH_TOKEN_PEPPER", "").strip()

# Read at import time (expects .env to be loaded before importing this module)
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
if not SECRET_KEY:
    raise RuntimeError("Missing SECRET_KEY environment variable")

if not REFRESH_TOKEN_PEPPER:
    # You CAN run without this, but you shouldn't if you're serious.
    raise RuntimeError("Missing REFRESH_TOKEN_PEPPER environment variable")

# Optional: rotate tokens if you ever need “log out everywhere”
# Bump this value in DB for a user to invalidate all access tokens if you store `ver` claim.
USE_TOKEN_VERSION = os.getenv("USE_TOKEN_VERSION", "0").strip() == "1"

# -------------------------------------------------------------------
# Password hashing (Argon2)
# -------------------------------------------------------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -------------------------------------------------------------------
# JWT helpers (Access token)
# -------------------------------------------------------------------
def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
    token_version: Optional[int] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    # Optional token versioning hook
    if USE_TOKEN_VERSION:
        payload["ver"] = int(token_version or 0)

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------------------------------------------
# Refresh token helpers (Random + hashed)
# -------------------------------------------------------------------
def new_refresh_token_raw() -> str:
    # urlsafe, high entropy. Store only the hash server-side.
    return secrets.token_urlsafe(32)

def hash_refresh_token(raw: str) -> str:
    # Hash with a pepper so DB leak doesn't allow offline guessing.
    data = (raw + REFRESH_TOKEN_PEPPER).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def refresh_expires_at() -> datetime:
    # Stored as naive UTC or aware UTC depending on your DB style.
    # We'll use aware UTC consistently.
    return datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_DAYS)

# -------------------------------------------------------------------
# FastAPI dependency
# -------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user_id(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Returns authenticated user id (JWT 'sub').

    - No Authorization header -> 401
    - Not Bearer -> 401
    - Invalid/expired -> 401
    """
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    payload = decode_access_token(creds.credentials)
    return str(payload["sub"])


