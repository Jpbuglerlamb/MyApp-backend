# core/auth_utils.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from jwt import PyJWTError

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour

if not SECRET_KEY:
    # For production this MUST be set; failing fast prevents insecure tokens.
    raise RuntimeError(
        "Missing SECRET_KEY environment variable. "
        "Set it in your server environment (e.g., Render → Service → Environment)."
    )

# -------------------------------------------------------------------
# Password hashing (Argon2)
# -------------------------------------------------------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -------------------------------------------------------------------
# JWT token creation
# -------------------------------------------------------------------
def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------------------------------------------------
# JWT token verification
# -------------------------------------------------------------------
def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token (missing subject)")
        return str(user_id)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------------------------------------------
# FastAPI dependency (Bearer token)
# -------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user_id(
    creds: Optional[HTTPAuthorizationCredentials] = bearer_scheme,
) -> str:
    if creds is None:
        # Clean 401 instead of FastAPI 422 “missing header”
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme (expected Bearer)")

    return decode_access_token(creds.credentials)
