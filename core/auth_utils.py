# core/auth_utils.py
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import HTTPException, Header
from passlib.context import CryptContext
from jwt import PyJWTError
import os
# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError("Missing SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour

# -------------------------------------------------------------------
# Use Argon2 instead of bcrypt
# -------------------------------------------------------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# -------------------------------------------------------------------
# Password hashing
# -------------------------------------------------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -------------------------------------------------------------------
# JWT token creation
# -------------------------------------------------------------------
def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": user_id}
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------------------------------------------------
# JWT token verification
# -------------------------------------------------------------------
def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------------------------------------------
# FastAPI dependency
# -------------------------------------------------------------------
def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[len("Bearer "):]
    return decode_access_token(token)