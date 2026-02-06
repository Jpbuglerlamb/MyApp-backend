# api/auth.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.user import User
from core.database import get_async_session
from core.auth_utils import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])  # <-- must be at top-level

# -------------------------------
# Request schemas
# -------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# -------------------------------
# Signup endpoint
# -------------------------------
@router.post("/signup")
async def signup(req: SignupRequest, session: AsyncSession = Depends(get_async_session)):
    # Check if user exists
    result = await session.execute(select(User).where(User.email == req.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Create user
    user = User(
        email=req.email,
        hashed_password=hash_password(req.password)
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    token = create_access_token(str(user.id))
    return {"userId": user.id, "token": token}

# -------------------------------
# Login endpoint
# -------------------------------
@router.post("/login")
async def login(req: LoginRequest, session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(User).where(User.email == req.email))
    user = result.scalars().first()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user.id))
    return {"userId": user.id, "token": token}