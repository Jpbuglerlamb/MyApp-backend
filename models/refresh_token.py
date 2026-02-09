# models/refresh_token.py
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Index
from sqlalchemy.orm import relationship

from core.database import Base

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Store ONLY a hash of the refresh token (never the raw token)
    token_hash = Column(String(128), nullable=False, unique=True, index=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)

    user = relationship("User", backref="refresh_tokens")

Index("ix_refresh_tokens_user_active", RefreshToken.user_id, RefreshToken.revoked_at)
