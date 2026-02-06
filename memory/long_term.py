# app/memory/long_term.py
from core.database import get_async_session
from models.chat_history import ChatMessage
from sqlalchemy.ext.asyncio import AsyncSession

async def save_chat_message(user_id: int, role: str, content: str, session: AsyncSession):
    msg = ChatMessage(user_id=user_id, role=role, content=content)
    session.add(msg)
    await session.commit()