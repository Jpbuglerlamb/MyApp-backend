# api/chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth_utils import get_current_user_id
from core.chat_orchestrator import chat_with_user, WELCOME_TEXT

router = APIRouter()

# --------- Models (match iOS decoding) ---------

class ChatRequest(BaseModel):
    message: Optional[str] = None
    conversation_id: str

class ActionItem(BaseModel):
    type: str
    label: Optional[str] = None
    deckId: Optional[str] = None
    yesLabel: Optional[str] = None
    yesValue: Optional[str] = None
    noLabel: Optional[str] = None
    noValue: Optional[str] = None

class LinkItem(BaseModel):
    label: str
    url: str

class ChatResponse(BaseModel):
    assistantText: str
    actions: List[ActionItem] = Field(default_factory=list)
    links: List[LinkItem] = Field(default_factory=list)

# --------- Route ---------

@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user_id: str = Depends(get_current_user_id),
) -> ChatResponse:
    # Welcome ping: treat empty message as “give welcome text”
    msg = (req.message or "").strip()
    if msg == "":
        return ChatResponse(
            assistantText=WELCOME_TEXT,
            actions=[],
            links=[],
        )

    try:
        # Expect your orchestrator to return a dict-like payload
        result: Dict[str, Any] = await chat_with_user(
            user_id=user_id, 
            conversation_id=req.conversation_id,
            user_message=msg)
        
        assistant_text = (result.get("assistantText") or result.get("assistant_text") or "").strip()
        actions = result.get("actions") or []
        links = result.get("links") or []

        # Hard guarantees: never None
        if not isinstance(actions, list):
            actions = []
        if not isinstance(links, list):
            links = []

        return ChatResponse(
            assistantText=assistant_text,
            actions=actions,
            links=links,
        )

    except HTTPException:
        raise

    except Exception as e:
        # Keep server response JSON-shaped and predictable for iOS
        raise HTTPException(status_code=500, detail=f"Chat error: {type(e).__name__}")
