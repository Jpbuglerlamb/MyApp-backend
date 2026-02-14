# api/deck.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth_utils import get_current_user_id
from memory.store import get_user_state, append_user_memory

router = APIRouter(tags=["deck"])

# ----------------------------
# Shared response shapes
# ----------------------------
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

class JobCard(BaseModel):
    id: str
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    redirect_url: Optional[str] = None
    score: Optional[int] = None
    reasons: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    action: Optional[str] = None

class APIResponse(BaseModel):
    assistantText: str = ""
    mode: str = "chat"
    actions: List[ActionItem] = Field(default_factory=list)
    jobs: List[JobCard] = Field(default_factory=list)
    links: List[LinkItem] = Field(default_factory=list)

# ----------------------------
# Requests
# ----------------------------
class DeckRequest(BaseModel):
    deckId: str

class SwipeSubmitRequest(BaseModel):
    deckId: str
    liked: List[str] = Field(default_factory=list)
    passed: List[str] = Field(default_factory=list)

# ----------------------------
# Routes
# ----------------------------
@router.post("/deck", response_model=APIResponse)
async def get_deck(
    req: DeckRequest,
    user_id: str = Depends(get_current_user_id),
) -> APIResponse:
    state: Dict[str, Any] = get_user_state(str(conversation_id))
    deck = state.get("current_deck") or {}
    deck_id = deck.get("deck_id")

    if not deck_id or deck_id != req.deckId:
        raise HTTPException(status_code=404, detail="Deck not found")

    cached = state.get("cached_jobs") or []
    job_ids = set(deck.get("job_ids") or [])

    # Return only the cards for this deck, in the same order
    by_id = {c.get("id"): c for c in cached if c.get("id")}
    ordered_cards = [by_id[jid] for jid in (deck.get("job_ids") or []) if jid in by_id]

    return APIResponse(
        assistantText="",
        mode="deck",
        jobs=ordered_cards,
    )

@router.post("/swipe/submit", response_model=APIResponse)
async def submit_swipes(
    req: SwipeSubmitRequest,
    user_id: str = Depends(get_current_user_id),
) -> APIResponse:
    state: Dict[str, Any] = get_user_state(str(conversation_id))
    deck = state.get("current_deck") or {}
    deck_id = deck.get("deck_id")

    if not deck_id or deck_id != req.deckId:
        raise HTTPException(status_code=404, detail="Deck not found")

    # Persist to state
    deck["liked"] = list(dict.fromkeys(req.liked))
    deck["passed"] = list(dict.fromkeys(req.passed))
    deck["complete"] = True
    state["current_deck"] = deck

    # Optional: nudge the orchestrator state machine
    state["phase"] = "post_swipe"

    # Optional: store a tiny memory breadcrumb
    append_user_memory(str(user_id), "system", f"Deck complete. liked={len(req.liked)} passed={len(req.passed)}")

    return APIResponse(
        assistantText="Nice. Want to talk about the ones you liked?",
        mode="post_swipe",
        actions=[
            ActionItem(
                type="YES_NO",
                yesLabel="Yes",
                yesValue="talk_yes",
                noLabel="No",
                noValue="talk_no",
            )
        ],
        jobs=[],
        links=[],
    )
