# api/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from core.chat_orchestrator import chat_with_user
from core.auth_utils import get_current_user_id
from memory.store import get_user_state

router = APIRouter()


# ----------------------------
# Request models
# ----------------------------
class ChatRequest(BaseModel):
    message: str


class DeckRequest(BaseModel):
    deckId: str


class SwipeSubmitRequest(BaseModel):
    deckId: str
    liked: List[str] = Field(default_factory=list)
    passed: List[str] = Field(default_factory=list)


# ----------------------------
# Helpers: consistent responses
# ----------------------------
def _chat_response(
    assistant_text: str,
    mode: str = "chat",
    actions: List[Dict[str, Any]] | None = None,
    jobs: List[Dict[str, Any]] | None = None,
    links: List[Dict[str, Any]] | None = None,
    debug: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "assistantText": assistant_text,
        "mode": mode,
        "actions": actions or [],
        "jobs": jobs or [],
        "links": links or [],
        "debug": debug or {},
    }


def _friendly_error_message(error_code: str) -> str:
    mapping = {
        "TIMEOUT": "That took too long on my end. Want to try again?",
        "UPSTREAM": "The job source is being slow right now. Try again in a moment?",
        "VALIDATION": "I didn’t catch that fully. Tell me the role and location you want.",
        "INTERNAL": "Something slipped on my end. Try that again?",
    }
    return mapping.get(error_code, mapping["INTERNAL"])


# ----------------------------
# Routes
# ----------------------------
@router.post("/chat")
async def chat(req: ChatRequest, user_id: str = Depends(get_current_user_id)):
    """
    Front door:
    - Keep this file dumb.
    - Always return 200-style payload to protect the Swift UI.
    - Product logic lives in the orchestrator.
    """
    msg = (req.message or "").strip()

    # If client ever pings with empty (shouldn't happen now), return a safe prompt.
    if not msg:
        return _chat_response("Tell me the role and location you’re looking for.", mode="chat")

    try:
        result = await chat_with_user(user_id=str(user_id), user_message=msg)

        if not isinstance(result, dict):
            return _chat_response(
                _friendly_error_message("INTERNAL"),
                debug={"error_code": "INTERNAL", "where": "chat_orchestrator_return_shape"},
            )

        return _chat_response(
            assistant_text=str(result.get("assistantText", "")),
            mode=str(result.get("mode", "chat")),
            actions=result.get("actions") or [],
            jobs=result.get("jobs") or [],
            links=result.get("links") or [],
            debug=result.get("debug") or {},
        )

    except TimeoutError:
        return _chat_response(_friendly_error_message("TIMEOUT"), debug={"error_code": "TIMEOUT"})

    except ValueError:
        return _chat_response(_friendly_error_message("VALIDATION"), debug={"error_code": "VALIDATION"})

    except Exception as e:
        return _chat_response(
            _friendly_error_message("INTERNAL"),
            debug={"error_code": "INTERNAL", "detail": repr(e)},
        )


@router.post("/deck")
async def get_deck(req: DeckRequest, user_id: str = Depends(get_current_user_id)):
    state = get_user_state(str(user_id))
    deck = (state.get("current_deck") or {})

    if deck.get("deck_id") != req.deckId:
        return _chat_response(
            "That swipe deck expired. Want me to search again?",
            mode="chat",
            debug={"deckId": req.deckId, "error_code": "DECK_EXPIRED"},
        )

    all_jobs = state.get("cached_jobs") or []
    job_ids = set(deck.get("job_ids") or [])
    deck_jobs = [j for j in all_jobs if j.get("id") in job_ids]

    return _chat_response(
        assistant_text="",
        mode="swipe_deck",
        jobs=deck_jobs,
        debug={"deckId": req.deckId},
    )


@router.post("/swipe/submit")
async def submit_swipes(req: SwipeSubmitRequest, user_id: str = Depends(get_current_user_id)):
    state = get_user_state(str(user_id))
    deck = (state.get("current_deck") or {})

    if deck.get("deck_id") != req.deckId:
        return _chat_response(
            "That swipe deck expired. Want me to search again?",
            mode="chat",
            debug={"deckId": req.deckId, "error_code": "DECK_EXPIRED"},
        )

    deck["liked"] = req.liked
    deck["passed"] = req.passed
    deck["complete"] = True
    state["phase"] = "post_swipe"

    total = len(deck.get("job_ids") or [])
    assistant_text = f"Nice. You liked {len(req.liked)} out of {total}. Want to talk about the ones you liked?"
    actions = [{"type": "YES_NO", "yesValue": "talk_yes", "noValue": "talk_no"}]

    return _chat_response(
        assistant_text,
        mode="post_swipe",
        actions=actions,
        debug={"deckId": req.deckId},
    )
