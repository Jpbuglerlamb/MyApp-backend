# api/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time

from core.chat_orchestrator import chat_with_user
from core.auth_utils import get_current_user_id
from memory.store import get_user_state

from telemetry.logger import log_event

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
    actions: Optional[List[Dict[str, Any]]] = None,
    jobs: Optional[List[Dict[str, Any]]] = None,
    links: Optional[List[Dict[str, Any]]] = None,
    debug: Optional[Dict[str, Any]] = None,
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
        "VALIDATION": "I didn’t catch that. Tell me the role and location you want (e.g. “waiter in Edinburgh”).",
        "INTERNAL": "Something slipped on my end. Want to try again?",
    }
    return mapping.get(error_code, mapping["INTERNAL"])


def _msg_meta(msg: str) -> Dict[str, Any]:
    low = (msg or "").lower()
    return {
        "len": len(msg or ""),
        "has_digits": any(ch.isdigit() for ch in (msg or "")),
        "has_location_hint": (" in " in low) or (" near " in low) or (" based " in low),
        "has_income_hint": any(k in low for k in ("part", "full", "intern", "contract", "freelance")),
        "has_reflective_hint": any(k in low for k in ("confused", "lost", "stuck", "unsure", "future", "life", "career")),
    }


# ----------------------------
# Routes
# ----------------------------
@router.post("/chat")
async def chat(req: ChatRequest, user_id: str = Depends(get_current_user_id)):
    """
    Front door:
    - Always return a stable payload (protect Swift UI)
    - No product logic here (that lives in the orchestrator)
    - Telemetry is meta-only by default
    """
    uid = str(user_id)
    msg = (req.message or "").strip()
    t0 = time.time()

    log_event("chat_received", {"msg_meta": _msg_meta(msg)}, user_id=uid)

    # Empty message means "first load ping" from client.
    if not msg:
        out = _chat_response(
            "Hey 👋 Tell me the role and location you’re looking for (e.g. “backend developer in London”).",
            mode="chat",
            debug={"intent": "warm_start"},
        )
        log_event(
            "chat_responded",
            {"mode": out["mode"], "assistant_len": len(out["assistantText"]), "latency_ms": int((time.time() - t0) * 1000)},
            user_id=uid,
        )
        return out

    try:
        result = await chat_with_user(user_id=uid, user_message=msg)

        if not isinstance(result, dict):
            out = _chat_response(
                _friendly_error_message("INTERNAL"),
                debug={"error_code": "INTERNAL", "where": "chat_orchestrator_return_shape"},
            )
            log_event("chat_error_shape", {"type": str(type(result))}, user_id=uid)
            return out

        out = _chat_response(
            assistant_text=str(result.get("assistantText", "")),
            mode=str(result.get("mode", "chat")),
            actions=result.get("actions") or [],
            jobs=result.get("jobs") or [],
            links=result.get("links") or [],
            debug=result.get("debug") or {},
        )

        log_event(
            "chat_responded",
            {
                "mode": out["mode"],
                "actions_count": len(out["actions"]),
                "jobs_count": len(out["jobs"]),
                "latency_ms": int((time.time() - t0) * 1000),
                "debug_intent": (out.get("debug") or {}).get("intent"),
            },
            user_id=uid,
        )
        return out

    except TimeoutError:
        log_event("chat_error", {"error_code": "TIMEOUT"}, user_id=uid)
        return _chat_response(_friendly_error_message("TIMEOUT"), debug={"error_code": "TIMEOUT"})

    except ValueError:
        log_event("chat_error", {"error_code": "VALIDATION"}, user_id=uid)
        return _chat_response(_friendly_error_message("VALIDATION"), debug={"error_code": "VALIDATION"})

    except Exception as e:
        log_event("chat_error", {"error_code": "INTERNAL", "detail": repr(e)}, user_id=uid)
        return _chat_response(_friendly_error_message("INTERNAL"), debug={"error_code": "INTERNAL"})


@router.post("/deck")
async def get_deck(req: DeckRequest, user_id: str = Depends(get_current_user_id)):
    uid = str(user_id)
    state = get_user_state(uid)
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
    uid = str(user_id)
    state = get_user_state(uid)
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
    actions = [{"type": "YES_NO", "yesLabel": "Yes", "noLabel": "No", "yesValue": "talk_yes", "noValue": "talk_no"}]

    return _chat_response(
        assistant_text,
        mode="post_swipe",
        actions=actions,
        debug={"deckId": req.deckId},
    )
