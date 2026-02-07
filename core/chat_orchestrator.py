# core/chat_orchestrator.py
import hashlib
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel

from ai.extraction import extract_signals, NEW_SEARCH_RE, normalize_role_with_api
from ai.generation import generate_coached_reply
from ai.role_resolver import resolve_role_from_dataset, build_search_keywords, strip_time_modifiers
from jobs.adzuna import fetch_jobs
from jobs.job_cards import to_job_cards
from memory.store import (
    get_user_state,
    get_user_memory,
    append_user_memory,
    save_user_job,
    clear_user_state,
    clear_user_memory,
    clear_user_jobs,
)
from core.state_machine import next_discovery_question

WELCOME_TEXT = (
    "Hey 👋 Tell me the role and location you’re looking for "
    "(e.g. “backend developer in London”)."
)

# -------------------------------------------------------------------
# In-memory session tracker (not persisted)
# -------------------------------------------------------------------
user_sessions: Dict[str, List["ChatSession"]] = {}


class ChatMessage(BaseModel):
    text: str
    sender: str  # "user" or "ai"
    timestamp: datetime


class ChatSession(BaseModel):
    id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime


# -------------------------------------------------------------------
# Response helpers
# -------------------------------------------------------------------
def make_response(
    assistantText: str,
    mode: str = "chat",
    actions: Optional[list] = None,
    jobs: Optional[list] = None,
    links: Optional[list] = None,
    debug: Optional[dict] = None,
) -> Dict[str, Any]:
    return {
        "assistantText": assistantText,
        "mode": mode,
        "actions": actions or [],
        "jobs": jobs or [],
        "links": links or [],
        "debug": debug or {},
    }


def _remember_and_return(
    *,
    user_id: str,
    sessions: List[ChatSession],
    session: ChatSession,
    now: datetime,
    text: str,
    mode: str = "chat",
    actions: Optional[list] = None,
    jobs: Optional[list] = None,
    links: Optional[list] = None,
    debug: Optional[dict] = None,
) -> Dict[str, Any]:
    append_user_memory(str(user_id), "assistant", text)
    session.messages.append(ChatMessage(text=text, sender="ai", timestamp=now))
    user_sessions[user_id] = sessions
    return make_response(text, mode=mode, actions=actions, jobs=jobs, links=links, debug=debug)


# -------------------------------------------------------------------
# Job ID: stable-ish dedupe key
# -------------------------------------------------------------------
def _job_id(job: dict) -> str:
    raw = (
        f"{job.get('title','')}|{job.get('company','')}|"
        f"{job.get('location','')}|{job.get('redirect_url','')}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------
# State helpers
# -------------------------------------------------------------------
def _strip_role_fields_in_state(state: Dict[str, Any]) -> None:
    for k in ("role_raw", "resolved_role", "role_canon"):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            state[k] = strip_time_modifiers(v).strip()


def _reset_search_state(state: Dict[str, Any], *, keep_location: bool = True) -> None:
    loc = state.get("location") if keep_location else None
    state.update(
        {
            "phase": "discovery",
            "jobs_shown": False,
            "asked_income_type": False,
            "income_type": None,
            "current_deck": None,
            "cached_jobs": [],
            "resolved_role": None,
            "role_raw": None,
            "role_keywords": None,
            "role_display": None,
            "role_canon": None,
            "role_query": None,
            "location": loc,
        }
    )


def _get_or_create_session(user_id: str, now: datetime) -> Tuple[List[ChatSession], ChatSession]:
    sessions = user_sessions.setdefault(user_id, [])
    if not sessions or (now - sessions[-1].last_activity) > timedelta(hours=2):
        sessions.append(
            ChatSession(
                id=str(now.timestamp()),
                messages=[],
                created_at=now,
                last_activity=now,
            )
        )
    session = sessions[-1]
    session.last_activity = now
    return sessions, session


def _role_logic(state: Dict[str, Any]) -> str:
    return (state.get("role_canon") or state.get("role_raw") or "").strip().lower()


def _role_display(state: Dict[str, Any]) -> str:
    return (state.get("role_display") or state.get("role_keywords") or "").strip()


# -------------------------------------------------------------------
# Intent detection
# -------------------------------------------------------------------
PIVOT_RE = re.compile(r"\b(actually|instead|change|different|switch|new\s+role|new\s+job)\b", re.I)
ACK_ONLY_RE = re.compile(r"^\s*(thanks|thank you|ok|okay|cool|great)\s*[.!?]?\s*$", re.I)
RESET_RE = re.compile(r"\b(reset|start over|new search|clear everything)\b", re.I)

REFLECTIVE_RE = re.compile(r"\b(confused|lost|unsure|not sure|don't know|future|career|stuck)\b", re.I)


def _is_greeting(low: str) -> bool:
    return low in {"hi", "hello", "hey", "hiya"}


def _is_ack(low: str) -> bool:
    return bool(ACK_ONLY_RE.match(low))


def _is_new_search_intent(low: str) -> bool:
    return bool(NEW_SEARCH_RE.search(low) or PIVOT_RE.search(low))


def _is_reflective(low: str) -> bool:
    return bool(REFLECTIVE_RE.search(low))


# -------------------------------------------------------------------
# Role resolution (dataset first, API fallback)
# -------------------------------------------------------------------
async def _resolve_role_for_search(state: Dict[str, Any]) -> Tuple[str, str]:
    role_input = _role_logic(state)
    role_input = strip_time_modifiers(role_input).strip()

    resolved = resolve_role_from_dataset(role_input)
    if resolved:
        resolved = strip_time_modifiers(resolved).strip()

    if not resolved:
        ai_clean = await normalize_role_with_api(role_input)
        resolved = strip_time_modifiers(ai_clean).strip()

    if not resolved:
        resolved = role_input

    resolved = resolved.lower().strip()
    search_keywords = build_search_keywords(resolved)
    return resolved, search_keywords


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
async def chat_with_user(
    *,
    user_id: str,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    now = datetime.utcnow()
    user_message = user_message or ""
    low = user_message.lower().strip()

    sessions, session = _get_or_create_session(user_id, now)
    session.messages.append(ChatMessage(text=user_message, sender="user", timestamp=now))

    state: Dict[str, Any] = get_user_state(str(user_id))
    memory: List[Dict[str, Any]] = conversation_history or get_user_memory(str(user_id))
    append_user_memory(str(user_id), "user", user_message)

    # 1) Greeting (soft – no reset)
    if _is_greeting(low):
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=WELCOME_TEXT,
            actions=[],
            links=[],
            debug={"intent": "greeting"},
        )

    # 2) Reset intent (hard reset)
    if RESET_RE.search(low):
        clear_user_state(str(user_id))
        clear_user_memory(str(user_id))
        clear_user_jobs(str(user_id))
        state = get_user_state(str(user_id))

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Alright, starting fresh. What role and location are you looking for?",
            actions=[],
            links=[],
            debug={"intent": "reset"},
        )

    # 3) Simple acknowledgements
    if _is_ack(low):
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="You’re welcome. Want to search for something else, or tweak the role/location?",
            actions=[],
            links=[],
            debug={"intent": "ack"},
        )

    # 4) Adaptive “career confusion” routing (state-aware)
    #    IMPORTANT: do this before extract_signals so we don’t force job parsing on reflective input.
    already_have_role = bool(_role_logic(state))
    already_have_loc = bool((state.get("location") or "").strip())

    # Handle responses to clarity offer
    if state.get("phase") == "clarity_offer":
        if low in {"yes", "yeah", "yep", "clarity_yes"}:
            state["phase"] = "clarity_level"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text=(
                    "Cool. Quick one.\n"
                    "What’s your experience level?\n"
                    "1) Student  2) Entry  3) 1–3 yrs  4) 3–7 yrs  5) 7+ yrs"
                ),
                actions=[],
                links=[],
                debug={"intent": "clarity_level"},
            )
        if low in {"no", "nah", "nope", "clarity_no"}:
            state["phase"] = "discovery"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="All good. Tell me a role + city and I’ll pull listings.",
                actions=[],
                links=[],
                debug={"intent": "clarity_declined"},
            )

        # If they typed something else, gently re-ask
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Do you want the quick clarity pass? Yes or no.",
            actions=[{"type": "YES_NO", "yesValue": "clarity_yes", "noValue": "clarity_no"}],
            links=[],
            debug={"intent": "clarity_offer_repeat"},
        )

    # Minimal clarity flow (V1): capture experience level then move back to practical search
    if state.get("phase") == "clarity_level":
        # Accept 1-5 or words
        if any(x in low for x in {"1", "student"}):
            state["clarity_level"] = "student"
        elif any(x in low for x in {"2", "entry"}):
            state["clarity_level"] = "entry"
        elif any(x in low for x in {"3", "1–3", "1-3"}):
            state["clarity_level"] = "1-3"
        elif any(x in low for x in {"4", "3–7", "3-7"}):
            state["clarity_level"] = "3-7"
        elif any(x in low for x in {"5", "7+"}):
            state["clarity_level"] = "7+"
        else:
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Pick one number: 1) Student  2) Entry  3) 1–3 yrs  4) 3–7 yrs  5) 7+ yrs",
                actions=[],
                links=[],
                debug={"intent": "clarity_level_reprompt"},
            )

        state["phase"] = "discovery"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "Nice. Now tell me either:\n"
                "• a role + location to search (example: waiter in Edinburgh)\n"
                "or\n"
                "• two roles + your city (example: barista vs receptionist, Edinburgh)."
            ),
            actions=[],
            links=[],
            debug={"intent": "clarity_to_search", "clarity_level": state.get("clarity_level")},
        )

    # If the new message is reflective, route based on what we already know
    if _is_reflective(low):
        if already_have_role and already_have_loc:
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Got you. Want to keep it practical and continue the search, or change role/location?",
                actions=[{"type": "YES_NO", "yesValue": "continue_search", "noValue": "change_search"}],
                links=[],
                debug={"intent": "reflective_with_context"},
            )

        if already_have_role and not already_have_loc:
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Got you. Which city should I search in?",
                actions=[],
                links=[],
                debug={"intent": "reflective_missing_location"},
            )

        if already_have_loc and not already_have_role:
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Got you. What kind of role should I search for?",
                actions=[],
                links=[],
                debug={"intent": "reflective_missing_role"},
            )

        # Nothing known: offer clarity pass (YES/NO)
        state["phase"] = "clarity_offer"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "Got you. Want a quick clarity pass (2 minutes) so I can suggest roles to search for?\n"
                "Or if you already have something in mind, just tell me a role + city."
            ),
            actions=[{"type": "YES_NO", "yesValue": "clarity_yes", "noValue": "clarity_no"}],
            links=[],
            debug={"intent": "clarity_offer"},
        )

    # Handle yes/no from reflective-with-context path
    if low in {"continue_search"}:
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Alright. Tell me the role and location you want to search.",
            actions=[],
            links=[],
            debug={"intent": "continue_search"},
        )

    if low in {"change_search"}:
        _reset_search_state(state, keep_location=False)
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="No problem. What role and location do you want instead?",
            actions=[],
            links=[],
            debug={"intent": "change_search"},
        )

    # 5) New search intent early (your existing pivot rule)
    if _is_new_search_intent(low):
        _reset_search_state(state, keep_location=True)

    prev_role = _role_logic(state)
    prev_loc = (state.get("location") or "").strip().lower()
    prev_income = (state.get("income_type") or "").strip().lower()

    # Extract signals (mutates state)
    await extract_signals(user_message, state)
    _strip_role_fields_in_state(state)

    new_role = _role_logic(state)
    new_loc = (state.get("location") or "").strip().lower()
    new_income = (state.get("income_type") or "").strip().lower()

    # 6) If user changed role/location/income, restart job search state
    changed_role = bool(prev_role and new_role and prev_role != new_role)
    changed_loc = bool(prev_loc and new_loc and prev_loc != new_loc)
    changed_income = bool(prev_income and new_income and prev_income != new_income)

    if changed_role or changed_loc or changed_income:
        keep_location_value = state.get("location")
        keep_role_canon = state.get("role_canon")
        keep_role_display = state.get("role_display") or state.get("role_keywords")
        keep_income = state.get("income_type")

        _reset_search_state(state, keep_location=False)
        state["location"] = keep_location_value
        state["role_canon"] = keep_role_canon
        state["role_display"] = keep_role_display
        state["role_keywords"] = keep_role_display  # legacy
        state["income_type"] = keep_income

    # 7) Post-swipe flow
    if state.get("phase") == "post_swipe":
        deck = state.get("current_deck") or {}
        liked_ids = set(deck.get("liked") or [])
        cached_cards = state.get("cached_jobs") or []
        liked_cards = [c for c in cached_cards if c.get("id") in liked_ids]

        if low in {"yes", "yeah", "yep", "talk_yes"}:
            state["phase"] = "discuss_likes"
            text = "Cool. What matters most to you: pay, flexibility, location, or growth?"
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=text)

        if low in {"no", "n", "nah", "nope", "talk_no"}:
            links = [
                {"label": f"{c.get('title','Job')} at {c.get('company','')}".strip(), "url": c.get("redirect_url", "")}
                for c in liked_cards
                if c.get("redirect_url")
            ]
            state["phase"] = "ready"
            text = "All good. Here are the direct links to the ones you liked:"
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=text, links=links)

        text = "Quick one: do you want to talk about the jobs you liked? Yes or no."
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=text,
            mode="post_swipe",
            actions=[{"type": "YES_NO", "yesValue": "talk_yes", "noValue": "talk_no"}],
        )

    # 8) Discovery (ONLY if not ready)
    if state.get("phase") == "discovery" and not state.get("readiness"):
        q = next_discovery_question(state)
        if q:
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=q)

    # 9) Income type clarification (only when we have role + location)
    if state.get("income_type") is None and not state.get("asked_income_type"):
        if _role_logic(state) and state.get("location"):
            state["asked_income_type"] = True
            role_text = _role_display(state) or "that role"
            text = f"Do you want full-time or part-time {role_text} work in {state['location']}?"
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=text)

    if state.get("asked_income_type"):
        if "part" in low:
            state["income_type"] = "part-time"
            state["jobs_shown"] = False
        elif "full" in low:
            state["income_type"] = "full-time"
            state["jobs_shown"] = False
        else:
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text="Full-time or part-time?")
        state["asked_income_type"] = False

    # 10) Fetch jobs -> cards -> deck
    if _role_logic(state) and state.get("location") and not state.get("jobs_shown"):
        income_type = state.get("income_type") or "job"
        resolved_role, search_keywords = await _resolve_role_for_search(state)
        state["resolved_role"] = resolved_role
        state["role_query"] = search_keywords

        jobs = await fetch_jobs(role_keywords=search_keywords, location=state["location"], income_type=income_type)

        uniq = {(j.get("title", ""), j.get("company", ""), j.get("location", "")): j for j in (jobs or [])}
        jobs = list(uniq.values())

        for j in jobs:
            j["id"] = _job_id(j)

        if not jobs:
            state["jobs_shown"] = False
            state["phase"] = "no_results"
            text = (
                f"I couldn’t find any listings for '{resolved_role or 'that role'}' in {state['location']}. "
                "Try a nearby city, a broader title (e.g. 'software engineer'), or remove 'part-time'."
            )
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text=text,
                mode="no_results",
                debug={"resolved_role": resolved_role, "search_keywords": search_keywords, "income_type": income_type},
            )

        cards = to_job_cards(
            jobs,
            role_canon=_role_logic(state) or resolved_role,
            location=state["location"],
            income_type=income_type,
        )
        state["cached_jobs"] = cards
        state["jobs_shown"] = True
        state["phase"] = "results_found"

        for job in jobs:
            save_user_job(str(user_id), job)

        deck_id = uuid.uuid4().hex
        deck_cards = cards[:8]
        state["current_deck"] = {
            "deck_id": deck_id,
            "job_ids": [c["id"] for c in deck_cards],
            "liked": [],
            "passed": [],
            "complete": False,
        }

        text = (
            f"Found {len(cards)} listings for '{resolved_role}' in {state['location']}. "
            f"Want to swipe through the top {len(deck_cards)}?"
        )
        actions = [{"type": "OPEN_SWIPE", "label": "Swipe jobs", "deckId": deck_id}]

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=text,
            mode="results_found",
            actions=actions,
            debug={
                "resolved_role": resolved_role,
                "search_keywords": search_keywords,
                "totalFound": len(cards),
                "deckSize": len(deck_cards),
            },
        )

    # 11) Fallback
    reply = await generate_coached_reply(state, memory, user_message)
    reply = reply.strip() or "Tell me the role and location you want, and I’ll find jobs."
    return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=reply)
