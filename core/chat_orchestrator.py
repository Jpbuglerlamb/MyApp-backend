# core/chat_orchestrator.py
import hashlib
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel

from ai.extraction import extract_signals, NEW_SEARCH_RE
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
    "(e.g. “waiter in Edinburgh” or “backend developer in London”)."
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
    """
    Strip time/contract modifiers ONLY from canonical-ish role fields.
    Do NOT touch display fields.
    """
    for k in ("role_raw", "resolved_role", "role_canon"):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            state[k] = strip_time_modifiers(v).strip()


def _reset_search_state(state: Dict[str, Any], *, keep_location: bool = True) -> None:
    loc = state.get("location") if keep_location else None
    # Keep clarity fields; only reset job-search fields.
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

# If users say “confused about my life/career”, we route to clarity chat instead of searching.
REFLECTIVE_RE = re.compile(
    r"\b(confused|lost|unsure|stuck|overwhelmed|anxious|stressed|not sure|dont know|don't know|future|career|life)\b",
    re.I,
)

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
        # If we have nothing, just keep input (avoid AI calls here to prevent crashes).
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
            debug={"intent": "ack"},
        )

    # -------------------------------------------------------------------
    # 4) CLARITY CHAT (mini)
    # -------------------------------------------------------------------
    # IMPORTANT: handle this BEFORE extract_signals so we don't force job parsing.
    already_have_role = bool(_role_logic(state))
    already_have_loc = bool((state.get("location") or "").strip())

    # If user is reflective and we have no search context, offer clarity pass.
    if _is_reflective(low) and not (already_have_role and already_have_loc) and state.get("phase") not in {
        "clarity_offer", "clarity_profile", "clarity_constraints"
    }:
        state["phase"] = "clarity_offer"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "Got you.\n"
                "Want a quick clarity pass (2 minutes) so I can suggest roles to search for?\n"
                "Yes = clarity chat. No = straight to job search."
            ),
            actions=[{
                "type": "YES_NO",
                "yesLabel": "Yes",
                "noLabel": "No",
                "yesValue": "yes",
                "noValue": "no",
            }],
            debug={"intent": "clarity_offer"},
        )

    # Handle clarity offer response (use plain yes/no so UI shows normal words)
    if state.get("phase") == "clarity_offer":
        if low in {"yes", "y", "yeah", "yep"}:
            state["phase"] = "clarity_profile"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text=(
                    "Alright. Three quick things.\n"
                    "1) What’s your current level?\n"
                    "Student / Entry / 1–3 yrs / 3–7 yrs / 7+ yrs\n"
                    "2) What do you enjoy more: people, systems, visuals, numbers, or hands-on?\n"
                    "Reply in one line."
                ),
                debug={"intent": "clarity_profile"},
            )
        if low in {"no", "n", "nope", "nah"}:
            state["phase"] = "discovery"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="No problem. Tell me a role + city and I’ll pull listings.",
                debug={"intent": "clarity_declined"},
            )

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Quick one: do you want the clarity pass? Yes or no.",
            actions=[{
                "type": "YES_NO",
                "yesLabel": "Yes",
                "noLabel": "No",
                "yesValue": "yes",
                "noValue": "no",
            }],
            debug={"intent": "clarity_offer_repeat"},
        )

    # Capture profile answer (we keep it lightweight; don't overfit)
    if state.get("phase") == "clarity_profile":
        state["clarity_profile_raw"] = user_message.strip()[:280]
        state["phase"] = "clarity_constraints"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "Nice.\n"
                "Last one: what constraints matter right now?\n"
                "Examples: part-time, remote, no experience, need quick income, certain city.\n"
                "Reply in one line."
            ),
            debug={"intent": "clarity_constraints"},
        )

    if state.get("phase") == "clarity_constraints":
        state["clarity_constraints_raw"] = user_message.strip()[:280]
        state["phase"] = "discovery"
        # After clarity, we guide them into *either* search OR suggestions.
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "Got it.\n"
                "Now either:\n"
                "• Tell me a role + city to search (example: “waiter in Edinburgh”) OR\n"
                "• Tell me a city, and I’ll suggest 3 realistic roles to search for there."
            ),
            debug={"intent": "clarity_done"},
        )

    # If reflective but they DO have role+location already, keep it practical without derailing.
    if _is_reflective(low) and already_have_role and already_have_loc:
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=(
                "I hear you.\n"
                f"Do you want to keep searching for **{_role_display(state) or _role_logic(state)} in {state.get('location')}**, "
                "or do you want to change the role/city?"
            ),
            actions=[{
                "type": "YES_NO",
                "yesLabel": "Keep",
                "noLabel": "Change",
                "yesValue": "yes",
                "noValue": "no",
            }],
            debug={"intent": "reflective_with_context"},
        )

    # If we were in that reflective-with-context prompt: interpret yes/no based on phase-less context
    if state.get("last_intent") == "reflective_with_context":
        # (Optional hook if you store last_intent; not required. Leaving simple.)

        pass

    # -------------------------------------------------------------------
    # 5) New search intent early
    # -------------------------------------------------------------------
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
        state["role_keywords"] = keep_role_display
        state["income_type"] = keep_income

    # 7) Post-swipe flow
    if state.get("phase") == "post_swipe":
        deck = state.get("current_deck") or {}
        liked_ids = set(deck.get("liked") or [])
        cached_cards = state.get("cached_jobs") or []
        liked_cards = [c for c in cached_cards if c.get("id") in liked_ids]

        if low in {"yes", "yeah", "yep"}:
            state["phase"] = "discuss_likes"
            text = "Cool. What matters most to you: pay, flexibility, location, or growth?"
            return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=text)

        if low in {"no", "n", "nah", "nope"}:
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
            actions=[{
                "type": "YES_NO",
                "yesLabel": "Yes",
                "noLabel": "No",
                "yesValue": "yes",
                "noValue": "no",
            }],
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
        actions = [{
            "type": "OPEN_SWIPE",
            "label": "Swipe jobs",
            "deckId": deck_id
        }]

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=text,
            mode="results_found",
            actions=actions,
            debug={"resolved_role": resolved_role, "search_keywords": search_keywords, "totalFound": len(cards), "deckSize": len(deck_cards)},
        )

    # 11) Fallback (more “alive” because it can use clarity context if present)
    reply = await generate_coached_reply(state, memory, user_message)
    reply = reply.strip() or "Tell me the role and location you want, and I’ll find jobs."
    return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=reply)
