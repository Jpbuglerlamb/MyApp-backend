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
from ai.extraction import normalize_role_with_api
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
    "Hey 👋 Tell me what you’re looking for.\n"
    "Example: “waiter in Edinburgh” or “backend developer in London”."
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


def _job_id(job: dict) -> str:
    raw = (
        f"{job.get('title','')}|{job.get('company','')}|"
        f"{job.get('location','')}|{job.get('redirect_url','')}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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
            "role_keywords": None,   # legacy display fallback
            "role_display": None,
            "role_canon": None,
            "role_query": None,
            "location": loc,
            # clarity fields
            "clarity_profile": None,        # dict
            "clarity_answers": {},          # dict
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

REFLECTIVE_RE = re.compile(
    r"\b(confused|lost|unsure|not sure|dont know|don't know|stuck|overwhelmed|anxious|future|career)\b",
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

def _is_yes(low: str) -> bool:
    return low in {"yes", "yeah", "yep", "y", "sure", "ok", "okay"}

def _is_no(low: str) -> bool:
    return low in {"no", "nah", "nope", "n"}


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
# Clarity flow (simple, stateful, feels “alive” without changing UI)
# -------------------------------------------------------------------
def _clarity_offer_actions() -> list:
    # IMPORTANT: send "yes"/"no" so the user's bubble doesn't show "clarity_yes"
    return [{"type": "YES_NO", "yesLabel": "Yes", "noLabel": "No", "yesValue": "yes", "noValue": "no"}]

def _clarity_question_1() -> str:
    return (
        "Got you. Want a quick clarity pass (2 mins) so I can suggest directions?\n\n"
        "I’ll ask 3 quick questions. Ready?"
    )

def _clarity_question_2() -> str:
    return (
        "1/3: What’s your experience level?\n"
        "Student, Entry, 1–3 yrs, 3–7 yrs, 7+ yrs"
    )

def _clarity_question_3() -> str:
    return (
        "2/3: What do you enjoy more?\n"
        "A) People + pace  B) Deep focus  C) Creative work  D) Systems/logic"
    )

def _clarity_question_4() -> str:
    return (
        "3/3: What matters most right now?\n"
        "A) Money  B) Flexibility  C) Growth  D) Stability"
    )

def _clarity_summary(state: Dict[str, Any]) -> str:
    a = state.get("clarity_answers") or {}
    level = a.get("level")
    enjoy = a.get("enjoy")
    priority = a.get("priority")

    # Light, deterministic mapping (no model call) so it's reliable.
    suggestions: List[str] = []
    if enjoy in {"b", "systems", "logic", "d"}:
        suggestions += ["IT Support (entry)", "Junior QA", "Junior Data Analyst"]
    if enjoy in {"c", "creative"}:
        suggestions += ["Junior UI/UX", "Junior Graphic Designer", "Content / Social (entry)"]
    if enjoy in {"a", "people"}:
        suggestions += ["Sales Development", "Customer Support", "Hospitality (short-term)"]

    if level in {"student", "entry"}:
        suggestions = [s for s in suggestions if "Junior" in s or "entry" in s.lower() or "IT Support" in s]

    suggestions = suggestions[:4] or ["IT Support (entry)", "Customer Support", "Junior QA"]

    lines = "\n".join([f"• {s}" for s in suggestions])
    return (
        "Nice. Based on that, here are a few directions to explore:\n"
        f"{lines}\n\n"
        "Now make it practical:\n"
        "Tell me either:\n"
        "• a role + city to search (example: “waiter in Edinburgh”)\n"
        "or\n"
        "• two roles + your city (example: “IT support vs customer support, Edinburgh”)."
    )


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

    # 0) Empty ping -> welcome (lets the client call /chat with "")
    if low == "":
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=WELCOME_TEXT,
            actions=[],
            links=[],
            debug={"intent": "welcome"},
        )

    # 1) Greeting (soft)
    if _is_greeting(low):
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=WELCOME_TEXT,
            debug={"intent": "greeting"},
        )

    # 2) Reset (hard)
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
            text="Alright, starting fresh. What are you looking for?",
            debug={"intent": "reset"},
        )

    # 3) Ack
    if _is_ack(low):
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Anytime. Want to search for something else, or tweak role/location?",
            debug={"intent": "ack"},
        )

    # 4) Clarity flow phases
    phase = state.get("phase")

    if phase == "clarity_offer":
        if _is_yes(low):
            state["phase"] = "clarity_q1"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text=_clarity_question_2(),
                debug={"intent": "clarity_q1"},
            )
        if _is_no(low):
            state["phase"] = "discovery"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="No worries. If you already know what you want, tell me a role + city.",
                debug={"intent": "clarity_declined"},
            )
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text="Want the quick clarity pass? Yes or no.",
            actions=_clarity_offer_actions(),
            debug={"intent": "clarity_offer_repeat"},
        )

    if phase == "clarity_q1":
        ans = low.strip()
        if "student" in ans:
            state.setdefault("clarity_answers", {})["level"] = "student"
        elif "entry" in ans:
            state.setdefault("clarity_answers", {})["level"] = "entry"
        elif "1" in ans and "3" in ans:
            state.setdefault("clarity_answers", {})["level"] = "1-3"
        elif "3" in ans and "7" in ans:
            state.setdefault("clarity_answers", {})["level"] = "3-7"
        elif "7" in ans or "+" in ans:
            state.setdefault("clarity_answers", {})["level"] = "7+"
        else:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="Pick one: Student / Entry / 1–3 yrs / 3–7 yrs / 7+ yrs",
                debug={"intent": "clarity_q1_reprompt"},
            )

        state["phase"] = "clarity_q2"
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text=_clarity_question_3(),
            debug={"intent": "clarity_q2"},
        )

    if phase == "clarity_q2":
        # accept A/B/C/D or words
        a = low.strip()
        val = None
        if a.startswith("a") or "people" in a:
            val = "a"
        elif a.startswith("b") or "focus" in a:
            val = "b"
        elif a.startswith("c") or "creative" in a:
            val = "c"
        elif a.startswith("d") or "systems" in a or "logic" in a:
            val = "d"

        if not val:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="Reply A / B / C / D (or say: people, focus, creative, systems).",
                debug={"intent": "clarity_q2_reprompt"},
            )

        state.setdefault("clarity_answers", {})["enjoy"] = val
        state["phase"] = "clarity_q3"
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text=_clarity_question_4(),
            debug={"intent": "clarity_q3"},
        )

    if phase == "clarity_q3":
        a = low.strip()
        val = None
        if a.startswith("a") or "money" in a or "pay" in a:
            val = "a"
        elif a.startswith("b") or "flex" in a:
            val = "b"
        elif a.startswith("c") or "growth" in a:
            val = "c"
        elif a.startswith("d") or "stability" in a:
            val = "d"

        if not val:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="Reply A / B / C / D (money, flexibility, growth, stability).",
                debug={"intent": "clarity_q3_reprompt"},
            )

        state.setdefault("clarity_answers", {})["priority"] = val
        state["phase"] = "discovery"
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text=_clarity_summary(state),
            debug={"intent": "clarity_done", "clarity": state.get("clarity_answers")},
        )

    # 5) Reflective input -> offer clarity (only if we lack enough slots)
    already_have_role = bool(_role_logic(state))
    already_have_loc = bool((state.get("location") or "").strip())

    if _is_reflective(low) and not (already_have_role and already_have_loc):
        state["phase"] = "clarity_offer"
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text=_clarity_question_1(),
            actions=_clarity_offer_actions(),
            debug={"intent": "clarity_offer"},
        )

    # 6) New search intent (pivot)
    if _is_new_search_intent(low):
        _reset_search_state(state, keep_location=True)

    prev_role = _role_logic(state)
    prev_loc = (state.get("location") or "").strip().lower()
    prev_income = (state.get("income_type") or "").strip().lower()

    # 7) Extract signals (mutates state)
    await extract_signals(user_message, state)

    new_role = _role_logic(state)
    new_loc = (state.get("location") or "").strip().lower()
    new_income = (state.get("income_type") or "").strip().lower()

    # 8) If user changed role/location/income, restart job search state
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

    # 9) Post-swipe flow (Yes/No buttons send "yes"/"no")
    if state.get("phase") == "post_swipe":
        deck = state.get("current_deck") or {}
        liked_ids = set(deck.get("liked") or [])
        cached_cards = state.get("cached_jobs") or []
        liked_cards = [c for c in cached_cards if c.get("id") in liked_ids]

        if _is_yes(low):
            state["phase"] = "discuss_likes"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="Cool. What matters most to you: pay, flexibility, location, or growth?",
                debug={"intent": "post_swipe_discuss"},
            )

        if _is_no(low):
            links = [
                {"label": f"{c.get('title','Job')} at {c.get('company','')}".strip(), "url": c.get("redirect_url", "")}
                for c in liked_cards if c.get("redirect_url")
            ]
            state["phase"] = "ready"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="All good. Here are the direct links to the ones you liked:",
                links=links,
                debug={"intent": "post_swipe_links"},
            )

        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text="Quick one: do you want to talk about the jobs you liked? Yes or no.",
            mode="post_swipe",
            actions=[{"type": "YES_NO", "yesLabel": "Yes", "noLabel": "No", "yesValue": "yes", "noValue": "no"}],
            debug={"intent": "post_swipe_reprompt"},
        )

    # 10) Discovery (ONLY if not ready)
    if state.get("phase") == "discovery" and not state.get("readiness"):
        q = next_discovery_question(state)
        if q:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now, text=q,
                debug={"intent": "discovery_question"},
            )

    # 11) Income type clarification (only when we have role + location)
    if state.get("income_type") is None and not state.get("asked_income_type"):
        if _role_logic(state) and state.get("location"):
            state["asked_income_type"] = True
            role_text = _role_display(state) or "that role"
            text = f"Do you want full-time or part-time {role_text} work in {state['location']}?"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text=text,
                actions=[
                    {"type": "YES_NO", "yesLabel": "Full-time", "noLabel": "Part-time", "yesValue": "full-time", "noValue": "part-time"}
                ],
                debug={"intent": "income_type_ask"},
            )

    if state.get("asked_income_type"):
        if "part" in low:
            state["income_type"] = "part-time"
            state["jobs_shown"] = False
        elif "full" in low:
            state["income_type"] = "full-time"
            state["jobs_shown"] = False
        else:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
                text="Full-time or part-time?",
                debug={"intent": "income_type_reprompt"},
            )
        state["asked_income_type"] = False

    # 12) Fetch jobs -> cards -> deck
    if _role_logic(state) and state.get("location") and not state.get("jobs_shown"):
        income_type = state.get("income_type") or "job"
        resolved_role, search_keywords = await _resolve_role_for_search(state)
        state["resolved_role"] = resolved_role
        state["role_query"] = search_keywords

        jobs = await fetch_jobs(
            role_keywords=search_keywords,
            location=state["location"],
            income_type=income_type,
        )

        # Deduplicate
        uniq = {(j.get("title", ""), j.get("company", ""), j.get("location", "")): j for j in (jobs or [])}
        jobs = list(uniq.values())

        # Stable IDs
        for j in jobs:
            j["id"] = _job_id(j)

        if not jobs:
            state["jobs_shown"] = False
            state["phase"] = "no_results"
            text = (
                f"I couldn’t find any listings for '{resolved_role or 'that role'}' in {state['location']}.\n"
                "Try a nearby city, a broader title, or change full/part-time."
            )
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now,
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

        text = f"Found {len(cards)} listings for '{resolved_role}' in {state['location']}. Want to swipe the top {len(deck_cards)}?"
        actions = [{"type": "OPEN_SWIPE", "label": "Swipe jobs", "deckId": deck_id}]

        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now,
            text=text,
            mode="results_found",
            actions=actions,
            debug={"resolved_role": resolved_role, "search_keywords": search_keywords, "totalFound": len(cards), "deckSize": len(deck_cards)},
        )

    # 13) Fallback
    reply = await generate_coached_reply(state, memory, user_message)
    reply = reply.strip() or "Tell me the role and location you want, and I’ll find jobs."
    return _remember_and_return(
        user_id=user_id, sessions=sessions, session=session, now=now,
        text=reply,
        debug={"intent": "fallback"},
    )
