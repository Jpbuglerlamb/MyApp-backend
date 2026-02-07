# core/chat_orchestrator.py
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

from telemetry.logger import log_event

WELCOME_TEXT = "Hey 👋 Tell me the role and location you’re looking for (e.g. “backend developer in London”)."

# -------------------------------------------------------------------
# In-memory sessions (optional)
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

    # telemetry: outcome
    log_event(
        "orchestrator_responded",
        {"mode": mode, "intent": (debug or {}).get("intent"), "assistant_len": len(text or "")},
        user_id=user_id,
    )

    return make_response(text, mode=mode, actions=actions, jobs=jobs, links=links, debug=debug)


def _job_id(job: dict) -> str:
    raw = (
        f"{job.get('title','')}|{job.get('company','')}|"
        f"{job.get('location','')}|{job.get('redirect_url','')}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _get_or_create_session(user_id: str, now: datetime) -> Tuple[List[ChatSession], ChatSession]:
    sessions = user_sessions.setdefault(user_id, [])
    if not sessions or (now - sessions[-1].last_activity) > timedelta(hours=2):
        sessions.append(ChatSession(id=str(now.timestamp()), messages=[], created_at=now, last_activity=now))
    session = sessions[-1]
    session.last_activity = now
    return sessions, session


def _role_logic(state: Dict[str, Any]) -> str:
    return (state.get("role_canon") or state.get("role_raw") or "").strip().lower()


def _role_display(state: Dict[str, Any]) -> str:
    return (state.get("role_display") or state.get("role_keywords") or "").strip()


def _strip_role_fields_in_state(state: Dict[str, Any]) -> None:
    for k in ("role_raw", "resolved_role", "role_canon"):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            state[k] = strip_time_modifiers(v).strip()


def _reset_search_state(state: Dict[str, Any], *, keep_location: bool = True) -> None:
    loc = state.get("location") if keep_location else None
    # Keep clarity fields
    clarity_level = state.get("clarity_level")
    clarity_interest = state.get("clarity_interest")
    clarity_constraints = state.get("clarity_constraints")

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
            # keep clarity memory
            "clarity_level": clarity_level,
            "clarity_interest": clarity_interest,
            "clarity_constraints": clarity_constraints,
        }
    )


# -------------------------------------------------------------------
# Intent detection
# -------------------------------------------------------------------
PIVOT_RE = re.compile(r"\b(actually|instead|change|different|switch|new\s+role|new\s+job)\b", re.I)
ACK_ONLY_RE = re.compile(r"^\s*(thanks|thank you|ok|okay|cool|great)\s*[.!?]?\s*$", re.I)
RESET_RE = re.compile(r"\b(reset|start over|new search|clear everything)\b", re.I)
REFLECTIVE_RE = re.compile(r"\b(confused|lost|unsure|not sure|don\'t know|stuck|future|career|life|anxious|stressed)\b", re.I)


def _is_greeting(low: str) -> bool:
    return low in {"hi", "hello", "hey", "hiya"}


def _is_ack(low: str) -> bool:
    return bool(ACK_ONLY_RE.match(low))


def _is_new_search_intent(low: str) -> bool:
    return bool(NEW_SEARCH_RE.search(low) or PIVOT_RE.search(low))


def _is_reflective(low: str) -> bool:
    return bool(REFLECTIVE_RE.search(low))


# -------------------------------------------------------------------
# Clarity pass (simple but "alive")
# -------------------------------------------------------------------
def _yes_no_actions(yes_value: str, no_value: str, yes_label: str = "Yes", no_label: str = "No") -> list:
    return [{"type": "YES_NO", "yesLabel": yes_label, "noLabel": no_label, "yesValue": yes_value, "noValue": no_value}]


def _role_suggestions_from_clarity(level: str, interest: str) -> List[str]:
    """
    Tiny rules engine. No deep learning required.
    """
    interest_low = (interest or "").lower()
    level_low = (level or "").lower()

    techy = any(k in interest_low for k in ["tech", "software", "coding", "data", "ai", "computer"])
    creative = any(k in interest_low for k in ["design", "ui", "ux", "creative", "graphics"])
    people = any(k in interest_low for k in ["people", "help", "support", "care", "customer"])

    if "student" in level_low or "entry" in level_low:
        if techy:
            return ["Junior Software Developer", "IT Support Technician", "Data Analyst (Junior)"]
        if creative:
            return ["Junior UI Designer", "Junior UX Research Assistant", "Marketing Assistant"]
        if people:
            return ["Customer Support Advisor", "Recruitment Resourcer", "Sales Assistant"]
        return ["Customer Support Advisor", "Administrative Assistant", "Retail Assistant"]

    # experienced
    if techy:
        return ["Software Engineer", "Data Engineer", "Product Analyst"]
    if creative:
        return ["UI Designer", "UX Designer", "Content Designer"]
    if people:
        return ["Account Manager", "Team Lead (Support)", "Recruiter"]
    return ["Operations Coordinator", "Project Coordinator", "Account Manager"]


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

    # Ensure core keys exist
    state.setdefault("phase", "discovery")
    state.setdefault("jobs_shown", False)
    state.setdefault("asked_income_type", False)

    log_event(
        "orchestrator_received",
        {"phase": state.get("phase"), "has_role": bool(_role_logic(state)), "has_loc": bool(state.get("location"))},
        user_id=user_id,
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
        state.setdefault("phase", "discovery")

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Alright, starting fresh. What role and location are you looking for?",
            debug={"intent": "reset"},
        )

    # 3) Acknowledgements
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
    # 4) Clarity pass state machine (handles reflective inputs)
    # -------------------------------------------------------------------
    # Handle "action values" without needing UI changes
    if low in {"clarity_yes", "clarity_no", "talk_yes", "talk_no", "continue_search", "change_search"}:
        # keep as-is; handled below by phase checks
        pass

    if state.get("phase") == "clarity_offer":
        if low in {"yes", "y", "yeah", "yep", "clarity_yes"}:
            state["phase"] = "clarity_level"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Cool. What’s your experience level?\n1) Student  2) Entry  3) 1–3 yrs  4) 3–7 yrs  5) 7+ yrs",
                debug={"intent": "clarity_level"},
            )
        if low in {"no", "n", "nah", "nope", "clarity_no"}:
            state["phase"] = "discovery"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="All good. If you tell me a role + city, I’ll pull listings straight away.",
                debug={"intent": "clarity_declined"},
            )
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Do you want the quick clarity pass? Yes or no.",
            actions=_yes_no_actions("clarity_yes", "clarity_no"),
            debug={"intent": "clarity_offer_repeat"},
        )

    if state.get("phase") == "clarity_level":
        lvl = None
        if "1" in low or "student" in low:
            lvl = "student"
        elif "2" in low or "entry" in low:
            lvl = "entry"
        elif "3" in low or "1-3" in low or "1–3" in low:
            lvl = "1-3"
        elif "4" in low or "3-7" in low or "3–7" in low:
            lvl = "3-7"
        elif "5" in low or "7+" in low:
            lvl = "7+"

        if not lvl:
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Pick one: 1) Student  2) Entry  3) 1–3 yrs  4) 3–7 yrs  5) 7+ yrs",
                debug={"intent": "clarity_level_reprompt"},
            )

        state["clarity_level"] = lvl
        state["phase"] = "clarity_interest"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Nice. What are you more into right now?\nA) Tech/Coding  B) Design/Creative  C) People/Support  D) Not sure",
            debug={"intent": "clarity_interest"},
        )

    if state.get("phase") == "clarity_interest":
        interest = None
        if low.startswith("a") or "tech" in low or "coding" in low or "software" in low or "data" in low or "ai" in low:
            interest = "tech"
        elif low.startswith("b") or "design" in low or "ux" in low or "ui" in low or "creative" in low:
            interest = "design"
        elif low.startswith("c") or "people" in low or "support" in low or "customer" in low:
            interest = "people"
        elif low.startswith("d") or "not sure" in low or "dont know" in low:
            interest = "unsure"
        else:
            # allow free text
            interest = low[:40]

        state["clarity_interest"] = interest
        state["phase"] = "clarity_constraints"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Any constraints? (e.g. part-time, remote, needs to be in Edinburgh, no weekends). If none, say “none”.",
            debug={"intent": "clarity_constraints"},
        )

    if state.get("phase") == "clarity_constraints":
        constraints = (user_message or "").strip()
        state["clarity_constraints"] = constraints if constraints and constraints.lower() != "none" else ""

        # Offer role suggestions
        sugg = _role_suggestions_from_clarity(state.get("clarity_level", ""), state.get("clarity_interest", ""))
        state["clarity_suggestions"] = sugg
        state["phase"] = "clarity_pick_role"

        text = (
            "Got it. Here are 3 solid roles to try searching:\n"
            f"1) {sugg[0]}\n2) {sugg[1]}\n3) {sugg[2]}\n\n"
            "Reply with 1/2/3, or type your own role + city."
        )
        return _remember_and_return(
            user_id=user_id, sessions=sessions, session=session, now=now, text=text, debug={"intent": "clarity_suggest_roles"}
        )

    if state.get("phase") == "clarity_pick_role":
        # If they pick 1/2/3, set role and continue normal extraction for location/income.
        if low in {"1", "2", "3"}:
            idx = int(low) - 1
            chosen = (state.get("clarity_suggestions") or ["", "", ""])[idx] or ""
            if chosen:
                state["role_canon"] = strip_time_modifiers(chosen).lower().strip()
                state["role_display"] = chosen
                state["role_raw"] = state["role_canon"]
                state["phase"] = "discovery"
                return _remember_and_return(
                    user_id=user_id,
                    sessions=sessions,
                    session=session,
                    now=now,
                    text=f"Cool. What city should I search for {chosen} in?",
                    debug={"intent": "clarity_role_chosen"},
                )
        # Otherwise fall through to normal pipeline; their message might contain role+location.

        state["phase"] = "discovery"

    # If reflective and we are NOT already in clarity flow, offer it.
    if _is_reflective(low) and state.get("phase") not in {"clarity_offer", "clarity_level", "clarity_interest", "clarity_constraints", "clarity_pick_role"}:
        have_role = bool(_role_logic(state))
        have_loc = bool((state.get("location") or "").strip())

        if have_role and have_loc:
            state["phase"] = "reflective_with_context"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Got you. Do you want to keep it practical and continue the search, or do a quick clarity pass?",
                actions=_yes_no_actions("continue_search", "clarity_yes", yes_label="Continue search", no_label="Clarity pass"),
                debug={"intent": "reflective_with_context"},
            )

        state["phase"] = "clarity_offer"
        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Got you. Want a quick clarity pass (under 2 minutes) so I can suggest roles to search for?\nOr you can just type a role + city.",
            actions=_yes_no_actions("clarity_yes", "clarity_no"),
            debug={"intent": "clarity_offer"},
        )

    # Handle "continue_search" from reflective-with-context
    if state.get("phase") == "reflective_with_context":
        if low in {"continue_search"}:
            state["phase"] = "discovery"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Alright. Tell me the role and location you want to search.",
                debug={"intent": "continue_search"},
            )

    # -------------------------------------------------------------------
    # 5) New search intent pivot
    # -------------------------------------------------------------------
    if _is_new_search_intent(low):
        _reset_search_state(state, keep_location=True)

    # Snapshot before extraction
    prev_role = _role_logic(state)
    prev_loc = (state.get("location") or "").strip().lower()
    prev_income = (state.get("income_type") or "").strip().lower()

    # 6) Extract signals (mutates state)
    await extract_signals(user_message, state)
    _strip_role_fields_in_state(state)

    # Snapshot after extraction
    new_role = _role_logic(state)
    new_loc = (state.get("location") or "").strip().lower()
    new_income = (state.get("income_type") or "").strip().lower()

    log_event(
        "signals_after",
        {
            "role_canon": state.get("role_canon"),
            "role_display": state.get("role_display"),
            "location": state.get("location"),
            "income_type": state.get("income_type"),
            "phase": state.get("phase"),
        },
        user_id=user_id,
    )

    # If user changed role/location/income, restart search fields cleanly
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

    # -------------------------------------------------------------------
    # 7) Post-swipe flow
    # -------------------------------------------------------------------
    if state.get("phase") == "post_swipe":
        deck = state.get("current_deck") or {}
        liked_ids = set(deck.get("liked") or [])
        cached_cards = state.get("cached_jobs") or []
        liked_cards = [c for c in cached_cards if c.get("id") in liked_ids]

        if low in {"yes", "yeah", "yep", "talk_yes"}:
            state["phase"] = "discuss_likes"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="Cool. What matters most to you: pay, flexibility, location, or growth?",
                debug={"intent": "post_swipe_discuss"},
            )

        if low in {"no", "n", "nah", "nope", "talk_no"}:
            links = [
                {"label": f"{c.get('title','Job')} at {c.get('company','')}".strip(), "url": c.get("redirect_url", "")}
                for c in liked_cards
                if c.get("redirect_url")
            ]
            state["phase"] = "ready"
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text="All good. Here are the direct links to the ones you liked:",
                links=links,
                debug={"intent": "post_swipe_links"},
            )

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text="Quick one: do you want to talk about the jobs you liked? Yes or no.",
            mode="post_swipe",
            actions=_yes_no_actions("talk_yes", "talk_no"),
            debug={"intent": "post_swipe_prompt"},
        )

    # -------------------------------------------------------------------
    # 8) Discovery prompt (ONLY if we still lack essentials)
    # -------------------------------------------------------------------
    if state.get("phase") == "discovery":
        # If they have nothing, ask discovery question
        q = next_discovery_question(state)
        if q and not (_role_logic(state) and state.get("location")):
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now, text=q, debug={"intent": "discovery_question"}
            )

    # -------------------------------------------------------------------
    # 9) Income type clarification (only when we have role + location)
    # -------------------------------------------------------------------
    if state.get("income_type") is None and not state.get("asked_income_type"):
        if _role_logic(state) and state.get("location"):
            state["asked_income_type"] = True
            role_text = _role_display(state) or "that role"
            text = f"Do you want full-time or part-time {role_text} work in {state['location']}?"
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now, text=text, debug={"intent": "ask_income_type"}
            )

    if state.get("asked_income_type"):
        if "part" in low:
            state["income_type"] = "part-time"
            state["jobs_shown"] = False
            state["asked_income_type"] = False
        elif "full" in low:
            state["income_type"] = "full-time"
            state["jobs_shown"] = False
            state["asked_income_type"] = False
        else:
            return _remember_and_return(
                user_id=user_id, sessions=sessions, session=session, now=now, text="Full-time or part-time?", debug={"intent": "reprompt_income"}
            )

    # -------------------------------------------------------------------
    # 10) Fetch jobs -> cards -> deck
    # -------------------------------------------------------------------
    if _role_logic(state) and state.get("location") and not state.get("jobs_shown"):
        income_type = state.get("income_type") or "job"

        # Resolve role for search
        role_input = strip_time_modifiers(_role_logic(state)).strip()
        resolved = resolve_role_from_dataset(role_input) or role_input
        resolved = strip_time_modifiers(resolved).strip().lower()
        search_keywords = build_search_keywords(resolved)

        state["resolved_role"] = resolved
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
                f"I couldn’t find any listings for '{resolved or 'that role'}' in {state['location']}. "
                "Try a nearby city, a broader title, or remove ‘part-time’."
            )
            return _remember_and_return(
                user_id=user_id,
                sessions=sessions,
                session=session,
                now=now,
                text=text,
                mode="no_results",
                debug={"intent": "no_results", "resolved_role": resolved, "search_keywords": search_keywords, "income_type": income_type},
            )

        cards = to_job_cards(jobs, role_canon=_role_logic(state) or resolved, location=state["location"], income_type=income_type)
        state["cached_jobs"] = cards
        state["jobs_shown"] = True
        state["phase"] = "results_found"

        for job in jobs:
            save_user_job(str(user_id), job)

        deck_id = uuid.uuid4().hex
        deck_cards = cards[:8]
        state["current_deck"] = {"deck_id": deck_id, "job_ids": [c["id"] for c in deck_cards], "liked": [], "passed": [], "complete": False}

        text = f"Found {len(cards)} listings for '{resolved}' in {state['location']}. Want to swipe through the top {len(deck_cards)}?"
        actions = [{"type": "OPEN_SWIPE", "label": "Swipe jobs", "deckId": deck_id}]

        return _remember_and_return(
            user_id=user_id,
            sessions=sessions,
            session=session,
            now=now,
            text=text,
            mode="results_found",
            actions=actions,
            debug={"intent": "results_found", "resolved_role": resolved, "search_keywords": search_keywords, "totalFound": len(cards), "deckSize": len(deck_cards)},
        )

    # -------------------------------------------------------------------
    # 11) Fallback: coached reply
    # -------------------------------------------------------------------
    reply = await generate_coached_reply(state, memory, user_message)
    reply = (reply or "").strip() or "Tell me the role and location you want, and I’ll find jobs."
    return _remember_and_return(user_id=user_id, sessions=sessions, session=session, now=now, text=reply, debug={"intent": "fallback"})
