# ai/extraction.py
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional

from ai.client import client
from ai.role_resolver import (
    build_search_keywords,
    canonicalize_role,
    resolve_role_from_dataset,
)
from core.state_machine import advance_phase

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

_STOP: str = r"(?:\s+(?:in|near|around|based in|based)\b|[.,;!?]|$)"
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about|show me)\b", re.I)

UK_CITIES: List[str] = [
    "Edinburgh",
    "London",
    "Manchester",
    "Bristol",
    "Glasgow",
    "Leeds",
    "Liverpool",
    "Belfast",
    "Cardiff",
    "Dundee",
    "Aberdeen",
]

STANDARD_INCOME_TYPES: Dict[str, List[str]] = {
    "full-time": ["full time", "full-time", "permanent"],
    "part-time": ["part time", "part-time", "casual", "zero hour", "zero-hours", "zero hours"],
    "temporary": ["temporary", "temp", "short-term"],
    "freelance": ["freelance", "gig", "self-employed"],
    "contract": ["contract", "contractor"],
    "internship": ["intern", "internship", "trainee"],
}

# Display role synonyms (UX)
ROLE_SYNONYMS: Dict[str, str] = {
    # Tech
    "software engineer": "Software Developer",
    "software developer": "Software Developer",
    "web developer": "Software Developer",
    "app developer": "Software Developer",
    "frontend": "Frontend Developer",
    "front end": "Frontend Developer",
    "backend": "Backend Developer",
    "back end": "Backend Developer",
    "data analyst": "Data Analyst",
    "data scientist": "Data Scientist",
    "ui designer": "UI Designer",
    "ux designer": "UX Designer",
    "product designer": "Product Designer",
    # Hospitality
    "waiter": "Waiter",
    "waitress": "Waiter",
    "waiting staff": "Waiter",
    "server": "Waiter",
    "bartender": "Bartender",
    "bar staff": "Bartender",
    "barista": "Barista",
    "chef": "Chef",
    "cook": "Chef",
    # Other common
    "driver": "Driver",
    "delivery driver": "Driver",
    "customer service": "Customer Service",
    "sales": "Sales",
    "marketing": "Marketing",
}

# If these appear, they are NOT part of a role, they are “context glue”
_CONTEXT_CLAUSES_RE = re.compile(
    r"\b(while|until|so that|because|as i|so i can|then i|and then|to fund|to pay)\b.*$",
    re.I,
)

__all__ = [
    "extract_signals",
    "extract_dynamic_keywords",
    "normalize_income_type",
    "map_role_synonym",
    "normalize_role_for_api",
    "normalize_role_with_api",
    "NEW_SEARCH_RE",
]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _strip_fillers(text: str) -> str:
    """
    Remove filler words, but do NOT delete role words.
    Keep it conservative.
    """
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _drop_context_clauses(text: str) -> str:
    """Remove trailing 'while I...' style clauses that should not become role text."""
    return re.sub(_CONTEXT_CLAUSES_RE, "", (text or "").strip()).strip()


def normalize_income_type(user_text: str) -> Optional[str]:
    low = (user_text or "").lower()
    for key, variants in STANDARD_INCOME_TYPES.items():
        for v in variants:
            if v in low:
                return key
    return None


def map_role_synonym(role_text: str, cutoff: float = 0.72) -> str:
    if not role_text:
        return ""
    lowered = role_text.lower()

    for key, standard in ROLE_SYNONYMS.items():
        if key in lowered:
            return standard

    best_match: Optional[str] = None
    highest_ratio = 0.0
    for key, standard in ROLE_SYNONYMS.items():
        ratio = difflib.SequenceMatcher(None, lowered, key).ratio()
        if ratio > highest_ratio and ratio >= cutoff:
            best_match = standard
            highest_ratio = ratio

    return best_match if best_match else role_text.title()


def _best_city_match(loc: str) -> str:
    loc = (loc or "").strip().title()
    if not loc:
        return ""
    matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.70)
    return matches[0] if matches else loc


def _extract_location_fallback(message: str) -> Optional[str]:
    low = (message or "").lower()

    # direct "in X"
    m = re.search(r"\b(?:in|near|around|based in|based)\s+(.+?)" + _STOP, low, re.I)
    if m:
        return _best_city_match(m.group(1))

    # if user just types a city name
    for c in UK_CITIES:
        if c.lower() in low:
            return c
    return None


def _extract_role_fallback(message: str) -> str:
    """
    Try: "as a waiter", "looking for waiter", "job as waiter", etc.
    If not found, return "" (do NOT dump whole sentence).
    """
    low = (message or "").lower().strip()

    # kill trailing context clause early
    cleaned = _drop_context_clauses(low)

    patterns = [
        r"(?:work as|job as|as a|as an|be a|be an)\s+(.+?)" + _STOP,
        r"(?:looking for|find me|search for|need)\s+(.+?)" + _STOP,
        r"(?:role|position)\s+(?:as)?\s*(.+?)" + _STOP,
    ]
    for p in patterns:
        m = re.search(p, cleaned, re.I)
        if m:
            candidate = (m.group(1) or "").strip()
            candidate = re.sub(r"\b(full[-\s]?time|part[-\s]?time|permanent|temporary|contract)\b", "", candidate, flags=re.I)
            candidate = candidate.strip()
            return candidate

    # try a synonym keyword presence
    for key in ROLE_SYNONYMS.keys():
        if key in cleaned:
            return key

    return ""


# -------------------------------------------------------------------
# Role normalization utilities
# -------------------------------------------------------------------


def normalize_role_for_api(role: str) -> str:
    role = (role or "").strip()
    if not role:
        return ""

    role = canonicalize_role(role)
    role = resolve_role_from_dataset(role) or role
    return build_search_keywords(role).strip()


async def normalize_role_with_api(role: str) -> str:
    role = (role or "").strip()
    if not role or len(role) < 2:
        return canonicalize_role(role)

    prompt = (
        "Clean and normalize this job role for a job search.\n"
        "- Remove words like 'job', 'jobs', 'position', 'role'.\n"
        "- Remove contract/time modifiers like full-time/part-time/seasonal/evening/night/weekend.\n"
        "- Return only the job title.\n"
        "- No quotes.\n\n"
        f"Text: {role}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        cleaned = (response.choices[0].message.content or "").strip().strip("\"'")
        return canonicalize_role(cleaned)
    except Exception:
        return canonicalize_role(role)


async def extract_dynamic_keywords(user_message: str) -> Dict[str, Any]:
    prompt = (
        "Return ONLY valid minified JSON.\n"
        "Keys: role, location, income_type, salary.\n"
        "If unknown, use null.\n"
        "Role must be ONLY the job title (no 'while', no extra plans).\n"
        f"Text: {user_message}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception:
        return {}


# -------------------------------------------------------------------
# Signal extraction
# -------------------------------------------------------------------


async def extract_signals(message: str, state: Dict[str, Any]) -> None:
    msg = (message or "").strip()
    low = msg.lower().strip()

    # 0) Small talk marker
    small_talk_keywords = {"thanks", "thank you", "ok", "okay", "cool", "nice", "helpful", "great"}
    if any(w in low for w in small_talk_keywords):
        state["last_small_talk"] = msg

    # 1) Income type
    explicit_income = normalize_income_type(low)
    if explicit_income:
        state["income_type"] = explicit_income

    # 2) Ask AI for role/location (but do not trust blindly)
    ai_role: Optional[str] = None
    ai_location: Optional[str] = None
    ai_income: Optional[str] = None
    ai_salary: Optional[str] = None

    ai_payload = await extract_dynamic_keywords(_strip_fillers(_drop_context_clauses(msg)))
    if isinstance(ai_payload, dict):
        ai_role = ai_payload.get("role")
        ai_location = ai_payload.get("location")
        ai_income = ai_payload.get("income_type")
        ai_salary = ai_payload.get("salary")

    if isinstance(ai_income, str) and ai_income.strip():
        # allow AI to set it too, but only if known keyword
        normalized = normalize_income_type(ai_income)
        if normalized:
            state["income_type"] = normalized

    # 3) Role candidate selection (prefer AI role if short + sane)
    role_candidate = ""
    if isinstance(ai_role, str):
        r = ai_role.strip()
        # guard: avoid whole sentence role
        if 1 <= len(r.split()) <= 6:
            role_candidate = r

    if not role_candidate:
        role_candidate = _extract_role_fallback(msg)

    # If still none, do NOT set role from entire message
    if role_candidate:
        role_candidate = _drop_context_clauses(role_candidate)
        role_canon = canonicalize_role(role_candidate)

        if role_canon:
            state["role_canon"] = role_canon
            state["role_display"] = map_role_synonym(role_canon)

            resolved = resolve_role_from_dataset(role_canon) or role_canon
            state["resolved_role"] = resolved
            state["role_query"] = build_search_keywords(resolved)

            # legacy/backwards compat
            state["role_keywords"] = state["role_display"]
            state["role_raw"] = role_canon

    # 4) Location candidate selection
    loc_candidate: Optional[str] = None
    if isinstance(ai_location, str) and ai_location.strip():
        loc_candidate = _best_city_match(ai_location)

    if not loc_candidate:
        loc_candidate = _extract_location_fallback(msg)

    if loc_candidate:
        state["location"] = loc_candidate

    # 5) Salary
    if isinstance(ai_salary, str) and ai_salary.strip():
        state["salary"] = ai_salary.strip()
    else:
        salary_match = re.search(
            r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?",
            low,
        )
        if salary_match:
            state["salary"] = salary_match.group(0)

    # 6) Advance phase
    advance_phase(state)
