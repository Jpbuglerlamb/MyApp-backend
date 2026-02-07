# ai/extraction.py
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ai.client import client
from ai.role_resolver import build_search_keywords, canonicalize_role, resolve_role_from_dataset
from core.state_machine import advance_phase

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
_STOP: str = r"(?:\s+(?:in|near|based in|based)\b|[.,;!?]|$)"
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)

UK_CITIES: List[str] = [
    "Edinburgh", "London", "Manchester", "Bristol", "Glasgow", "Leeds", "Liverpool", "Belfast", "Cardiff"
]

STANDARD_INCOME_TYPES: Dict[str, List[str]] = {
    "full-time": ["full time", "full-time", "permanent"],
    "part-time": ["part time", "part-time", "casual", "zero hour", "zero-hours", "zero hours"],
    "temporary": ["temporary", "temp", "short-term"],
    "freelance": ["freelance", "gig", "self-employed", "contract"],
    "internship": ["intern", "internship", "trainee"],
}

ROLE_SYNONYMS: Dict[str, str] = {
    # Tech
    "app development": "Software Developer",
    "software engineer": "Software Developer",
    "web developer": "Software Developer",
    "frontend": "Frontend Developer",
    "backend": "Backend Developer",
    "ux": "UX Designer",
    "ui": "UI Designer",
    "data analyst": "Data Analyst",
    "data scientist": "Data Scientist",
    "mobile developer": "Mobile Developer",
    "qa": "QA Tester",
    "it support": "IT Support",
    "support technician": "IT Support",
    # Hospitality
    "waiter": "Waiter",
    "waitress": "Waiter",
    "waiting staff": "Waiter",
    "server": "Waiter",
    "bar staff": "Bartender",
    "bartender": "Bartender",
    "barista": "Barista",
    "chef": "Chef",
    "cook": "Chef",
    # Other
    "teacher": "Teacher",
    "driver": "Driver",
    "delivery driver": "Driver",
}

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
    Remove some filler words, but DON'T over-strip.
    Over-stripping is how you end up with role='edinburgh' etc.
    """
    t = (text or "").lower()

    # keep “a/an/the” because removing them can merge phrases weirdly
    fillers = [
        "i'm", "im", "i am",
        "looking for", "looking", "i want", "i need",
        "find me", "search", "show me",
        "job", "jobs", "role", "position", "work",
        "please", "thanks", "thank you",
    ]
    for f in fillers:
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)

    # normalize whitespace
    return re.sub(r"\s+", " ", t).strip()


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


def _looks_like_city(text: str) -> bool:
    if not text:
        return False
    t = text.strip().title()
    return t in UK_CITIES


def _best_role_from_text(message: str) -> Optional[str]:
    """
    Strong heuristic: if message mentions a known role keyword, prefer it.
    Fixes: "well part time as a waiter in Edinburgh..." becoming "well water technician".
    """
    low = (message or "").lower()
    # pick the longest matching key to avoid 'server' inside something else
    matches: List[Tuple[int, str]] = []
    for k in ROLE_SYNONYMS.keys():
        if k in low:
            matches.append((len(k), k))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


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
        "- Remove words like job/jobs/position/role.\n"
        "- Remove contract/time modifiers like full-time/part-time/seasonal/night/weekend.\n"
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
    """
    LLM extraction as a *helper*, never the only source of truth.
    """
    prompt = (
        "Return ONLY valid minified JSON.\n"
        "Keys: role, location, income_type, salary.\n"
        "If unknown, use null.\n"
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
    msg = message or ""
    low = msg.lower().strip()

    # 0) Income type (cheap + reliable)
    explicit_income = normalize_income_type(low)
    if explicit_income:
        state["income_type"] = explicit_income

    # 1) Location (regex first)
    loc: Optional[str] = None
    loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
    if loc_match:
        loc = loc_match.group(1).strip().title()
    else:
        # also accept “Edinburgh” alone as location
        if low.title() in UK_CITIES:
            loc = low.title()

    if loc:
        matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.7)
        state["location"] = matches[0] if matches else loc

    # 2) Role (heuristics first)
    role_candidate: Optional[str] = None

    # a) known-role keyword wins
    best = _best_role_from_text(msg)
    if best:
        role_candidate = best

    # b) common “as a/an …” patterns
    if not role_candidate:
        m = re.search(r"\b(?:as a|as an|work as|job as|be a|be an)\s+(.+?)" + _STOP, low, re.I)
        if m:
            role_candidate = m.group(1).strip()

    # c) LLM extraction as fallback
    ai_role: Optional[str] = None
    ai_location: Optional[str] = None
    if not role_candidate:
        try:
            ai = await extract_dynamic_keywords(_strip_fillers(msg))
            ai_role = ai.get("role")
            ai_location = ai.get("location")
            if isinstance(ai_location, str) and ai_location.strip() and not state.get("location"):
                loc2 = ai_location.strip().title()
                matches = difflib.get_close_matches(loc2, UK_CITIES, n=1, cutoff=0.7)
                state["location"] = matches[0] if matches else loc2
            if isinstance(ai_role, str) and ai_role.strip():
                role_candidate = ai_role.strip()
        except Exception:
            pass

    # d) last resort: cleaned text, but do NOT accept city-as-role
    if not role_candidate:
        role_candidate = _strip_fillers(msg)

    role_canon = canonicalize_role(role_candidate or "")

    # Guard: don't let a city become a role
    if role_canon and _looks_like_city(role_canon):
        role_canon = ""

    if role_canon:
        state["role_canon"] = role_canon
        state["role_display"] = map_role_synonym(role_canon)

        resolved = resolve_role_from_dataset(role_canon) or role_canon
        state["resolved_role"] = resolved
        state["role_query"] = build_search_keywords(resolved)

        # legacy/back-compat
        state["role_keywords"] = state["role_display"]
        state["role_raw"] = role_canon

    # 3) Salary
    salary_match = re.search(r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?", low)
    if salary_match:
        state["salary"] = salary_match.group(0)

    # 4) Advance phase
    advance_phase(state)
