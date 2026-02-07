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

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

_STOP: str = r"(?:\s+(?:in|near|based in|based)\b|[.,;!?]|$)"
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)

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
    t = (text or "").lower()
    fillers = [
        "i'm", "im", "i am",
        "looking", "looking for", "looking at",
        "i want", "i need",
        "find me", "search", "show me",
        "a", "an", "the",
        "job", "jobs", "role", "position", "work",
        "please", "thanks",
    ]
    for f in fillers:
        t = re.sub(rf"\b{re.escape(f)}\b", "", t)
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
        "- Remove time/contract modifiers (full-time/part-time/night/weekend/seasonal).\n"
        "- Return ONLY the job title.\n"
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
# Signal extraction (job search signals only)
# -------------------------------------------------------------------


async def extract_signals(message: str, state: Dict[str, Any]) -> None:
    low = (message or "").lower().strip()

    # 1) Income type
    explicit_income = normalize_income_type(low)
    if explicit_income:
        state["income_type"] = explicit_income

    # 2) AI extraction for role/location
    ai_role: Optional[str] = None
    ai_location: Optional[str] = None
    ai_income: Optional[str] = None
    ai_salary: Optional[str] = None

    try:
        ai_keywords = await extract_dynamic_keywords(_strip_fillers(message))
        ai_role = ai_keywords.get("role")
        ai_location = ai_keywords.get("location")
        ai_income = ai_keywords.get("income_type")
        ai_salary = ai_keywords.get("salary")
    except Exception:
        pass

    if isinstance(ai_income, str) and ai_income.strip():
        state["income_type"] = ai_income.strip().lower()

    if isinstance(ai_salary, str) and ai_salary.strip():
        state["salary"] = ai_salary.strip()

    # 3) Role candidate
    role_candidate: str = ""
    if isinstance(ai_role, str) and ai_role.strip():
        role_candidate = ai_role.strip()
    else:
        role_match = re.search(
            r"(?:work as|job as|be a|be an|looking for|i am a|i am an)\s+(.+?)" + _STOP,
            low,
            re.I,
        )
        if role_match:
            role_candidate = role_match.group(1)

    if not role_candidate:
        role_candidate = _strip_fillers(message)

    role_canon = canonicalize_role(role_candidate)

    if role_canon:
        state["role_canon"] = role_canon
        state["role_display"] = map_role_synonym(role_canon)

        resolved = resolve_role_from_dataset(role_canon) or role_canon
        state["resolved_role"] = resolved
        state["role_query"] = build_search_keywords(resolved)

        # Backwards compat
        state["role_keywords"] = state["role_display"]
        state["role_raw"] = role_canon

    # 4) Location
    if isinstance(ai_location, str) and ai_location.strip():
        loc = ai_location.strip().title()
    else:
        loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
        loc = loc_match.group(1).strip().title() if loc_match else None

    if loc:
        matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.7)
        state["location"] = matches[0] if matches else loc

    # 5) Salary fallback regex
    if "salary" not in state:
        salary_match = re.search(
            r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?",
            low,
        )
        if salary_match:
            state["salary"] = salary_match.group(0)
