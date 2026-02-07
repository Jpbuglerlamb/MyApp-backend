# ai/extraction.py
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

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
_STOP: str = r"(?:\s+(?:in|near|based in|based|at)\b|[.,;!?]|$)"
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)

# Add Dundee (you were testing Dundee)
UK_CITIES: List[str] = [
    "Edinburgh",
    "Glasgow",
    "Dundee",
    "Aberdeen",
    "London",
    "Manchester",
    "Bristol",
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
    "marketing": "Marketing Assistant",
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
    # Keep this conservative; over-stripping causes weird role guesses.
    fillers = [
        "i'm", "im", "i am",
        "looking for", "looking", "find me", "search", "show me",
        "please", "thanks", "thank you",
        "a", "an", "the",
        "job", "jobs", "role", "position",
    ]
    for f in fillers:
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _clip_context(text: str) -> str:
    """
    If user writes a long explanation, keep the first clause that typically contains role/location.
    Prevents 'well water technician' style accidents.
    """
    t = (text or "").strip()
    # Split on common narrative connectors that come after the “core ask”
    parts = re.split(r"\b(while|until|then|so that|so i can|because)\b", t, flags=re.I)
    return (parts[0] or t).strip()


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


def _closest_city(raw: str) -> Optional[str]:
    if not raw:
        return None
    loc = raw.strip().title()
    matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.70)
    return matches[0] if matches else loc


def _rule_extract_role_location(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Rule-based extraction first. This catches:
    - "waiter in Edinburgh"
    - "part time as a waiter in Edinburgh"
    - "marketing in Dundee"
    """
    low = (text or "").lower().strip()

    # location: "in X", "near X", "based in X"
    loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
    loc = loc_match.group(1).strip() if loc_match else None

    # role patterns (capture role before location or inside "as a/an")
    role = None

    # "as a waiter", "as an assistant"
    m_as = re.search(r"\b(?:as)\s+(?:a|an)\s+(.+?)" + _STOP, low, re.I)
    if m_as:
        role = m_as.group(1).strip()

    # "job as X"
    if not role:
        m_jobas = re.search(r"\b(?:job|work)\s+(?:as)\s+(.+?)" + _STOP, low, re.I)
        if m_jobas:
            role = m_jobas.group(1).strip()

    # "X in Edinburgh" (role before location)
    if not role and loc:
        # Take words before "in <loc>"
        m_role_in = re.search(r"^(.+?)\s+\b(?:in|near|based in|based)\b", low, re.I)
        if m_role_in:
            candidate = m_role_in.group(1).strip()
            # Remove obvious filler at start
            candidate = re.sub(r"^(find|search|look for|can you find)\s+", "", candidate).strip()
            role = candidate or None

    return (role, loc)


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
        "Return only a clean job title for a job search.\n"
        "- Remove filler words.\n"
        "- Remove contract/time modifiers (full-time/part-time/weekend/evening).\n"
        "- No quotes.\n"
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
# Signal extraction
# -------------------------------------------------------------------
async def extract_signals(message: str, state: Dict[str, Any]) -> None:
    raw = (message or "").strip()
    low = raw.lower().strip()

    # 0) Income type (explicit)
    explicit_income = normalize_income_type(low)
    if explicit_income:
        state["income_type"] = explicit_income

    # 1) First: rule-based extraction on a clipped version of the message
    clipped = _clip_context(raw)
    role_rb, loc_rb = _rule_extract_role_location(clipped)

    if loc_rb:
        state["location"] = _closest_city(loc_rb)

    # If we got a role from rules, use it (prevents “well water technician” accidents)
    role_candidate = role_rb

    # 2) If rules didn't find role or location, use AI extraction as fallback
    ai_role: Optional[str] = None
    ai_location: Optional[str] = None
    if not role_candidate or (not loc_rb):
        try:
            ai_keywords = await extract_dynamic_keywords(_strip_fillers(clipped))
            ai_role = ai_keywords.get("role")
            ai_location = ai_keywords.get("location")
        except Exception:
            ai_role, ai_location = None, None

        if not role_candidate and isinstance(ai_role, str) and ai_role.strip():
            role_candidate = ai_role.strip()

        if not loc_rb and isinstance(ai_location, str) and ai_location.strip():
            state["location"] = _closest_city(ai_location.strip())

    # 3) If still no role, attempt a last-resort heuristic (very conservative)
    if not role_candidate:
        # Example: "marketing in Dundee" -> "marketing"
        m = re.search(r"^(.+?)\s+\b(?:in|near|based in|based)\b", _strip_fillers(clipped), re.I)
        if m:
            role_candidate = m.group(1).strip()

    # 4) Normalize role fields
    if role_candidate:
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

    # 5) Salary
    salary_match = re.search(
        r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?",
        low,
    )
    if salary_match:
        state["salary"] = salary_match.group(0)

    # 6) Advance phase
    advance_phase(state)

