# ai/extraction.py
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ai.client import client
from ai.role_resolver import build_search_keywords, canonicalize_role, resolve_role_from_dataset
from core.state_machine import advance_phase
from telemetry.logger import log_event

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

_STOP: str = r"(?:\s+(?:in|near|based in|based)\b|[.,;!?]|$)"
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)

# Add more cities as you see them in real usage (Dundee added).
UK_CITIES: List[str] = [
    "Edinburgh",
    "Glasgow",
    "Dundee",
    "Aberdeen",
    "St Andrews",
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
    "it support": "IT Support Technician",
    "support technician": "IT Support Technician",
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


def _first_clause(text: str) -> str:
    """
    Prevent “well water technician” style failures by focusing on the first search clause.
    Examples:
      "part time as a waiter in Edinburgh while I find..." -> take up to "while"
    """
    t = (text or "").strip()
    # Split on common “secondary intent” connectors
    parts = re.split(r"\b(while|then|but|however|until|and then|so that)\b", t, flags=re.I)
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


def _best_city_match(loc: str) -> str:
    loc = (loc or "").strip().title()
    if not loc:
        return ""
    matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.7)
    return matches[0] if matches else loc


def _extract_role_loc_regex(text: str) -> Tuple[str, str]:
    """
    Prefer patterns like:
      "part time as a waiter in Edinburgh"
      "looking for a waiter in Edinburgh"
      "waiter in Edinburgh"
    """
    low = (text or "").lower().strip()

    # role in location
    m = re.search(r"\b(?:as a|as an|work as|job as|be a|be an|looking for|find)\s+(.+?)\s+(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    m = re.search(r"\b(.+?)\s+(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    return ("", "")


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
        "- Remove filler like 'job', 'jobs', 'position', 'role'.\n"
        "- Remove time/contract modifiers like full-time/part-time/weekend/night/seasonal.\n"
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
# Signal extraction
# -------------------------------------------------------------------


async def extract_signals(message: str, state: Dict[str, Any]) -> None:
    raw = (message or "").strip()
    low = raw.lower().strip()

    # 0) Small talk marker
    if any(w in low for w in {"thanks", "thank you", "ok", "cool", "nice", "helpful", "great"}):
        state["last_small_talk"] = raw

    # 1) Income type
    explicit_income = normalize_income_type(low)
    if explicit_income:
        state["income_type"] = explicit_income

    # 2) Use first clause for search extraction
    primary = _first_clause(raw)
    primary_low = primary.lower()

    # 3) Regex first (fast + reliable)
    role_r, loc_r = _extract_role_loc_regex(primary)

    # 4) AI assist if regex is weak
    ai_role: Optional[str] = None
    ai_location: Optional[str] = None
    ai_income: Optional[str] = None

    try:
        ai_keywords = await extract_dynamic_keywords(_strip_fillers(primary))
        ai_role = ai_keywords.get("role")
        ai_location = ai_keywords.get("location")
        ai_income = ai_keywords.get("income_type")
    except Exception:
        pass

    # Apply AI income if present
    if isinstance(ai_income, str) and ai_income.strip() and not state.get("income_type"):
        inc = normalize_income_type(ai_income) or ai_income.strip().lower()
        if inc in STANDARD_INCOME_TYPES:
            state["income_type"] = inc

    # 5) Choose role candidate
    role_candidate = ""
    if role_r:
        role_candidate = role_r
        role_source = "regex"
    elif isinstance(ai_role, str) and ai_role.strip():
        role_candidate = ai_role.strip()
        role_source = "ai"
    else:
        # last resort: stripped fillers, but DON'T let it swallow the whole sentence
        role_candidate = _strip_fillers(primary)
        role_source = "fallback"

    # Defensive: if role candidate still looks like a whole sentence, bail.
    if len(role_candidate.split()) > 6 and role_source == "fallback":
        role_candidate = ""

    # Canonicalize
    role_canon = canonicalize_role(role_candidate) if role_candidate else ""

    if role_canon:
        state["role_canon"] = role_canon
        state["role_display"] = map_role_synonym(role_canon)

        resolved = resolve_role_from_dataset(role_canon) or role_canon
        state["resolved_role"] = resolved
        state["role_query"] = build_search_keywords(resolved)

        # legacy
        state["role_keywords"] = state["role_display"]
        state["role_raw"] = role_canon

    # 6) Location
    loc_candidate = ""
    if loc_r:
        loc_candidate = loc_r
        loc_source = "regex"
    elif isinstance(ai_location, str) and ai_location.strip():
        loc_candidate = ai_location.strip()
        loc_source = "ai"
    else:
        m = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, primary_low, re.I)
        loc_candidate = m.group(1).strip() if m else ""
        loc_source = "fallback"

    if loc_candidate:
        state["location"] = _best_city_match(loc_candidate)

    # 7) Salary
    salary_match = re.search(r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?", low)
    if salary_match:
        state["salary"] = salary_match.group(0)

    # 8) Store secondary intent (optional): what they said after "while..."
    if primary != raw:
        tail = raw[len(primary):].strip()
        if tail:
            state["note"] = tail[:200]

    # 9) Advance phase
    advance_phase(state)

    # telemetry (no raw message)
    log_event(
        "signals_extracted",
        {
            "role_source": role_source if role_candidate else "none",
            "loc_source": loc_source if loc_candidate else "none",
            "role_canon": state.get("role_canon"),
            "role_display": state.get("role_display"),
            "location": state.get("location"),
            "income_type": state.get("income_type"),
            "has_note": bool(state.get("note")),
        },
        user_id=str(state.get("user_id") or ""),
    )
