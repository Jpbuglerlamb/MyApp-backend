# ai/intent.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Intent:
    name: str  # "job_search" | "reflective" | "side_hustle" | "smalltalk" | "unknown"
    confidence: float
    debug: Optional[Dict[str, Any]] = None


# NOTE: Keep regex conservative. Extraction handles details.
_RE_JOB = re.compile(
    r"\b(find|search|look for|apply|applications|openings|vacancies|listings|hire|hiring|role|job|work)\b",
    re.I,
)
_RE_LOCATION_HINT = re.compile(r"\b(in|near|around|based in)\s+[a-zA-Z]", re.I)
_RE_REFLECTIVE = re.compile(
    r"\b(confused|lost|unsure|stuck|anxious|stressed|burnt out|burned out|future|life|career path|direction)\b",
    re.I,
)
_RE_SIDE_HUSTLE = re.compile(
    r"\b(side hustle|extra income|make money|earn more|freelance|fiverr|upwork|etsy|deliveroo|uber|just eat)\b",
    re.I,
)
_RE_SMALLTALK = re.compile(r"^\s*(thanks|thank you|ok|okay|cool|nice|great|legend)\s*[.!?]?\s*$", re.I)


def detect_intent(message: str, state: Dict[str, Any] | None = None) -> Intent:
    """
    Lightweight intent detector. No LLM required.
    Uses state as a tie-breaker (e.g., role/location already known).
    """
    msg = (message or "").strip()
    low = msg.lower()

    if not msg:
        return Intent("unknown", 0.0)

    if _RE_SMALLTALK.match(low):
        return Intent("smalltalk", 0.95)

    # Side hustle has priority over generic "job" words
    if _RE_SIDE_HUSTLE.search(low):
        return Intent("side_hustle", 0.85, debug={"matched": "side_hustle"})

    # Reflective
    if _RE_REFLECTIVE.search(low):
        # If they clearly also want listings (job words + location hint), treat as job search
        if _RE_JOB.search(low) and _RE_LOCATION_HINT.search(low):
            return Intent("job_search", 0.70, debug={"tie_break": "reflective+job+location"})
        return Intent("reflective", 0.80, debug={"matched": "reflective"})

    # Job search
    if _RE_JOB.search(low):
        return Intent("job_search", 0.70, debug={"matched": "job"})

    return Intent("unknown", 0.20)
