# ai/intent_router.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ai.client import client

INTENTS = {
    "JOB_SEARCH",
    "CLARITY",
    "SIDE_HUSTLE",
    "SKILLS_PROFILE",
    "APPLICATION_HELP",
    "CHAT",
}

# Fast heuristic fallback (no API call)
_SIDE_HUSTLE_RE = re.compile(r"\b(side hustle|extra money|make money|gig|fiverr|upwork|freelance|etsy|ebay)\b", re.I)
_SKILLS_RE = re.compile(r"\b(what am i good at|my skills|my experience|strengths|portfolio)\b", re.I)
_APP_HELP_RE = re.compile(r"\b(cv|resume|cover letter|interview|apply|application)\b", re.I)
_CLARITY_RE = re.compile(r"\b(confused|lost|stuck|unsure|not sure|don[’']?t know|no idea|future|career)\b", re.I)
_CHAT_RE = re.compile(r"^\s*(thanks|thank you|ok|okay|cool|nice|great)\s*[.!?]?\s*$", re.I)


async def classify_intent(
    text: str,
    *,
    state: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns dict:
      { intent: str, confidence: float, signals: dict, why: str }
    signals can include: { skills, constraints, goal, time_per_week, preferred_mode }
    """
    t = (text or "").strip()
    low = t.lower()

    # Heuristic first (cheap + reliable)
    if _CHAT_RE.match(low):
        return {"intent": "CHAT", "confidence": 0.95, "signals": {}, "why": "ack/smalltalk"}

    if _SIDE_HUSTLE_RE.search(low):
        return {"intent": "SIDE_HUSTLE", "confidence": 0.85, "signals": {}, "why": "side_hustle_keywords"}

    if _SKILLS_RE.search(low):
        return {"intent": "SKILLS_PROFILE", "confidence": 0.85, "signals": {}, "why": "skills_keywords"}

    if _APP_HELP_RE.search(low):
        return {"intent": "APPLICATION_HELP", "confidence": 0.80, "signals": {}, "why": "application_keywords"}

    if _CLARITY_RE.search(low):
        # If they already gave role+location, it might just be chat around the search
        have_role = bool((state or {}).get("role_canon") or (state or {}).get("role_raw"))
        have_loc = bool((state or {}).get("location"))
        if have_role or have_loc:
            return {"intent": "CLARITY", "confidence": 0.65, "signals": {}, "why": "reflective_with_context"}
        return {"intent": "CLARITY", "confidence": 0.85, "signals": {}, "why": "reflective_keywords"}

    # If it contains job-y structure, assume job search
    if any(x in low for x in [" in ", "near ", "based in ", "full-time", "part-time", "jobs", "role", "developer", "waiter", "driver"]):
        return {"intent": "JOB_SEARCH", "confidence": 0.60, "signals": {}, "why": "jobish_text"}

    # LLM router for messy text (only when heuristics didn't hit)
    prompt = (
        "Return ONLY minified JSON.\n"
        "Choose intent from: JOB_SEARCH, CLARITY, SIDE_HUSTLE, SKILLS_PROFILE, APPLICATION_HELP, CHAT.\n"
        "Also return optional signals: skills(list), constraints(list), goal(str), time_per_week(str), preferred_mode(str: online/local/both).\n"
        "Format: {\"intent\":\"...\",\"confidence\":0.0-1.0,\"signals\":{...},\"why\":\"...\"}\n\n"
        f"Text: {t}"
    )

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)

        intent = (data.get("intent") or "CHAT").strip().upper()
        if intent not in INTENTS:
            intent = "CHAT"

        conf = float(data.get("confidence") or 0.5)
        signals = data.get("signals") or {}
        why = data.get("why") or "llm_router"

        return {"intent": intent, "confidence": conf, "signals": signals, "why": why}
    except Exception:
        return {"intent": "CHAT", "confidence": 0.50, "signals": {}, "why": "router_fallback"}
