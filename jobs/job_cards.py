# jobs/job_cards.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

def _tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return {t for t in s.split() if t and len(t) > 2}

def score_job(job: Dict[str, Any], *, role_canon: str, location: str, income_type: Optional[str] = None) -> Dict[str, Any]:
    title = (job.get("title") or "")
    company = (job.get("company") or "")
    loc = (job.get("location") or "")

    role_toks = _tokens(role_canon)
    title_toks = _tokens(title)
    overlap = len(role_toks & title_toks)

    reasons: List[str] = []
    missing: List[str] = []

    score = 50  # base

    if overlap >= 2:
        score += 20
        reasons.append("Title matches your role keywords.")
    elif overlap == 1:
        score += 10
        reasons.append("Partial match to your role keywords.")
    else:
        score -= 10
        missing.append("Title doesn’t strongly match your role keywords.")

    if location and location.lower() in loc.lower():
        score += 15
        reasons.append("Location matches your preference.")
    else:
        score += 0  # neutral

    if company:
        score += 2  # tiny bonus for complete data

    score = max(0, min(100, score))

    # Recommended action is simple for now
    action = "Apply" if score >= 70 else ("Consider" if score >= 55 else "Skip")

    return {
        "score": score,
        "reasons": reasons[:2],
        "missing": missing[:2],
        "action": action,
    }

def to_job_cards(
    jobs: List[Dict[str, Any]],
    *,
    role_canon: str,
    location: str,
    income_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for j in jobs or []:
        meta = score_job(j, role_canon=role_canon, location=location, income_type=income_type)
        cards.append({
            "id": j.get("id"),
            "title": j.get("title"),
            "company": j.get("company"),
            "location": j.get("location"),
            "redirect_url": j.get("redirect_url"),
            **meta
        })

    cards.sort(key=lambda x: x.get("score", 0), reverse=True)
    return cards
