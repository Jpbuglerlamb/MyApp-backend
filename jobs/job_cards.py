# jobs/job_cards.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# -------------------------------------------------------------------
# Tokenization + relevance scoring for swipeable job cards
# -------------------------------------------------------------------

# Expand role families into “meaningful” keyword sets for matching.
# Keep conservative and product-oriented (ranking, not search).
ROLE_KEYWORD_EXPANSIONS: Dict[str, Set[str]] = {
    # PPC / Paid Search family
    "ppc": {"ppc", "paid", "search", "adwords", "google", "ads", "sem", "performance", "acquisition"},
    "google ads": {"ppc", "paid", "search", "adwords", "google", "ads", "sem", "performance", "acquisition"},
    "paid search": {"ppc", "paid", "search", "adwords", "google", "ads", "sem", "performance", "acquisition"},
    "adwords": {"ppc", "paid", "search", "adwords", "google", "ads", "sem", "performance", "acquisition"},
}


def _tokens(text: str) -> Set[str]:
    """
    Lightweight tokenizer:
      - lowercase
      - strip punctuation
      - keep tokens length >= 3 (reduces noise like 'in', 'to', 'of')
    """
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return {t for t in s.split() if t and len(t) >= 3}


def _expanded_role_tokens(role_canon: str) -> Set[str]:
    """
    Expand a canonical role into a broader token set based on known role families.
    Example:
      role_canon='google ads executive' -> adds PPC/SEM/acquisition/performance tokens.
    """
    base = _tokens(role_canon)
    low = (role_canon or "").lower()

    expanded = set(base)
    for key, extra in ROLE_KEYWORD_EXPANSIONS.items():
        if key in low:
            expanded |= set(extra)

    return expanded


def _strict_match_required(role_canon: str) -> bool:
    """
    If the user asked for certain specialist roles (PPC/Google Ads),
    require at least one strong marker in the job title/snippet to rank high.
    """
    low = (role_canon or "").lower()
    return any(k in low for k in ("ppc", "google ads", "paid search", "adwords", "sem"))


def _strict_match_hit(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in ("ppc", "google ads", "paid search", "adwords", "sem"))


def score_job(
    job: Dict[str, Any],
    *,
    role_canon: str,
    location: str,
    income_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns scoring metadata:
      - score: 0..100
      - reasons/missing: small strings for UI
      - action: Apply/Consider/Skip
    """
    title = (job.get("title") or "")
    company = (job.get("company") or "")
    loc = (job.get("location") or "")

    # If you have snippet/description from the upstream provider, include it.
    snippet = (job.get("description") or job.get("snippet") or "")

    role_toks = _expanded_role_tokens(role_canon)
    text_toks = _tokens(f"{title} {snippet}")
    overlap = len(role_toks & text_toks)

    reasons: List[str] = []
    missing: List[str] = []

    score = 50  # base

    # 1) Role relevance
    if overlap >= 3:
        score += 25
        reasons.append("Strong match to your role keywords.")
    elif overlap == 2:
        score += 18
        reasons.append("Good match to your role keywords.")
    elif overlap == 1:
        score += 8
        reasons.append("Partial match to your role keywords.")
    else:
        score -= 12
        missing.append("Doesn’t strongly match your role keywords.")

    # 1b) Specialist strictness (prevents “Growth Marketing Manager” dominating PPC searches)
    if _strict_match_required(role_canon) and not _strict_match_hit(f"{title} {snippet}"):
        score -= 25
        missing.append("Looks more like a related role than a direct match.")

    # 2) Location preference
    if location and location.lower() in (loc or "").lower():
        score += 15
        reasons.append("Location matches your preference.")

    # 3) Data completeness bonus
    if company:
        score += 2

    # 4) Clamp
    score = max(0, min(100, score))

    # 5) Action suggestion (simple heuristic)
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
    """
    Convert raw jobs into card payloads and sort by score descending.
    """
    cards: List[Dict[str, Any]] = []
    for j in (jobs or []):
        meta = score_job(j, role_canon=role_canon, location=location, income_type=income_type)
        cards.append(
            {
                "id": j.get("id"),
                "title": j.get("title"),
                "company": j.get("company"),
                "location": j.get("location"),
                "redirect_url": j.get("redirect_url"),
                **meta,
            }
        )

    cards.sort(key=lambda x: x.get("score", 0), reverse=True)
    return cards

