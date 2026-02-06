# ai/role_resolver.py
import json
import os
import re
from functools import lru_cache
from typing import Optional, List, Set, Dict


# -----------------------------
# Dataset path + loading
# -----------------------------
def _dataset_path() -> str:
    env_path = os.getenv("JOB_TITLES_DATASET", "").strip()
    if env_path:
        return os.path.abspath(env_path)

    here = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(here, "..", "data", "job_titles_dataset.json")),
        os.path.abspath(os.path.join(here, "..", "Data", "job_titles_dataset.json")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


# -----------------------------
# Canonicalization helpers
# -----------------------------
FILLER_WORDS: Set[str] = {
    "job", "jobs", "role", "position", "work", "working",
    "in", "at", "for", "a", "an", "the", "as",
    "looking", "searching", "find", "me", "please",
    "i", "im", "i'm", "am", "want", "need",
}

# “Not part of the title” for your product use-case
TIME_WORDS: Set[str] = {
    "evening", "night", "overnight",
    "weekend", "weekends", "weekday", "weekdays",
    "seasonal", "temporary", "temp",
    "casual", "permanent",
    "contract", "agency", "bank",
    "shift", "shifts",
    "zero", "hour", "hours",
    "part", "full", "time",
}

TIME_PHRASES: List[str] = [
    "part time", "part-time",
    "full time", "full-time",
    "zero hours", "zero-hour", "zero-hours",
]

BAD_ROLE_KEYWORDS: Set[str] = {
    "a job", "job", "jobs", "work", "position", "role", "career", "employment",
}

# Optional normalization to reduce synonyms in the *matching* space
# (keep conservative; your extraction layer can decide display roles)
ROLE_NORMALIZE_MAP: Dict[str, str] = {
    "waitress": "waiter",
    "waiting staff": "waiter",
    # "server": "waiter",  # uncomment if you want dataset matching to treat "server" as waiter
}


def _clean_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    s = _clean_text(s)
    return [t for t in s.split() if t]


def strip_fillers(text: str) -> str:
    toks = _tokenize(text)
    toks = [t for t in toks if t not in FILLER_WORDS]
    return " ".join(toks).strip()


def strip_time_modifiers(text: str) -> str:
    """
    Remove schedule/contract modifiers from a string (job-only output).

    Examples:
      "Evening Waiter" -> "waiter"
      "Part-time waiting staff" -> "waiting staff"
      "Weekend barista" -> "barista"
      "Zero hours bartender" -> "bartender"
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    s = _clean_text(text)

    # remove multi-word phrases first
    for ph in TIME_PHRASES:
        s = re.sub(rf"\b{re.escape(ph)}\b", " ", s)

    # remove generic "job" words + contract/time words
    for bad in sorted(BAD_ROLE_KEYWORDS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(bad)}\b", " ", s)

    toks = [t for t in s.split() if t and t not in TIME_WORDS]
    s = " ".join(toks).strip()

    return re.sub(r"\s+", " ", s).strip()


def _apply_role_normalization(s: str) -> str:
    if not s:
        return ""
    low = s.lower()
    # phrase replacements (simple contains replace; safe for your dataset scale)
    for k, v in ROLE_NORMALIZE_MAP.items():
        if k in low:
            low = low.replace(k, v)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def canonicalize_role(text: str) -> str:
    """
    The one true pipeline used by BOTH extraction and dataset matching.
    Output is job-only, lowercase, no time/contract modifiers.
    """
    s = text or ""
    s = strip_fillers(s)
    s = strip_time_modifiers(s)
    s = _apply_role_normalization(s)
    s = _clean_text(s)
    return s.strip()


@lru_cache(maxsize=1)
def _load_titles_raw() -> List[str]:
    path = _dataset_path()
    if not os.path.exists(path):
        print(f"[WARN] job titles dataset not found at: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load job titles dataset at {path}: {e}")
        return []

    titles: List[str] = []
    cats = data.get("categories") or {}
    if isinstance(cats, dict):
        for _, arr in cats.items():
            if isinstance(arr, list):
                titles.extend([x for x in arr if isinstance(x, str)])

    return [t for t in titles if t and t.strip()]


@lru_cache(maxsize=1)
def _load_titles_canonical() -> List[str]:
    raw = _load_titles_raw()
    canon = set()
    for t in raw:
        ct = canonicalize_role(t)
        if ct:
            canon.add(ct)
    return sorted(canon)


# -----------------------------
# Public API
# -----------------------------
def resolve_role_from_dataset(role_raw: str) -> Optional[str]:
    """
    Return canonical job-only role from dataset, or None.
    """
    titles = _load_titles_canonical()
    if not titles:
        return None

    query = canonicalize_role(role_raw)
    if not query:
        return None

    # 1) exact
    if query in titles:
        return query

    # 2) query contained in title
    contains_query = [t for t in titles if query in t]
    if contains_query:
        contains_query.sort(key=len)
        return contains_query[0]

    # 3) title contained in query
    contained_by_query = [t for t in titles if t in query]
    if contained_by_query:
        contained_by_query.sort(key=len, reverse=True)
        return contained_by_query[0]

    # 4) token overlap
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return None

    best_title = None
    best_score = 0
    best_len = 10**9

    for t in titles:
        t_tokens = set(_tokenize(t))
        if not t_tokens:
            continue
        score = len(q_tokens & t_tokens)
        if score > best_score or (score == best_score and score > 0 and len(t) < best_len):
            best_score = score
            best_title = t
            best_len = len(t)

    if best_score >= 1:
        return best_title

    # 5) prefix fallback
    for t in titles:
        t_tokens = _tokenize(t)
        for qt in q_tokens:
            if len(qt) >= 3 and any(tok.startswith(qt) for tok in t_tokens):
                return t

    return None


def build_search_keywords(canonical_role_or_title: str) -> str:
    """
    Build a safe, boring search string for Adzuna 'what'.

    - No boolean operators
    - No schedule/contract modifiers
    - Space-separated keywords only
    """
    s = canonicalize_role(canonical_role_or_title)

    # optional broadening (space-separated, not OR)
    if s == "waiter":
        return "waiter waitress waiting staff server front of house"
    if s == "bartender":
        return "bartender bar staff bar attendant"
    if s == "barista":
        return "barista coffee"
    if s == "chef":
        return "chef cook kitchen"

    return s