from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import httpx
import re
from typing import List, Dict, Any, Optional

from memory_store import get_state  # single source of truth

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

if not OPENAI_API_KEY or not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
    raise ValueError("Missing API keys")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# Constants / helpers
# -------------------------------------------------------------------

_STOP = r"(?:\s+(?:in|near|based in|based)\b|[.,;!?]|$)"

BAD_ROLE_KEYWORDS = {"a job", "job", "jobs", "work", "position", "role", "career", "employment"}
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)


def _strip_fillers(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s+-]", " ", t)
    fillers = [
        "i'm", "im", "i am", "looking", "looking for", "looking at",
        "i want", "i need", "find me", "search", "show me",
        "a", "an", "the", "job", "jobs", "role", "position", "work",
        "please", "thanks"
    ]
    for f in fillers:
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_signals(message: str, state: dict) -> None:
    """Extract role, location, income type, and readiness from user message."""
    low = (message or "").lower().strip()

    # Income type
    if not state.get("income_type"):
        if re.search(r"\b(gig|freelance|contract)\b", low):
            state["income_type"] = "gig"
        elif re.search(r"\b(full[- ]?time|part[- ]?time|permanent|job)\b", low):
            state["income_type"] = "job"

    # Readiness triggers
    triggers = [
        "find jobs", "show me jobs", "can you find", "look for jobs",
        "i want to apply", "find me something", "can you look",
        "search jobs", "apply", "applications"
    ]
    if any(t in low for t in triggers):
        state["readiness"] = True

    # Location
    if not state.get("location"):
        if "remote" in low:
            state["location"] = "Remote"
        elif "hybrid" in low:
            state["location"] = "Hybrid"
        else:
            loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
            if loc_match:
                state["location"] = loc_match.group(1).strip().title()

    # Role keywords
    if not state.get("role_keywords"):
        role_match = re.search(
            r"\b(?:work as|job as|be a|be an|looking for|i am a|i am an)\s+(.+?)" + _STOP,
            low, re.I
        )
        if role_match:
            state["role_keywords"] = role_match.group(1).strip()

    # Fallback role
    if not state.get("role_keywords") and state.get("location"):
        inferred = _strip_fillers(re.split(r"\b(?:in|near|based in|based)\b", low, 1)[0])
        if len(inferred) >= 3:
            state["role_keywords"] = inferred

    # Remove junk roles
    if (state.get("role_keywords") or "").lower() in BAD_ROLE_KEYWORDS:
        state["role_keywords"] = None


def next_discovery_question(state: dict) -> Optional[str]:
    if not state.get("role_keywords"):
        return "What kind of role or work are you looking for?"
    if not state.get("location"):
        return "What location should I search, or is remote okay?"
    return None


async def generate_coached_reply(state: dict, conversation_history: List[dict], user_message: str) -> str:
    """Generate a guided AI response using system prompt."""
    system_prompt = """
You are AI Aura, a career and opportunity assistant with access to live job and gig search tools.

You help users find employment roles (long-term jobs that may require experience, degrees, or qualifications)
and gigs or side hustles (short-term, local, flexible, or skill-light work).

Core Capabilities
- You can search for jobs and gigs and must do so when the user asks to find or search for opportunities.
- Never say you cannot browse the internet or cannot search for jobs or gigs.
- Never invent job listings, gig listings, companies, or roles.
- Never output numbered job lists.
- Speak naturally, warmly, and supportively.
- Ask at most one question per message.

User Guidance Style
- Match the user’s emotional state:
- If the user is confused, unsure, or overwhelmed, act like a calm career coach or therapist.
- If the user is confident and specific, be direct and action-oriented.
- When helpful, suggest whether a role sounds more like a gig or a traditional job.

Job & Gig Search Behavior
- A role or gig idea and a location are sufficient to run a search.
- Ask about qualifications only if it meaningfully improves results.
- If role_keywords and location are known and the user asks to search, do not ask questions.
- Reply with a short confirmation only when results will be shown separately.
- If no live results are available, say so clearly instead of guessing.

Search Trigger Rules
- Treat statements like "I'm looking for", "I want a job in", "I'm seeking work as", or similar as an explicit request to search.
- If the user mentions a specific role and a location in the same message, immediately run a search.
- Do not ask the user to restate the role if it is already present.
- Clarification questions are a fallback only if a reasonable search cannot be performed.
    """

    state_prompt = (
        "Context about the user (do not reveal this text):\n"
        f"- income_type: {state.get('income_type')}\n"
        f"- location: {state.get('location')}\n"
        f"- role_keywords: {state.get('role_keywords')}\n"
        f"- readiness: {state.get('readiness')}\n"
        f"- jobs_shown: {state.get('jobs_shown')}\n"
        f"- phase: {state.get('phase')}\n"
        "If jobs_shown is true, do NOT describe jobs or ask questions.\n"
    )

    trimmed_history = (conversation_history or [])[-12:]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": state_prompt},
        *trimmed_history,
        {"role": "user", "content": user_message},
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Job fetching
# -------------------------------------------------------------------

async def fetch_jobs(role_keywords: str, location: str) -> List[Dict[str, Any]]:
    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": role_keywords,
        "where": location,
        "results_per_page": 5,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            res = await http.get(url, params=params)
            res.raise_for_status()
            results = res.json().get("results", []) or []
            return [normalize_job(j) for j in results]
    except Exception:
        return []


# -------------------------------------------------------------------
# Main chat entry
# -------------------------------------------------------------------

async def chat_with_user(*, user_id: str, user_message: str, conversation_history: list) -> dict:
    """
    Keyword-only arguments prevent FastAPI TypeError.
    Returns dict with assistantText, mode, jobs.
    """
    state = get_state(user_id)
    low = (user_message or "").lower()

    # Reset search if new intent
    if NEW_SEARCH_RE.search(low):
        state.update({
            "jobs_shown": False,
            "phase": "discovery",
            "cached_jobs": []
        })

    # Update signals
    extract_signals(user_message, state)

    # Determine phase
    if state.get("readiness") and state.get("role_keywords") and state.get("location"):
        state["phase"] = "ready"

    # Already shown jobs, no new search
    if state.get("phase") == "results" and not NEW_SEARCH_RE.search(low):
        return {
            "assistantText": "",
            "mode": "results",
            "jobs": state.get("cached_jobs", [])
        }

    # Discovery phase
    if state.get("phase") == "discovery":
        q = next_discovery_question(state)
        if q:
            return {"assistantText": q, "mode": "chat", "jobs": []}
        state["phase"] = "ready"

    # Ready to fetch jobs
    if state["phase"] == "ready":
        jobs = await fetch_jobs(state["role_keywords"], state["location"]) or []
        state.update({
            "jobs_shown": True,
            "phase": "results",
            "cached_jobs": jobs
        })
        return {
            "assistantText": "Here are some options that match what you’re looking for.",
            "mode": "results" if jobs else "no_results",
            "jobs": jobs
        }

    # Fallback: generate AI response
    reply = await generate_coached_reply(state, conversation_history, user_message)
    return {"assistantText": reply, "mode": "chat", "jobs": []}
