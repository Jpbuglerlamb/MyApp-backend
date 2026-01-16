import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional
import difflib
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from memory_store import get_state

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
# Constants
# -------------------------------------------------------------------
_STOP = r"(?:\s+(?:in|near|based in|based)\b|[.,;!?]|$)"
BAD_ROLE_KEYWORDS = {"a job", "job", "jobs", "work", "position", "role", "career", "employment"}
NEW_SEARCH_RE = re.compile(r"\b(find|search|look for|can you find|what about)\b", re.I)
DYNAMIC_KEYWORDS = ["jobs", "gigs", "role", "industry", "location", "salary", "qualifications", "experience", "knowledge"]
ROLE_SYNONYMS = {
    # Tech / IT
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
    "waiter": "Server",
    "bar staff": "Server",
    "chef": "Chef",
    "cook": "Chef",
    # General / flexible
    "teacher": "Teacher",
    "driver": "Driver",
    "delivery driver": "Driver",
}
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _strip_fillers(text: str) -> str:
    t = text.lower()
    fillers = ["i'm","im","i am","looking","looking for","looking at","i want","i need","find me",
               "search","show me","a","an","the","job","jobs","role","position","work","please","thanks"]
    for f in fillers:
        t = re.sub(rf"\b{re.escape(f)}\b","",t)
    return re.sub(r"\s+"," ",t).strip()

def normalize_role_for_api(role: str) -> str:
    role = role.lower()
    fillers = ["part time job as","full time job as","job as","a","an","the"]
    for f in fillers:
        role = role.replace(f,"")
    return role.strip()

def map_role_synonym(role: str, cutoff: float = 0.7) -> str:
    """
    Map a user-entered role to a standard title using ROLE_SYNONYMS.
    Uses fuzzy matching for typos and partial matches.
    
    Args:
        role: The cleaned role from the user / AI
        cutoff: minimum similarity (0.0-1.0) to accept a match
    """
    if not role:
        return ""

    lowered = role.lower()

    # Exact substring match first
    for key, standard in ROLE_SYNONYMS.items():
        if key in lowered:
            return standard

    # Fuzzy match
    best_match = None
    highest_ratio = 0.0
    for key, standard in ROLE_SYNONYMS.items():
        ratio = difflib.SequenceMatcher(None, lowered, key).ratio()
        if ratio > highest_ratio and ratio >= cutoff:
            best_match = standard
            highest_ratio = ratio

    if best_match:
        return best_match

    # Fallback: capitalize words
    return role.title()

STANDARD_INCOME_TYPES = {
    "full-time": ["full time", "full-time", "permanent"],
    "part-time": ["part time", "part-time", "casual", "zero hour"],
    "temporary": ["temporary", "temp", "short-term"],
    "freelance": ["freelance", "gig", "self-employed", "contract"],
    "internship": ["intern", "internship", "trainee"]
}

def normalize_income_type(user_text: str) -> str:
    low = user_text.lower()
    for key, variants in STANDARD_INCOME_TYPES.items():
        for v in variants:
            if v in low:
                return key
    return "job"

async def normalize_role_with_ai(role: str) -> str:
    if not role:
        return ""

    # --- Step 1: AI cleanup prompt
    prompt = (
        "Clean and normalize this job role for a job search. "
        "Remove words like 'job', 'jobs', 'position', or 'role'. "
        "Do not include quotes. Return only the job title.\n\n"
        f"Text: {role}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        cleaned = response.choices[0].message.content.strip()
    except Exception:
        cleaned = role.strip()

    # --- Step 2: Remove wrapping quotes & trailing 'job'
    cleaned = cleaned.strip("\"'")
    cleaned = re.sub(r"\bjobs?\b", "", cleaned, flags=re.I).strip()

    # --- Step 3: Map synonyms
    lowered = cleaned.lower()
    for key, standard in ROLE_SYNONYMS.items():
        if key in lowered:
            return standard  # return standardized role

    # --- Step 4: fallback: capitalize words
    return cleaned.title()

async def extract_dynamic_keywords(user_message: str) -> Dict[str,str]:
    prompt = f"Extract keywords from this text: {DYNAMIC_KEYWORDS}. Text: {user_message}"
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception:
        return {}

# -------------------------------------------------------------------
# Signal extraction
# -------------------------------------------------------------------
async def extract_signals(message: str, state: dict) -> None:
    """
    Robust extraction of role, location, income type, and readiness.
    Steps:
    1️⃣ Normalize income type from user text
    2️⃣ AI extraction of role and location
    3️⃣ Regex fallback always runs to correct AI misreads
    4️⃣ Final GPT cleanup of role
    5️⃣ Set phase and readiness if both role and location exist
    """
    low = (message or "").lower().strip()

    # --- 1️⃣ Income type
    if not state.get("income_type"):
        state["income_type"] = normalize_income_type(message)

    # --- 2️⃣ AI dynamic keyword extraction (primary)
    ai_role, ai_location = None, None
    try:
        ai_keywords = await extract_dynamic_keywords(message)
        ai_role = ai_keywords.get("role")
        ai_location = ai_keywords.get("location")
    except Exception as e:
        print(f"[DEBUG] AI keyword extraction failed: {e}")

    # --- Tentatively set AI outputs
    if ai_role:
        cleaned_role = await normalize_role_with_ai(ai_role)
        standardized_role = map_role_synonym(cleaned_role)  # fuzzy mapping
        state["role_keywords"] = standardized_role
    if ai_location:
        state["location"] = ai_location.title()

    # --- 3️⃣ Regex fallback (always runs)
    role_match = re.search(
        r"(?:work as|job as|be a|be an|looking for|i am a|i am an|part[- ]?time job as|full[- ]?time job as)\s+(.+?)" + _STOP,
        low, re.I
    )
    if role_match:
        fallback_role = normalize_role_for_api(role_match.group(1))
        fallback_role = await normalize_role_with_ai(fallback_role)
        standardized_role = map_role_synonym(fallback_role)  # fuzzy mapping
        if standardized_role.lower() not in BAD_ROLE_KEYWORDS and standardized_role.lower() != (state.get("role_keywords") or "").lower():
            state["role_keywords"] = standardized_role

    # --- Location fallback
    if not state.get("location"):
        loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
        if loc_match:
            state["location"] = loc_match.group(1).strip().title()

    # --- 4️⃣ Final sanity check: remove junk roles
    if (state.get("role_keywords") or "").lower() in BAD_ROLE_KEYWORDS:
        state["role_keywords"] = None

    # --- 5️⃣ Ready phase
    if state.get("role_keywords") and state.get("location"):
        state["phase"] = "ready"
        state["readiness"] = True

    print(f"[DEBUG] extract_signals final: {state}")

# -------------------------------------------------------------------
# Discovery question
# -------------------------------------------------------------------
def next_discovery_question(state: dict) -> Optional[str]:
    if not state.get("role_keywords"): return "What kind of role or work are you looking for?"
    if not state.get("location"): return "What location should I search, or is remote okay?"
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
        f"- income_type: {state.get('income_type')}\n"
        f"- location: {state.get('location')}\n"
        f"- role_keywords: {state.get('role_keywords')}\n"
        f"- readiness: {state.get('readiness')}\n"
        f"- jobs_shown: {state.get('jobs_shown')}\n"
        f"- phase: {state.get('phase')}\n"
    )
    trimmed_history = (conversation_history or [])[-12:]
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"system","content":state_prompt},
        *trimmed_history,
        {"role":"user","content":user_message}
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Job fetching (enhanced)
# -------------------------------------------------------------------
async def fetch_jobs(role_keywords: str, location: str, income_type: str = "job") -> List[Dict[str, Any]]:
    """
    Fetch jobs from Adzuna API with proper role, location, and income type handling.
    - Normalizes role keywords to remove filler words
    - Maps income_type to Adzuna category keywords
    - Returns normalized job list
    """
    if not role_keywords or not location:
        return []

    # Normalize role for API
    role_keywords = normalize_role_for_api(role_keywords)

    # Map internal income_type to Adzuna search terms
    income_map = {
        "full-time": "fulltime",
        "part-time": "parttime",
        "temporary": "temporary",
        "freelance": "freelance",
        "internship": "internship",
        "job": ""  # no filter
    }
    adzuna_income = income_map.get(income_type.lower(), "")

    # Construct query string
    query = role_keywords
    if adzuna_income:
        query += f" {adzuna_income}"

    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location.title(),
        "results_per_page": 38,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client_http:
            response = await client_http.get(url, params=params)
            response.raise_for_status()
            results = response.json().get("results", []) or []

            # Normalize each job into internal structure
            normalized_jobs = []
            for job in results:
                title = job.get("title", "") or ""
                company = job.get("company", {}).get("display_name") if isinstance(job.get("company"), dict) else job.get("company", "")
                loc = job.get("location", {}).get("display_name") if isinstance(job.get("location"), dict) else job.get("location", "")
                redirect_url = job.get("redirect_url", "")
                normalized_jobs.append({
                    "title": title.strip(),
                    "company": company.strip() if company else "",
                    "location": loc.strip() if loc else "",
                    "redirect_url": redirect_url.strip() if redirect_url else ""
                })

            return normalized_jobs

    except Exception as e:
        print(f"[DEBUG] fetch_jobs failed for query='{query}' location='{location}': {e}")
        return []


# -------------------------------------------------------------------
# Main chat entry
# -------------------------------------------------------------------
async def chat_with_user(*, user_id: str, user_message: str, conversation_history: list) -> dict:
    state = get_state(user_id)
    low = (user_message or "").lower().strip()

    # --- Greeting resets conversation
    if low in {"hi", "hello", "hey", "hiya"}:
        state.clear()
        state.update({
            "phase": "discovery",
            "jobs_shown": False,
            "cached_jobs": []
        })

    # --- First message: AI intro
    if not conversation_history:
        reply = await generate_coached_reply(state, [], user_message)
        return {"assistantText": reply, "mode": "chat", "jobs": []}

    # --- Reset search if user explicitly starts a new one
    if NEW_SEARCH_RE.search(low):
        state.update({
            "jobs_shown": False,
            "phase": "discovery",
            "cached_jobs": []
        })

    # --- Extract signals
    await extract_signals(user_message, state)

    print(
        f"[DEBUG] chat_with_user: user_message='{user_message}', "
        f"role_keywords='{state.get('role_keywords')}', "
        f"location='{state.get('location')}', "
        f"income_type='{state.get('income_type')}', "
        f"phase='{state.get('phase')}'"
    )

    # --- Results already shown
    if state.get("phase") == "results" and not NEW_SEARCH_RE.search(low):
        return {
            "assistantText": "Here are the roles I found earlier. Want to refine or try something else?",
            "mode": "results",
            "jobs": state.get("cached_jobs", [])
        }

    # --- Ready phase logic
    if state.get("phase") == "ready":
        if NEW_SEARCH_RE.search(low):
            role = state.get("role_keywords")
            location = state.get("location")
            income_type = state.get("income_type") or "job"

            jobs = await fetch_jobs(role, location, income_type)
            state.update({
                "jobs_shown": True,
                "phase": "results",
                "cached_jobs": jobs
            })

            assistant_text = (
                f"Here are some {income_type} options for a '{role}' in {location}."
                if jobs else
                f"Sorry, I couldn't find any {income_type} jobs for a '{role}' in {location}."
            )

            return {
                "assistantText": assistant_text,
                "mode": "results" if jobs else "no_results",
                "jobs": jobs
            }

        reply = await generate_coached_reply(state, conversation_history, user_message)
        return {"assistantText": reply, "mode": "chat", "jobs": []}

    # --- Discovery or fallback
    reply = await generate_coached_reply(state, conversation_history, user_message)
    return {"assistantText": reply, "mode": "chat", "jobs": []}
