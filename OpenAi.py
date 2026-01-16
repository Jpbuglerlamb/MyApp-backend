import asyncio
import json
import os
import re
import difflib
from typing import List, Dict, Any, Optional
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
    "waiter": "Server",
    "bar staff": "Server",
    "chef": "Chef",
    "cook": "Chef",
    "teacher": "Teacher",
    "driver": "Driver",
    "delivery driver": "Driver",
}

UK_CITIES = ["Edinburgh", "London", "Manchester", "Bristol", "Glasgow", "Leeds", "Liverpool", "Belfast", "Cardiff"]

STANDARD_INCOME_TYPES = {
    "full-time": ["full time", "full-time", "permanent"],
    "part-time": ["part time", "part-time", "casual", "zero hour"],
    "temporary": ["temporary", "temp", "short-term"],
    "freelance": ["freelance", "gig", "self-employed", "contract"],
    "internship": ["intern", "internship", "trainee"]
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


def normalize_income_type(user_text: str) -> str:
    """
    Determine income type from user text. Defaults to 'job' (no filter) unless explicitly mentioned.
    """
    low = user_text.lower()
    for key, variants in STANDARD_INCOME_TYPES.items():
        for v in variants:
            if v in low:
                return key
    return "job"  # default: no income filter


def map_role_synonym(role: str, cutoff: float = 0.7) -> str:
    if not role:
        return ""
    lowered = role.lower()
    for key, standard in ROLE_SYNONYMS.items():
        if key in lowered:
            return standard
    best_match = None
    highest_ratio = 0.0
    for key, standard in ROLE_SYNONYMS.items():
        ratio = difflib.SequenceMatcher(None, lowered, key).ratio()
        if ratio > highest_ratio and ratio >= cutoff:
            best_match = standard
            highest_ratio = ratio
    return best_match if best_match else role.title()

async def normalize_role_with_ai(role: str) -> str:
    role = role.strip()
    if not role or len(role) < 2:
        return role  # Return original if too short
    prompt = (
        "Clean and normalize this job role for a job search. "
        "Remove words like 'job', 'jobs', 'position', or 'role'. "
        "Do not include quotes. Return only the job title.\n\n"
        f"Text: {role}"
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        cleaned = response.choices[0].message.content.strip()
        cleaned = cleaned.strip("\"'")
        cleaned = re.sub(r"\bjobs?\b", "", cleaned, flags=re.I).strip()
        return cleaned
    except Exception:
        return role  # fallback: return original role


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
# Signal extraction (improved)
# -------------------------------------------------------------------
async def extract_signals(message: str, state: dict) -> None:
    """
    Extract role, location, and income_type from user message.
    Handles vague input, partial info, and small talk gracefully.
    """
    low = (message or "").lower().strip()

    # -----------------------------------
    # 0. Ignore small talk / gratitude
    # -----------------------------------
    small_talk_keywords = {"thanks", "thank you", "ok", "cool", "nice", "helpful", "great"}
    if any(word in low for word in small_talk_keywords):
        # Don't overwrite state, just mark acknowledged
        state["last_small_talk"] = message
        print(f"[DEBUG] Small talk detected: '{message}'")
        return

    # -----------------------------------
    # 1. Income type (full/part-time) directly in message
    # -----------------------------------
    if "part" in low:
        state["income_type"] = "part-time"
    elif "full" in low:
        state["income_type"] = "full-time"

    # -----------------------------------
    # 2. Extract role using AI + regex
    # -----------------------------------
    ai_role, ai_location = None, None
    try:
        ai_keywords = await extract_dynamic_keywords(_strip_fillers(message))
        ai_role = ai_keywords.get("role")
        ai_location = ai_keywords.get("location")
    except Exception as e:
        print(f"[DEBUG] AI keyword extraction failed: {e}")

    # Prefer AI-extracted role
    if ai_role:
        cleaned_role = await normalize_role_with_ai(_strip_fillers(ai_role))
        state["role_keywords"] = map_role_synonym(cleaned_role)
    else:
        # Regex fallback for phrases like "I want a server job"
        role_match = re.search(
            r"(?:work as|job as|be a|be an|looking for|i am a|i am an|part[- ]?time job as|full[- ]?time job as)\s+(.+?)" + _STOP,
            low, re.I
        )
        if role_match:
            fallback_role = await normalize_role_with_ai(normalize_role_for_api(role_match.group(1)))
            state["role_keywords"] = map_role_synonym(fallback_role)
        elif not state.get("role_keywords"):
            # Last resort: strip fillers from message
            state["role_keywords"] = map_role_synonym(_strip_fillers(message))

    # -----------------------------------
    # 3. Extract location
    # -----------------------------------
    if ai_location:
        loc = ai_location.strip().title()
    else:
        loc_match = re.search(r"\b(?:in|near|based in|based)\s+(.+?)" + _STOP, low, re.I)
        loc = loc_match.group(1).strip().title() if loc_match else None

    if loc:
        # Match against UK cities (fuzzy)
        matches = difflib.get_close_matches(loc, UK_CITIES, n=1, cutoff=0.7)
        state["location"] = matches[0] if matches else loc

    # -----------------------------------
    # 4. Detect salary (optional)
    # -----------------------------------
    salary_match = re.search(r"\b£?\d+(?:,\d{3})*(?:\s*(?:per|/)\s*(?:year|month|week|hour))?", low)
    if salary_match:
        state["salary"] = salary_match.group(0)

    # -----------------------------------
    # 5. Mark ready if role + location known
    # -----------------------------------
    if state.get("role_keywords") and state.get("location"):
        state["phase"] = "ready"
        state["readiness"] = True

    # -----------------------------------
    # 6. Debug logging
    # -----------------------------------
    print(f"[DEBUG] extract_signals final: {state}")
# -------------------------------------------------------------------
# Broader AI Role
# -------------------------------------------------------------------
async def broaden_role_with_ai(role: str, location: str) -> List[str]:
    if not role:
        return []
    prompt = f"""
You are a global job search assistant. A user wants jobs in {location}. 
The user typed this role: '{role}'.
Return 2-3 broader, search-friendly variants. Comma-separated, lowercase, no punctuation.
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return [x.strip() for x in response.choices[0].message.content.strip().split(",") if x.strip()]
    except Exception:
        return [role.lower()]
# -------------------------------------------------------------------
# Job fetching
# -------------------------------------------------------------------
async def fetch_jobs(role_keywords: str, location: str, income_type: str = "job") -> List[Dict[str, Any]]:
    if not role_keywords or not location:
        return []

    role_keywords = normalize_role_for_api(role_keywords)
    income_map = {
        "full-time": "fulltime",
        "part-time": "parttime",
        "temporary": "temporary",
        "freelance": "freelance",
        "internship": "internship",
        "job": ""
    }
    adzuna_income = income_map.get(income_type.lower(), "")
    query = f"{role_keywords} {adzuna_income}" if adzuna_income else role_keywords

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
# Discovery question
# -------------------------------------------------------------------
def next_discovery_question(state: dict) -> Optional[str]:
    if not state.get("role_keywords"):
        return "What kind of role or work are you looking for?"
    if not state.get("location"):
        return "What location should I search, or is remote okay?"
    return None

# -------------------------------------------------------------------
# AI reply generator (preserving your original system prompt)
# -------------------------------------------------------------------
async def generate_coached_reply(state: dict, conversation_history: List[dict], user_message: str) -> str:
    """
    Generate a guided AI response using your original system prompt.
    Includes state and recent conversation for context.
    """
    # Keep your system prompt exactly as you wrote it
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

    # Include dynamic state to give AI context
    state_prompt = (
        f"- income_type: {state.get('income_type')}\n"
        f"- location: {state.get('location')}\n"
        f"- role_keywords: {state.get('role_keywords')}\n"
        f"- salary: {state.get('salary')}\n"
        f"- readiness: {state.get('readiness')}\n"
        f"- jobs_shown: {state.get('jobs_shown')}\n"
        f"- phase: {state.get('phase')}\n"
        f"- last_small_talk: {state.get('last_small_talk')}\n"
    )

    # Use recent conversation history to maintain context
    trimmed_history = (conversation_history or [])[-12:]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": state_prompt},
        *trimmed_history,
        {"role": "user", "content": user_message}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.45  # slightly warm for natural tone
        )
        reply = response.choices[0].message.content.strip()
        if not reply:
            reply = "I'm here to help you find jobs or gigs. Can you tell me what role you're interested in?"
        return reply
    except Exception as e:
        print(f"[DEBUG] generate_coached_reply failed: {e}")
        return "Sorry, I had trouble processing that. Can you tell me about the role or location you're interested in?"

# -------------------------------
# Main chat entry (human-friendly, adaptive)
# -------------------------------
async def chat_with_user(*, user_id: str, user_message: str, conversation_history: list) -> dict:
    state = get_state(user_id)
    low = (user_message or "").lower().strip()

    # -----------------------------------
    # 1. Greetings / reset
    # -----------------------------------
    if low in {"hi", "hello", "hey", "hiya"}:
        state.clear()
        state.update({
            "phase": "discovery",
            "jobs_shown": False,
            "cached_jobs": [],
            "last_question": None,
            "asked_income_type": False,
            "last_small_talk": None
        })
        return {
            "assistantText": "Hello! I'm here to help you find jobs or gigs. What role are you interested in?",
            "mode": "chat",
            "jobs": [],
            "debug": {}
        }

    # -----------------------------------
    # 2. Detect new search intent
    # -----------------------------------
    if NEW_SEARCH_RE.search(low):
        state.update({
            "jobs_shown": False,
            "phase": "discovery",
            "cached_jobs": [],
            "last_question": None,
            "asked_income_type": False
        })

    # -----------------------------------
    # 3. Extract signals (role, location, income, salary)
    # -----------------------------------
    await extract_signals(user_message, state)

    # -----------------------------------
    # 3a. Detect mid-chat changes
    # -----------------------------------
    previous_role = state.get("role_keywords")
    previous_location = state.get("location")
    previous_income = state.get("income_type", "job")

    if state.get("role_keywords") and state["role_keywords"] != previous_role:
        state["phase"] = "ready"
        state["jobs_shown"] = False

    if state.get("location") and state["location"] != previous_location:
        state["phase"] = "ready"
        state["jobs_shown"] = False

    if ("part" in low and previous_income != "part-time") or ("full" in low and previous_income != "full-time"):
        state["income_type"] = "part-time" if "part" in low else "full-time"
        state["phase"] = "ready"
        state["jobs_shown"] = False

    # -----------------------------------
    # 4. Handle explicit income-type mentions
    # -----------------------------------
    if "part" in low:
        state["income_type"] = "part-time"
    elif "full" in low:
        state["income_type"] = "full-time"

    # -----------------------------------
    # 5. Discovery phase: ask minimal clarifying questions
    # -----------------------------------
    if state.get("phase") == "discovery":
        question = next_discovery_question(state)
        if question:
            state["last_question"] = question
            return {
                "assistantText": question,
                "mode": "chat",
                "jobs": [],
                "debug": {"role": state.get("role_keywords"), "location": state.get("location")}
            }
        state["phase"] = "ready"

    # -----------------------------------
    # 6. Prompt for income-type if unknown
    # -----------------------------------
    income_type = state.get("income_type", "job")
    if income_type == "job" and not state.get("asked_income_type"):
        state["asked_income_type"] = True
        return {
            "assistantText": f"I've found some '{state['role_keywords']}' jobs in {state['location']}. Do you want full-time or part-time work?",
            "mode": "chat",
            "jobs": [],
            "debug": {"role": state.get("role_keywords"), "location": state.get("location")}
        }

    # Handle reply to income-type question
    if state.get("asked_income_type"):
        if "part" in low:
            state["income_type"] = "part-time"
        elif "full" in low:
            state["income_type"] = "full-time"
        state["asked_income_type"] = False  # reset after handling

    # -----------------------------------
    # 7. Ready phase: fetch jobs if role & location known
    # -----------------------------------
    if state.get("phase") == "ready" and state.get("role_keywords") and state.get("location") and not state.get("jobs_shown"):
        user_role = state["role_keywords"]

        # Normalize role
        user_role = await normalize_role_with_ai(user_role)
        user_role = map_role_synonym(user_role)

        # Broaden role for search
        variants = await broaden_role_with_ai(user_role, state["location"])
        all_jobs = []
        for v in variants:
            jobs = await fetch_jobs(v, state["location"], state.get("income_type", "job"))
            all_jobs.extend(jobs)

        # Deduplicate
        unique_jobs = {(job['title'], job['company'], job['location']): job for job in all_jobs}
        all_jobs = list(unique_jobs.values())

        state.update({"jobs_shown": True, "phase": "results", "cached_jobs": all_jobs})

        # Compose human-friendly, coaching-style response
        income_type = state.get("income_type", "job")
        num_jobs = len(all_jobs)

        if num_jobs > 0:
            # Suggest gig vs traditional job based on role
            if user_role.lower() in ["cleaner", "bar staff", "server", "driver", "delivery driver"]:
                role_type = "These could be flexible gigs or part-time roles depending on your preference."
            else:
                role_type = "These are likely traditional employment roles, requiring some experience or qualifications."

            assistant_text = (
                f"Got it! I found {num_jobs} {income_type} listings for '{user_role}' in {state['location']}. "
                f"{role_type} "
                f"Let me know if you want details on these listings or advice on applying."
            )
            mode = "results"
        else:
            # No jobs found → suggest alternatives
            suggestions = []
            if income_type != "part-time":
                suggestions.append("try part-time roles")
            if state["location"] not in ["London", "Edinburgh", "Manchester"]:  # simple nearby suggestion
                suggestions.append("check nearby cities")
            suggestions.append("consider similar roles")

            suggestion_text = ", or ".join(suggestions)
            assistant_text = (
                f"Sorry, I couldn't find any {income_type} jobs for '{user_role}' in {state['location']}. "
                f"You could {suggestion_text}."
            )
            mode = "no_results"

        return {
            "assistantText": assistant_text,
            "mode": mode,
            "jobs": all_jobs,
            "debug": {"role": user_role, "location": state["location"], "query_count": num_jobs}
        }

    # -----------------------------------
    # 8. Fallback: natural response for chit-chat / gratitude
    # -----------------------------------
    if any(x in low for x in ["thanks", "thank you", "ok", "okay", "cool", "great"]):
        state["last_small_talk"] = user_message
        return {
            "assistantText": "You’re welcome! Let me know if you want to search for more jobs or gigs.",
            "mode": "chat",
            "jobs": [],
            "debug": {}
        }

    # Otherwise, generate AI-guided career response
    reply = await generate_coached_reply(state, conversation_history, user_message)
    if not reply.strip():
        reply = "Hello! I can help you find jobs or gigs. What role are you interested in?"

    return {"assistantText": reply.strip(), "mode": "chat", "jobs": [], "debug": {"role": state.get("role_keywords"), "location": state.get("location")}}
