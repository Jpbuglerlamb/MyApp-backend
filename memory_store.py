from collections import defaultdict
from typing import Dict, List, Optional
import hashlib
import uuid

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# USER MEMORY & STATE STORE
# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Stores all registered users
# Each user has email, hashed password, session_id, state, memory, job history, preferences
USERS: Dict[str, dict] = defaultdict(
    lambda: {
        "first_name": None,   # optional
        "last_name": None,
        "email": None,
        "password_hash": None,  # hashed password
        "session_id": None,
        "state": {},
        "memory": [],           # last messages for context
        "job_history": [],      # job searches or saved listings
        "preferences": {},      # e.g., preferred locations, roles, income_type
    }
)

MAX_MESSAGES = 20  # max chat messages to store for LLM context

# -------------------------------------------------------------------
# Utility: password hashing (simple SHA256 for example; replace with bcrypt in prod)
# -------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash

# -------------------------------------------------------------------
# User account management
# -------------------------------------------------------------------
def create_user(email: str, password: str, first_name: Optional[str] = None, last_name: Optional[str] = None) -> str:
    """
    Create a new user and return session_id.
    """
    # Check if email already exists
    for user in USERS.values():
        if user["email"] == email:
            raise ValueError("Email already registered.")

    session_id = str(uuid.uuid4())
    USERS[session_id]["first_name"] = first_name
    USERS[session_id]["last_name"] = last_name
    USERS[session_id]["email"] = email
    USERS[session_id]["password_hash"] = hash_password(password)
    USERS[session_id]["session_id"] = session_id
    USERS[session_id]["state"] = default_state()
    USERS[session_id]["memory"] = []
    USERS[session_id]["job_history"] = []
    USERS[session_id]["preferences"] = {}
    return session_id

def login_user(email: str, password: str) -> str:
    """
    Validate credentials. Returns session_id if successful.
    """
    for user in USERS.values():
        if user["email"] == email and verify_password(password, user["password_hash"]):
            return user["session_id"]
    return None

# -------------------------------------------------------------------
# Default conversation state
# -------------------------------------------------------------------
def default_state() -> dict:
    """
    Returns a fresh state object for a new conversation.
    """
    return {
        "phase": "discovery",         # discovery | ready | presenting_jobs
        "role_keywords": None,
        "location": None,
        "skills": [],
        "income_type": None,           # job | gig | None
        "readiness": False,
        "last_question": None,
        "jobs_shown": False,
        "asked_income_type": False,
        "last_small_talk": None
    }

# -------------------------------------------------------------------
# User state access
# -------------------------------------------------------------------
def get_user_state(session_id: str) -> dict:
    """
    Return mutable state for a user session.
    """
    if session_id not in USERS:
        raise ValueError("Invalid session_id")
    return USERS[session_id]["state"]

def update_user_state(session_id: str, key: str, value) -> None:
    """
    Update a single field in user state.
    """
    USERS[session_id]["state"][key] = value

def clear_user_state(session_id: str) -> None:
    """
    Reset user state.
    """
    if session_id in USERS:
        USERS[session_id]["state"] = default_state()

# -------------------------------------------------------------------
# User chat memory (LLM context)
# -------------------------------------------------------------------
def get_user_memory(session_id: str) -> List[dict]:
    return USERS[session_id]["memory"]

def append_user_memory(session_id: str, role: str, content: str) -> None:
    """
    Append message to user memory and trim to MAX_MESSAGES
    """
    USERS[session_id]["memory"].append({"role": role, "content": content})
    if len(USERS[session_id]["memory"]) > MAX_MESSAGES:
        USERS[session_id]["memory"] = USERS[session_id]["memory"][-MAX_MESSAGES:]

def clear_user_memory(session_id: str) -> None:
    USERS[session_id]["memory"] = []

# -------------------------------------------------------------------
# User job history & preferences
# -------------------------------------------------------------------
def save_user_job(session_id: str, job: dict) -> None:
    """
    Save a job search / listing for the user
    """
    USERS[session_id]["job_history"].append(job)

def get_user_job_history(session_id: str) -> List[dict]:
    return USERS[session_id]["job_history"]

def update_user_preferences(session_id: str, key: str, value) -> None:
    USERS[session_id]["preferences"][key] = value

def get_user_preferences(session_id: str) -> dict:
    return USERS[session_id]["preferences"]