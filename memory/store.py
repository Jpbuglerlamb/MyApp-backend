# memory/store.py
from collections import defaultdict
from typing import List, Dict, Any, Optional
from memory.models import default_state
from settings import MAX_MESSAGES

# -------------------------------------------------------------------
# In-memory short-term memory (session_id -> user data)
# -------------------------------------------------------------------
USERS: defaultdict[str, Dict[str, Any]] = defaultdict(lambda: {
    "state": default_state(),
    "memory": [],
    "jobs": []
})

# -------------------------------------------------------------------
# Accessors
# -------------------------------------------------------------------
def get_user_state(session_id: str) -> Dict[str, Any]:
    return USERS[session_id]["state"]

def get_user_memory(session_id: str) -> List[Dict[str, str]]:
    return USERS[session_id]["memory"]

def get_user_jobs(session_id: str) -> List[Dict[str, Any]]:
    return USERS[session_id]["jobs"]

# -------------------------------------------------------------------
# Mutators
# -------------------------------------------------------------------
def append_user_memory(session_id: str, role: str, content: str) -> None:
    USERS[session_id]["memory"].append({"role": role, "content": content})
    USERS[session_id]["memory"] = USERS[session_id]["memory"][-MAX_MESSAGES:]

def save_user_job(session_id: str, job: Dict[str, Any]) -> None:
    USERS[session_id]["jobs"].append(job)

def clear_user_memory(session_id: str) -> None:
    USERS[session_id]["memory"] = []

def clear_user_state(session_id: str) -> None:
    USERS[session_id]["state"] = default_state()

def clear_user_jobs(session_id: str) -> None:
    USERS[session_id]["jobs"] = []

def clear_user(session_id: str) -> None:
    """Reset everything for a user session"""
    clear_user_state(session_id)
    clear_user_memory(session_id)
    clear_user_jobs(session_id)