from collections import defaultdict
from typing import Dict, List

# -------------------------------------------------------------------
# Session message memory
# -------------------------------------------------------------------
# Stores raw chat messages for LLM context only
# -------------------------------------------------------------------

SESSION_MEMORY: Dict[str, List[dict]] = defaultdict(list)

MAX_MESSAGES = 20  # total messages (user + assistant)


def get_memory(session_id: str) -> List[dict]:
    """
    Return chat history for a session.
    """
    return SESSION_MEMORY.get(session_id, [])


def append_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to session memory and trim oldest entries.
    """
    SESSION_MEMORY[session_id].append({
        "role": role,
        "content": content
    })

    # Trim to most recent MAX_MESSAGES
    if len(SESSION_MEMORY[session_id]) > MAX_MESSAGES:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-MAX_MESSAGES:]


def clear_memory(session_id: str) -> None:
    """
    Clear message history for a session.
    """
    SESSION_MEMORY.pop(session_id, None)


# -------------------------------------------------------------------
# Session conversational state (AI Aura canonical truth)
# -------------------------------------------------------------------
# This is NOT chat memory.
# This is structured, authoritative state used for decisions.
# -------------------------------------------------------------------

SESSION_STATE: Dict[str, dict] = defaultdict(
    lambda: {
        # Conversation control
        "phase": "discovery",  # discovery | ready | presenting_jobs

        # Signals
        "role_keywords": None,
        "location": None,
        "skills": [],
        "income_type": None,  # job | gig | None
        "readiness": False,

        # UX / guards
        "last_question": None,
        "jobs_shown": False,  # prevents loops
    }
)


def get_state(session_id: str) -> dict:
    """
    Return mutable session state.
    """
    return SESSION_STATE[session_id]


def update_state(session_id: str, key: str, value) -> None:
    """
    Update a single state field.
    """
    SESSION_STATE[session_id][key] = value


def set_phase(session_id: str, phase: str) -> None:
    """
    Explicitly set conversation phase.
    """
    SESSION_STATE[session_id]["phase"] = phase


def clear_state(session_id: str) -> None:
    """
    Clear all state for a session.
    """
    SESSION_STATE.pop(session_id, None)