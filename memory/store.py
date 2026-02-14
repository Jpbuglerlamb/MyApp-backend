# memory/store.py
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any

from memory.models import default_state
from settings import MAX_MESSAGES

# -------------------------------------------------------------------
# Users store: user_id -> { conversations: {conversation_id: {...}}, decks: {...} }
# -------------------------------------------------------------------
USERS: defaultdict[str, Dict[str, Any]] = defaultdict(lambda: {
    "conversations": defaultdict(lambda: {
        "state": default_state(),
        "memory": [],
        "jobs": [],
        "cached_jobs": [],     # optional convenience
        "current_deck": None,  # optional convenience
    }),
    "decks": {}  # deck_id -> deck payload (cards + liked/passed)
})


def _conv(user_id: str, conversation_id: str) -> Dict[str, Any]:
    return USERS[str(user_id)]["conversations"][str(conversation_id)]


# -----------------------
# Accessors
# -----------------------
def get_user_state(user_id: str, conversation_id: str) -> Dict[str, Any]:
    return _conv(user_id, conversation_id)["state"]

def get_user_memory(user_id: str, conversation_id: str) -> List[Dict[str, str]]:
    return _conv(user_id, conversation_id)["memory"]

def get_user_jobs(user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
    return _conv(user_id, conversation_id)["jobs"]


# -----------------------
# Mutators
# -----------------------
def append_user_memory(user_id: str, conversation_id: str, role: str, content: str) -> None:
    mem = _conv(user_id, conversation_id)["memory"]
    mem.append({"role": role, "content": content})
    _conv(user_id, conversation_id)["memory"] = mem[-MAX_MESSAGES:]

def save_user_job(user_id: str, conversation_id: str, job: Dict[str, Any]) -> None:
    _conv(user_id, conversation_id)["jobs"].append(job)

def clear_user_memory(user_id: str, conversation_id: str) -> None:
    _conv(user_id, conversation_id)["memory"] = []

def clear_user_state(user_id: str, conversation_id: str) -> None:
    _conv(user_id, conversation_id)["state"] = default_state()

def clear_user_jobs(user_id: str, conversation_id: str) -> None:
    _conv(user_id, conversation_id)["jobs"] = []

def clear_conversation(user_id: str, conversation_id: str) -> None:
    clear_user_state(user_id, conversation_id)
    clear_user_memory(user_id, conversation_id)
    clear_user_jobs(user_id, conversation_id)
    _conv(user_id, conversation_id)["cached_jobs"] = []
    _conv(user_id, conversation_id)["current_deck"] = None


# -----------------------
# Deck registry (per user)
# -----------------------
def put_deck(user_id: str, deck_id: str, deck: Dict[str, Any]) -> None:
    USERS[str(user_id)]["decks"][str(deck_id)] = deck

def get_deck(user_id: str, deck_id: str) -> Dict[str, Any] | None:
    return USERS[str(user_id)]["decks"].get(str(deck_id))

def delete_deck(user_id: str, deck_id: str) -> None:
    USERS[str(user_id)]["decks"].pop(str(deck_id), None)
