def default_state() -> dict:
    return {
        "phase": "discovery",
        "role_raw": None,
        "role_keywords": None,   # keep for compatibility
        "role_display": None,
        "role_canon": None,
        "role_query": None,
        "resolved_role": None,
        "location": None,
        "income_type": None,
        "salary": None,
        "last_small_talk": None,
        "jobs_shown": False,
        "asked_income_type": False,
        "asked_questions": [],
        "readiness": False,
        "current_deck": None,
        "cached_jobs": [],       # now holds JobCards
    }
