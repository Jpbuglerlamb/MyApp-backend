# core/state_machine.py

REQUIRED_FIELDS = ("role_canon", "location")  # changed

DISCOVERY_QUESTIONS = [
    "What role are you interested in?",
    "Which city or location do you prefer?",
    "Are you looking for full-time or part-time work?"
]

def is_ready(state: dict) -> bool:
    return all(state.get(field) for field in REQUIRED_FIELDS)

def advance_phase(state: dict) -> None:
    """
    Only updates readiness flags.
    DO NOT mutate state["phase"] because orchestrator owns the flow phase.
    """
    state["readiness"] = bool(is_ready(state))

def next_discovery_question(state: dict) -> str | None:
    asked = state.get("asked_questions", [])
    for question in DISCOVERY_QUESTIONS:
        if question not in asked:
            asked.append(question)
            state["asked_questions"] = asked
            return question
    return None
