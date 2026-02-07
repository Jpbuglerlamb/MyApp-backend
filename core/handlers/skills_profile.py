# core/handlers/skills_profile.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from memory.profile_store import get_profile, update_profile


def _format_skills(profile: Dict[str, Any]) -> str:
    skills = profile.get("skills") or []
    if not skills:
        return "I don’t have any skills stored yet."

    # Sort by confidence desc
    skills_sorted = sorted(skills, key=lambda s: float(s.get("confidence") or 0.0), reverse=True)
    top = skills_sorted[:8]
    parts = []
    for s in top:
        name = (s.get("name") or "").strip()
        conf = float(s.get("confidence") or 0.0)
        if name:
            parts.append(f"• {name} ({int(conf*100)}%)")
    return "\n".join(parts) if parts else "I don’t have any skills stored yet."


def handle_skills_profile(
    *,
    user_id: str,
    low: str,
    user_message: str,
    state: Dict[str, Any],
    profile: Dict[str, Any],
    router_signals: Dict[str, Any] | None = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Tiny flow:
      - if no skills: ask 2 questions
      - if skills exist: summarise + offer next step (job search or side hustle)
    """
    router_signals = router_signals or {}
    phase = state.get("skills_phase") or "start"

    if phase == "start":
        skills = profile.get("skills") or []
        if skills:
            text = (
                "Here’s what I’ve got for you so far:\n"
                f"{_format_skills(profile)}\n\n"
                "Want to use these for: (1) job search, or (2) a side hustle plan?"
            )
            state["skills_phase"] = "next"
            return (
                text,
                [{"type": "YES_NO", "yesValue": "skills_job", "noValue": "skills_hustle"}],
                {"intent": "SKILLS_PROFILE", "phase": "summary"},
            )

        state["skills_phase"] = "collect"
        return (
            "Quick skills snapshot. What have you done before that you’re genuinely good at? (even informal stuff)",
            [],
            {"intent": "SKILLS_PROFILE", "phase": "collect"},
        )

    if phase == "collect":
        # Store as a note (safe + simple), you can parse into structured skills later
        update_profile(user_id, {"preferences": {"skills_free_text": user_message.strip()}})
        state["skills_phase"] = "constraints"
        return (
            "Nice. Any constraints right now? (time per week, location limits, remote vs local, anything you won’t do)",
            [],
            {"intent": "SKILLS_PROFILE", "phase": "constraints"},
        )

    if phase == "constraints":
        update_profile(user_id, {"constraints": {"notes": user_message.strip()}})
        state["skills_phase"] = None
        return (
            "Got it. Tell me what you want next:\n"
            "• a role + city for job search\n"
            "or\n"
            "• ‘side hustle’ and I’ll tailor 3 routes.",
            [],
            {"intent": "SKILLS_PROFILE", "phase": "done"},
        )

    if phase == "next":
        if "skills_job" in low:
            state["skills_phase"] = None
            return (
                "Cool. Tell me the role + location you want (example: backend developer in London).",
                [],
                {"intent": "SKILLS_PROFILE", "phase": "to_job_search"},
            )

        if "skills_hustle" in low:
            state["skills_phase"] = None
            state["side_hustle_phase"] = "start"
            return (
                "Nice. Say ‘side hustle’ and I’ll tailor 3 options to your skills.",
                [],
                {"intent": "SKILLS_PROFILE", "phase": "to_side_hustle"},
            )

        return (
            "Job search or side hustle?",
            [{"type": "YES_NO", "yesValue": "skills_job", "noValue": "skills_hustle"}],
            {"intent": "SKILLS_PROFILE", "phase": "reprompt"},
        )
