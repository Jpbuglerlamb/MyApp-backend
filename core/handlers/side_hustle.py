# core/handlers/side_hustle.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from memory.profile_store import update_profile


def _get_skills(profile: Dict[str, Any]) -> List[str]:
    skills = profile.get("skills") or []
    names = []
    for s in skills:
        n = (s.get("name") or "").strip()
        if n:
            names.append(n)
    return names


def handle_side_hustle(
    *,
    user_id: str,
    low: str,
    user_message: str,
    state: Dict[str, Any],
    profile: Dict[str, Any],
    router_signals: Dict[str, Any] | None = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (text, actions, debug)
    Uses state["side_hustle_phase"] to run a tiny conversational flow.
    """
    router_signals = router_signals or {}

    phase = state.get("side_hustle_phase") or "start"

    # Accept yes/no shortcuts
    if phase == "offer" and low in {"yes", "yeah", "yep", "side_yes"}:
        phase = "hours"
    if phase == "offer" and low in {"no", "nah", "nope", "side_no"}:
        state["side_hustle_phase"] = None
        return (
            "All good. If you want, tell me a role + location and I’ll search listings.",
            [],
            {"intent": "SIDE_HUSTLE", "phase": "declined"},
        )

    # Start
    if phase == "start":
        state["side_hustle_phase"] = "offer"
        return (
            "Got you. Want me to tailor 3 side-hustle options to you? (2 quick questions)",
            [{"type": "YES_NO", "yesValue": "side_yes", "noValue": "side_no"}],
            {"intent": "SIDE_HUSTLE", "phase": "offer"},
        )

    # Hours per week
    if phase == "hours":
        # crude parse
        hours = None
        for token in ["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "15", "20"]:
            if f" {token} " in f" {low} ":
                hours = int(token)
                break

        if hours is None and any(x in low for x in ["weekend", "evening", "nights"]):
            # allow non-numeric answer
            update_profile(user_id, {"constraints": {"schedule": user_message.strip()}})
            state["side_hustle_phase"] = "skills"
            return (
                "Nice. What skills can you sell *today*? (pick a couple: coding, design, writing, trades, delivery, tutoring, admin, sales)",
                [],
                {"intent": "SIDE_HUSTLE", "phase": "skills"},
            )

        if hours is None:
            return (
                "How many hours per week can you realistically do? (example: 5, 10, 15)",
                [],
                {"intent": "SIDE_HUSTLE", "phase": "hours_reprompt"},
            )

        update_profile(user_id, {"constraints": {"hours_per_week": hours}})
        state["side_hustle_phase"] = "skills"
        return (
            "Cool. What skills can you sell *today*? (coding, design, writing, trades, delivery, tutoring, admin, sales)",
            [],
            {"intent": "SIDE_HUSTLE", "phase": "skills"},
        )

    # Skills capture (light)
    if phase == "skills":
        # If profile already has some skills and user says "use what you know"
        if any(x in low for x in ["you know", "use my skills", "from before"]):
            state["side_hustle_phase"] = "mode"
            return (
                "Got it. Do you want online, local, or both?",
                [
                    {"type": "BUTTON", "label": "Online", "value": "mode_online"},
                    {"type": "BUTTON", "label": "Local", "value": "mode_local"},
                    {"type": "BUTTON", "label": "Both", "value": "mode_both"},
                ],
                {"intent": "SIDE_HUSTLE", "phase": "mode"},
            )

        # Very simple keyword pickup (you can improve later)
        picks = []
        keywords = {
            "coding": ["code", "coding", "python", "javascript", "swift", "developer"],
            "design": ["design", "ui", "ux", "graphic"],
            "writing": ["write", "writing", "copy", "blog"],
            "trades": ["wood", "woodwork", "metal", "welding", "furniture", "construction"],
            "delivery": ["deliver", "delivery", "driver"],
            "tutoring": ["tutor", "teaching", "teacher"],
            "admin": ["admin", "assistant", "data entry"],
            "sales": ["sales", "sell", "closing"],
            "gardening": ["garden", "gardening", "landscap"],
        }
        for name, pats in keywords.items():
            if any(p in low for p in pats):
                picks.append(name)

        if not picks:
            # allow free-form, store as a note
            update_profile(user_id, {"preferences": {"sellable_skills_note": user_message.strip()}})
        else:
            # store as skills list
            existing = profile.get("skills") or []
            existing_names = { (s.get("name") or "").lower() for s in existing }
            new_skills = []
            for p in picks:
                if p.lower() not in existing_names:
                    new_skills.append({"name": p, "evidence": "user_message", "confidence": 0.65})
            if new_skills:
                update_profile(user_id, {"skills": (existing + new_skills)})

        state["side_hustle_phase"] = "mode"
        return (
            "Nice. Do you want online, local, or both?",
            [
                {"type": "BUTTON", "label": "Online", "value": "mode_online"},
                {"type": "BUTTON", "label": "Local", "value": "mode_local"},
                {"type": "BUTTON", "label": "Both", "value": "mode_both"},
            ],
            {"intent": "SIDE_HUSTLE", "phase": "mode"},
        )

    # Mode selection
    if phase == "mode":
        mode = None
        if "mode_online" in low or "online" in low:
            mode = "online"
        elif "mode_local" in low or "local" in low:
            mode = "local"
        elif "mode_both" in low or "both" in low:
            mode = "both"

        if not mode:
            return ("Online, local, or both?", [], {"intent": "SIDE_HUSTLE", "phase": "mode_reprompt"})

        update_profile(user_id, {"preferences": {"hustle_mode": mode}})
        state["side_hustle_phase"] = "recommend"

    # Recommendations
    # Use profile to tailor
    skills = _get_skills(profile)
    hours = (profile.get("constraints") or {}).get("hours_per_week")
    mode = (profile.get("preferences") or {}).get("hustle_mode", "both")

    # Basic scoring
    recs = []

    if mode in {"online", "both"}:
        if any(s.lower() in {"coding", "design", "writing"} for s in skills):
            recs.append(("Freelance services", "Fiverr/Upwork/Freelancer", "Pick ONE service, make 3 examples, publish a gig."))
        else:
            recs.append(("Micro-services", "Fiverr", "Offer something simple: CV formatting, basic admin, or listings research."))

    if mode in {"local", "both"}:
        if any(s.lower() in {"trades", "woodwork", "gardening", "delivery"} for s in skills):
            recs.append(("Local cash route", "Facebook Marketplace / Gumtree", "Offer a weekend service: flat-pack builds, garden tidy, small repairs."))
        else:
            recs.append(("Local helper route", "Local groups", "One flyer message: 'Weekend help: moving, cleaning, errands'."))

    # Always include a resale/flip option if time is low
    recs.append(("Flip route", "Facebook Marketplace / eBay", "Find free/cheap items, clean, re-list with better photos."))

    # Format response
    line0 = "Alright. Based on what I know so far, here are 3 practical routes:"
    lines = [line0, ""]
    for i, (name, platform, step) in enumerate(recs[:3], start=1):
        lines.append(f"{i}) **{name}** via *{platform}*")
        lines.append(f"   First step: {step}")

    if hours:
        lines.append("")
        lines.append(f"With ~{hours} hrs/week, I’d start with option 1 and run a 7-day mini sprint.")

    state["side_hustle_phase"] = None
    return ("\n".join(lines), [], {"intent": "SIDE_HUSTLE", "phase": "recommend", "mode": mode, "hours": hours, "skills": skills})
