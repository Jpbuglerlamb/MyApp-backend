# ai/generation.py
from typing import List
from ai.client import client
from ai.prompts import SYSTEM_PROMPT

async def generate_coached_reply(
    state: dict,
    conversation_history: List[dict],
    user_message: str
) -> str:
    """
    Generate a guided AI response using your original system prompt.
    Includes state and recent conversation for context.
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": state_prompt},
        *trimmed_history,
        {"role": "user", "content": user_message}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.45
        )
        reply = response.choices[0].message.content.strip()
        return reply or (
            "I'm here to help you find jobs or gigs. "
            "Can you tell me what role you're interested in?"
        )
    except Exception as e:
        print(f"[DEBUG] generate_coached_reply failed: {e}")
        return (
            "Sorry, I had trouble processing that. "
            "Can you tell me about the role or location you're interested in?"
        )