# app/jobs/role_expansion.py
from typing import List
from ai.client import client

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