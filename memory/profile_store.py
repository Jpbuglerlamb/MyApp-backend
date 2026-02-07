# memory/profile_store.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

DB_PATH = os.getenv("AXIS_PROFILE_DB", "axis_profiles.sqlite3")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_profile_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
              user_id TEXT PRIMARY KEY,
              profile_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_profile(user_id: str) -> Dict[str, Any]:
    init_profile_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()

    if not row:
        return {
            "skills": [],         # [{name, evidence, confidence, last_seen}]
            "preferences": {},    # e.g. {"mode":"remote", "social":"low"}
            "constraints": {},    # e.g. {"hours_per_week": 10, "schedule":"weekends"}
            "goals": {},          # e.g. {"income_target":"£300/m", "timeline":"3 months"}
        }

    try:
        return json.loads(row["profile_json"])
    except Exception:
        return {}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _upsert_profile(user_id: str, profile: Dict[str, Any]) -> None:
    init_profile_db()
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              profile_json = excluded.profile_json,
              updated_at = excluded.updated_at
            """,
            (str(user_id), json.dumps(profile, ensure_ascii=False), now),
        )
        conn.commit()


def update_profile(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge patch into profile and persist.
    Returns updated profile.
    """
    cur = get_profile(user_id)
    merged = _deep_merge(cur, patch)
    _upsert_profile(user_id, merged)
    return merged


def add_skill(
    user_id: str,
    *,
    name: str,
    evidence: str = "",
    confidence: float = 0.7,
) -> Dict[str, Any]:
    profile = get_profile(user_id)
    skills = profile.get("skills") or []

    # Update if exists
    for s in skills:
        if (s.get("name") or "").lower() == name.lower():
            s["confidence"] = max(float(s.get("confidence") or 0.0), confidence)
            if evidence:
                s["evidence"] = evidence
            s["last_seen"] = datetime.utcnow().isoformat()
            profile["skills"] = skills
            _upsert_profile(user_id, profile)
            return profile

    skills.append(
        {
            "name": name,
            "evidence": evidence,
            "confidence": confidence,
            "last_seen": datetime.utcnow().isoformat(),
        }
    )
    profile["skills"] = skills
    _upsert_profile(user_id, profile)
    return profile
