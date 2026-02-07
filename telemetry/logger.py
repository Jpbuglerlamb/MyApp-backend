# telemetry/logger.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DB_PATH = "telemetry.sqlite3"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_id TEXT,
            event TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    c.commit()
    return c


def log_event(event: str, payload: Dict[str, Any], user_id: Optional[str] = None) -> None:
    """
    Telemetry must NEVER crash production logic.
    Payload should avoid raw user text by default.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            c.execute(
                "INSERT INTO events (ts, user_id, event, payload) VALUES (?, ?, ?, ?)",
                (ts, user_id, event, json.dumps(payload, ensure_ascii=False)),
            )
            c.commit()
    except Exception:
        pass
