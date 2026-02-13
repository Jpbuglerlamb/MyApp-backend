import os, sqlite3
from datetime import datetime
from typing import List, Dict

DB_PATH = os.getenv("AXIS_CHAT_DB", "axis_chat.sqlite3")

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_chat_db():
    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
          conversation_id TEXT NOT NULL,
          user_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_conv ON chat_messages(conversation_id)")
        conn.commit()

def add_message(conversation_id: str, user_id: str, role: str, content: str):
    init_chat_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()

def get_messages(conversation_id: str, limit: int = 30) -> List[Dict[str, str]]:
    init_chat_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    # return oldest -> newest
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
