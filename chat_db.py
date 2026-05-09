import json
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

# ─── ENGINE ──────────────────────────────────────────────────────
def get_engine():
    url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url)

engine = get_engine()

# ─── CREATE TABLES IF NOT EXISTS ─────────────────────────────────
def init_chat_table():
    with engine.connect() as conn:
        # ... your existing CREATE TABLE statements ...

        # ── Auto cleanup ─────────────────────────────────────────
        # Delete sessions older than 90 days
        conn.execute(text("""
            DELETE FROM chat_history 
            WHERE session_id IN (
                SELECT session_id FROM chat_sessions 
                WHERE updated_at < NOW() - INTERVAL 90 DAY
            )
        """))
        conn.execute(text("""
            DELETE FROM chat_sessions 
            WHERE updated_at < NOW() - INTERVAL 90 DAY
        """))

        # Keep only the 30 most recent sessions
        conn.execute(text("""
            DELETE ch FROM chat_history ch
                    LEFT JOIN (
                        SELECT session_id 
                        FROM chat_sessions 
                        ORDER BY updated_at DESC 
                        LIMIT 30
                    ) AS latest_sessions
                    ON ch.session_id = latest_sessions.session_id
                    WHERE latest_sessions.session_id IS NULL;
        """))
        conn.execute(text("""
            DELETE FROM chat_sessions 
            WHERE session_id NOT IN (
                SELECT * FROM (
                    SELECT session_id FROM chat_sessions 
                    ORDER BY updated_at DESC LIMIT 30
                ) AS keep
            )
        """))

        conn.commit()

# ─── SESSION MANAGEMENT ──────────────────────────────────────────
def create_session(session_id, title="New Chat"):
    """Register a new session in chat_sessions."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT IGNORE INTO chat_sessions (session_id, title)
            VALUES (:sid, :title)
        """), {"sid": session_id, "title": title})
        conn.commit()

def rename_session(session_id, title):
    """Update the title of a session."""
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE chat_sessions SET title = :title
            WHERE session_id = :sid
        """), {"sid": session_id, "title": title})
        conn.commit()

def delete_session(session_id):
    """Delete a session and all its messages."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM chat_history WHERE session_id = :sid"), {"sid": session_id})
        conn.execute(text("DELETE FROM chat_sessions WHERE session_id = :sid"), {"sid": session_id})
        conn.commit()

def load_all_sessions():
    """
    Returns all sessions ordered by most recently updated.
    Each entry: {"session_id": ..., "title": ..., "updated_at": ...}
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT session_id, title, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
        """)).mappings().fetchall()
    return [dict(r) for r in rows]

# ─── SAVE ONE CHAT TURN ──────────────────────────────────────────
def save_chat(session_id, query, data, insight, chart_type, chart_data):
    """
    Saves a user message and assistant response as two rows.
    Also auto-sets session title from first message if still 'New Chat'.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO chat_history
                (session_id, role, message, data, chart_type, chart_data)
            VALUES
                (:sid, 'user',      :query,   NULL,  NULL,         NULL),
                (:sid, 'assistant', :insight, :data, :chart_type,  :chart_data)
        """), {
            "sid":        session_id,
            "query":      query,
            "insight":    insight,
            "data":       json.dumps(data, default=str),
            "chart_type": chart_type,
            "chart_data": json.dumps(chart_data, default=str),
        })

        # Auto-title the session from the first user message (truncated)
        conn.execute(text("""
            UPDATE chat_sessions
            SET title = LEFT(:title, 60), updated_at = CURRENT_TIMESTAMP
            WHERE session_id = :sid AND title = 'New Chat'
        """), {"sid": session_id, "title": query})

        # Always bump updated_at so ordering stays correct
        conn.execute(text("""
            UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = :sid
        """), {"sid": session_id})

        conn.commit()

# ─── LOAD FULL HISTORY FOR A SESSION ────────────────────────────
def load_chat(session_id):
    """
    Returns a list of chat entries (each entry = one user+assistant pair).
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT role, message, data, chart_type, chart_data
            FROM chat_history
            WHERE session_id = :sid
            ORDER BY created_at ASC, id ASC
        """), {"sid": session_id}).mappings().fetchall()

    rows = [dict(r) for r in rows]
    history = []
    i = 0
    while i < len(rows) - 1:
        user_row = rows[i]
        bot_row  = rows[i + 1]
        if user_row["role"] == "user" and bot_row["role"] == "assistant":
            history.append({
                "query":      user_row["message"],
                "insight":    bot_row["message"],
                "data":       json.loads(bot_row["data"])       if bot_row["data"]       else None,
                "chart_type": bot_row["chart_type"]             if bot_row["chart_type"] else "none",
                "chart_data": json.loads(bot_row["chart_data"]) if bot_row["chart_data"] else [],
            })
        i += 2
    return history

# ─── LOAD CONVERSATION CONTEXT FOR LLM ──────────────────────────
def load_conversation_context(session_id, last_n=6):
    """
    Returns the last N exchanges as {"role", "content"} dicts for the LLM.
    """
    history = load_chat(session_id)
    context = []
    for entry in history[-last_n:]:
        context.append({"role": "user",      "content": entry["query"]})
        context.append({"role": "assistant",  "content": entry["insight"]})
    return context

# ─── CLEAR HISTORY FOR A SESSION (keep session row) ─────────────
def clear_chat(session_id):
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM chat_history WHERE session_id = :sid"),
            {"sid": session_id}
        )
        conn.execute(text("""
            UPDATE chat_sessions SET title = 'New Chat', updated_at = CURRENT_TIMESTAMP
            WHERE session_id = :sid
        """), {"sid": session_id})
        conn.commit()