# agents/memory_logger.py

"""
Memory Logger for InsightGPT
----------------------------
Logs user feedback, goal input, and agent session metadata to a local JSON file for recall and context reuse.
"""

import os
import json
from datetime import datetime

# --- Setup Memory Log File ---
LOG_FILE = "memory/memory_log.json"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_session(state: dict, df_shape=None):
    """Logs session metadata like goal, completed stages, and feedback."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "goal": state.get("goal", "N/A"),
        "completed_stages": state.get("completed_stages", []),
        "feedback_log": state.get("feedback_log", []),
        "df_shape": df_shape if df_shape else "unknown"
    }

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(log_entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data[-20:], f, indent=4)  # Keep only last 20 sessions
    
    return log_entry


def load_recent_sessions(limit=3):
    """Loads the most recent N sessions from memory."""
    if not os.path.exists(LOG_FILE):
        return []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[-limit:]
    except json.JSONDecodeError:
        return []


def log_feedback(goal: str, df_shape: tuple, feedback: str):
    """Logs standalone feedback (called during regeneration only)."""
    new_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "goal": goal,
        "df_shape": str(df_shape),
        "feedback_log": [feedback],
        "completed_stages": []
    }

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        except json.JSONDecodeError:
            sessions = []
    else:
        sessions = []

    sessions.append(new_entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions[-20:], f, indent=4)
