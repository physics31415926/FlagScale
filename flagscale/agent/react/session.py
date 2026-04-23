"""Session persistence — save/load conversation history to disk."""

import json
import os
import time

from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_session_dir() -> str:
    return os.path.join(Path.home(), ".flagscale", "sessions")


def save_session(messages: List[Dict[str, Any]], session_dir: Optional[str] = None,
                 session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    d = session_dir or _default_session_dir()
    os.makedirs(d, exist_ok=True)
    sid = session_id or f"session_{int(time.time())}"
    path = os.path.join(d, f"{sid}.json")
    data = {
        "id": sid,
        "timestamp": time.time(),
        "metadata": metadata or {},
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_session(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions(session_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    d = session_dir or _default_session_dir()
    if not os.path.isdir(d):
        return []
    sessions = []
    for fname in sorted(os.listdir(d), reverse=True):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(d, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", fname),
                "path": path,
                "timestamp": data.get("timestamp", 0),
                "turns": len([m for m in data.get("messages", []) if m.get("role") == "user"]),
            })
        except Exception:
            continue
    return sessions
