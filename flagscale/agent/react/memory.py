"""Session memory — stores key findings, decisions, and todos across conversations."""

import logging
import os
import re
import time

from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_TYPE_PRIORITY = {"finding": 0, "decision": 1, "todo": 2, "context": 3}


class SessionMemory:
    """Incremental memory for cross-session continuity with TTL expiration."""

    def __init__(self, memory_dir: str, ttl_days: int = 30):
        self._dir = memory_dir
        self._ttl = ttl_days * 86400
        self._cleanup_expired()  # Clean up expired entries on init

    def _cleanup_expired(self):
        """Remove expired memory entries from disk."""
        if not os.path.isdir(self._dir):
            return
        removed = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if not self._is_valid(entry):
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
        if removed > 0:
            logger.info("Cleaned up %d expired memory entries", removed)

    _KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,78}[a-z0-9]$")

    @staticmethod
    def sanitize_key(raw: str) -> str:
        """Normalize a raw key to lowercase alphanumeric + underscores, max 80 chars."""
        k = raw.lower().strip()
        k = re.sub(r"[^a-z0-9]+", "_", k)
        k = k.strip("_")
        if len(k) > 80:
            k = k[:80].rstrip("_")
        return k

    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        return bool(cls._KEY_RE.match(key))

    def _entry_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.yaml")

    def get(self, key: str) -> Optional[dict]:
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key
        path = self._entry_path(safe)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = yaml.safe_load(f)
        except Exception:
            return None
        if not self._is_valid(entry):
            self.delete(safe)
            return None
        return entry

    def put(self, key: str, mem_type: str, content: str, session_id: str = "", task: str = "", priority: str = "normal", scope: str = "persistent"):
        os.makedirs(self._dir, exist_ok=True)
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key
        entry = {
            "key": safe,
            "type": mem_type,
            "content": content,
            "session_id": session_id,
            "task": task,
            "priority": priority,
            "scope": scope,
            "created": time.time(),
        }
        path = self._entry_path(safe)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
        return path

    def delete(self, key: str) -> bool:
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key
        path = self._entry_path(safe)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def cleanup_session(self, session_id: str) -> int:
        """Remove all session-scoped entries for a given session. Called at session end."""
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if entry.get("scope") == "session" and entry.get("session_id") == session_id:
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count

    def list_entries(self, scope_filter: str = "") -> List[dict]:
        if not os.path.isdir(self._dir):
            return []
        entries = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if self._is_valid(entry):
                    if scope_filter and entry.get("scope", "persistent") != scope_filter:
                        continue
                    entries.append(entry)
            except Exception:
                continue
        return entries

    def recent(self, max_tokens: int = 4000, task_filter: str = "", current_session_id: str = "") -> List[dict]:
        """Return entries within a token budget, prioritized by task relevance, type, then recency.

        If task_filter is provided, entries matching that task come first, then other entries.
        Priority within each group: finding > decision > todo > context.
        Within the same type, newest entries come first.
        Session-scoped entries from other sessions are excluded.
        """
        entries = self.list_entries()
        # Exclude session-scoped entries from other sessions
        if current_session_id:
            entries = [e for e in entries
                       if e.get("scope", "persistent") != "session"
                       or e.get("session_id") == current_session_id]

        if task_filter:
            # Split into task-matching and other entries
            task_entries = [e for e in entries if e.get("task", "") == task_filter]
            other_entries = [e for e in entries if e.get("task", "") != task_filter]

            # Sort each group by type priority, then recency
            task_entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))
            other_entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))

            # Task entries come first
            entries = task_entries + other_entries
        else:
            # No filter: sort by type priority, then recency
            entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))

        result = []
        used = 0
        for e in entries:
            content = e.get("content", "")
            cjk = sum(1 for c in content if '一' <= c <= '鿿' or '　' <= c <= '〿' or '가' <= c <= '힯' or '぀' <= c <= 'ヿ')
            ascii_chars = len(content) - cjk
            cost = ascii_chars // 4 + int(cjk * 1.5) + 10
            if used + cost > max_tokens:
                break
            result.append(e)
            used += cost
        return result

    def clear(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if fname.endswith(".yaml"):
                os.remove(os.path.join(self._dir, fname))
                count += 1
        return count

    def clear_by_type(self, mem_type: str) -> int:
        """Delete all entries of a specific type. Returns count deleted."""
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if entry.get("type") == mem_type:
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count

    def _is_valid(self, entry: dict) -> bool:
        if entry.get("priority") == "high":
            return True
        created = entry.get("created", 0)
        return time.time() - created <= self._ttl
