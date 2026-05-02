"""Session memory — stores key findings, decisions, and todos across conversations."""

import os
import time

from typing import Dict, List, Optional

import yaml


_TYPE_PRIORITY = {"todo": 0, "decision": 1, "finding": 2, "context": 3}


class SessionMemory:
    """Incremental memory for cross-session continuity with TTL expiration."""

    def __init__(self, memory_dir: str, ttl_days: int = 7):
        self._dir = memory_dir
        self._ttl = ttl_days * 86400

    def _entry_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace(" ", "_")
        return os.path.join(self._dir, f"{safe_key}.yaml")

    def get(self, key: str) -> Optional[dict]:
        path = self._entry_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = yaml.safe_load(f)
        except Exception:
            return None
        if not self._is_valid(entry):
            self.delete(key)
            return None
        return entry

    def put(self, key: str, mem_type: str, content: str, session_id: str = ""):
        os.makedirs(self._dir, exist_ok=True)
        entry = {
            "key": key,
            "type": mem_type,
            "content": content,
            "session_id": session_id,
            "created": time.time(),
        }
        path = self._entry_path(key)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
        return path

    def delete(self, key: str) -> bool:
        path = self._entry_path(key)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def list_entries(self) -> List[dict]:
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
                    entries.append(entry)
            except Exception:
                continue
        return entries

    def recent(self, max_tokens: int = 800) -> List[dict]:
        """Return entries within a token budget, prioritized by type then recency.

        Priority: todo > decision > finding > context.
        Within the same type, newest entries come first.
        """
        entries = self.list_entries()
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
        created = entry.get("created", 0)
        return time.time() - created <= self._ttl
