"""Project knowledge cache — stores derived knowledge from file analysis."""

import hashlib
import os
import time

from typing import Dict, List, Optional

import yaml


class KnowledgeCache:
    """Cache for project knowledge summaries with source-file-hash invalidation."""

    def __init__(self, cache_dir: str, ttl_days: int = 7):
        self._dir = cache_dir
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

        if not self._validate(entry):
            self.delete(key)
            return None
        return entry

    def put(self, key: str, description: str, content: str, sources: List[str]):
        os.makedirs(self._dir, exist_ok=True)
        source_entries = []
        for src in sources:
            h = self._file_hash(src)
            if h:
                source_entries.append({"path": src, "sha256": h})
            else:
                source_entries.append({"path": src, "sha256": None})

        entry = {
            "key": key,
            "description": description,
            "content": content,
            "sources": source_entries,
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
                valid = self._validate(entry)
                entries.append({
                    "key": entry.get("key", fname),
                    "description": entry.get("description", ""),
                    "created": entry.get("created", 0),
                    "valid": valid,
                    "sources": len(entry.get("sources", [])),
                })
            except Exception:
                continue
        return entries

    def clear(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if fname.endswith(".yaml"):
                os.remove(os.path.join(self._dir, fname))
                count += 1
        return count

    def query(self, text: str) -> List[dict]:
        if not os.path.isdir(self._dir):
            return []
        text_lower = text.lower()
        words = set(text_lower.replace("-", " ").replace("_", " ").split())
        results = []
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
            except Exception:
                continue

            if not self._validate(entry):
                self.delete(entry.get("key", fname.replace(".yaml", "")))
                continue

            key = entry.get("key", "")
            desc = entry.get("description", "")
            match_text = (key + " " + desc).lower()
            match_words = set(match_text.replace("-", " ").replace("_", " ").split())

            overlap = words & match_words
            if not overlap:
                for w in words:
                    if len(w) > 2 and w in match_text:
                        overlap.add(w)

            if overlap:
                results.append((len(overlap), entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results]

    def _validate(self, entry: dict) -> bool:
        created = entry.get("created", 0)
        if time.time() - created > self._ttl:
            return False

        for src in entry.get("sources", []):
            path = src.get("path", "")
            expected_hash = src.get("sha256")
            if expected_hash is None:
                continue
            current_hash = self._file_hash(path)
            if current_hash != expected_hash:
                return False
        return True

    @staticmethod
    def _file_hash(path: str) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, IOError):
            return None
