"""Skill manager — loads and parses SKILL.md files."""

import os

from typing import Dict, List, Optional, Tuple

import yaml


class SkillManager:
    """Manages skill loading from prioritized directories."""

    def __init__(self, dirs: List[str]):
        self._dirs = dirs

    def _scan(self) -> Dict[str, str]:
        """Build mapping: skill_name -> skill_file_path (later dirs override)."""
        mapping = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        name = meta.get("name", entry)
                    except Exception:
                        name = entry
                    mapping[name] = skill_file
                    mapping[entry] = skill_file
        return mapping

    def list_skills(self) -> List[Dict[str, str]]:
        """Scan all directories and return available skills (deduplicated)."""
        seen_paths = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        seen_paths[skill_file] = {
                            "name": meta.get("name", entry),
                            "description": meta.get("description", ""),
                            "keywords": meta.get("keywords", []),
                        }
                    except Exception:
                        seen_paths[skill_file] = {"name": entry, "description": "", "keywords": []}
        return list(seen_paths.values())

    def load(self, name: str) -> str:
        """Load a skill by frontmatter name or directory name. Later directories take priority."""
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            raise FileNotFoundError(f"Skill '{name}' not found in: {self._dirs}")
        meta, body = self._parse_file(skill_file)
        skill_name = meta.get("name", name)
        return f"<skill name=\"{skill_name}\">\n{body}\n</skill>"

    def _parse_file(self, path: str) -> Tuple[dict, str]:
        """Read a SKILL.md and split YAML frontmatter from body."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._parse_frontmatter(content)

    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[dict, str]:
        """Split --- delimited YAML frontmatter from markdown body."""
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            meta = {}
        body = parts[2].strip()
        return meta, body
