"""Declarative ChecklistItem + ChecklistEngine + Checklist base."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

from ..interrupt.base import Observation


@dataclass
class ChecklistItem:
    """A single domain check described declaratively.

    No evaluate callable — the engine evaluates rules automatically.
    This means Agent can create/modify constraints by writing YAML,
    not Python code.
    """

    id: str
    description: str
    phases: set[str] = field(default_factory=lambda: {"*"})  # Which phases this check applies to

    # ── Declarative rules (instead of callable) ──
    trigger_on: dict | None = None  # {"tool": "write_file", "path_match": "*.py"}
    content_rules: list[dict] = field(default_factory=list)  # Applied to source content
    result_rules: list[dict] = field(default_factory=list)  # Applied to tool result

    reminder: str = ""
    severity: str = "warning"  # "warning" | "error" | "info"
    max_reminders: int = 3


class ChecklistEngine:
    """Evaluates declarative ChecklistItems against Observations."""

    @staticmethod
    def _rule_matches(rule: dict, text: str) -> bool:
        pattern = rule["match"]
        mode = rule.get("mode", "contains")
        if mode == "regex":
            return bool(re.search(pattern, text))
        if mode == "exact":
            return pattern == text
        if mode == "not_contains":
            return pattern not in text
        if mode == "starts_with":
            return text.strip().startswith(pattern)
        # default: contains
        return pattern in text

    def evaluate(self, item: ChecklistItem, obs: Observation) -> bool:
        """Return False (check failed / rule matched) if any rule triggers.

        True = all rules passed (no reminder needed).
        False = at least one rule matched (remind the agent).
        """
        # Step 1: trigger filter — does this check even apply?
        if item.trigger_on:
            if obs.tool_name != item.trigger_on.get("tool"):
                return True  # Wrong tool — skip
            if "path_match" in item.trigger_on:
                path = (
                    obs.tool_args.get("path", "")
                    or obs.tool_args.get("file_path", "")
                )
                if not fnmatch.fnmatch(path, item.trigger_on["path_match"]):
                    return True  # Path doesn't match — skip

        # Step 2: check source content (for write_file/edit_file)
        content = obs.tool_args.get("content", "") or obs.tool_args.get("new_string", "")
        for rule in item.content_rules:
            if self._rule_matches(rule, content):
                return False

        # Step 3: check tool result (for shell, etc.)
        result = obs.tool_result or ""
        for rule in item.result_rules:
            if self._rule_matches(rule, result):
                return False

        return True


class Checklist:
    """A collection of domain checklist items.

    Items can be added at runtime (Agent can inject constraints).
    Constraint source is Skill YAML frontmatter, not Python code.
    """

    def __init__(self, engine: ChecklistEngine | None = None, items: list[ChecklistItem] | None = None):
        self.engine = engine or ChecklistEngine()
        self._items: list[ChecklistItem] = list(items) if items else []
        self._reminder_counts: dict[str, int] = {}

    def add(self, item: ChecklistItem):
        """Register a new check item. Agent can call this at runtime."""
        self._items.append(item)

    def check(self, obs: Observation) -> list[str]:
        """Evaluate all items matching the current phase. Return reminders to inject."""
        reminders = []
        for item in self._items:
            if obs.phase_name not in item.phases and "*" not in item.phases:
                continue
            if not self.engine.evaluate(item, obs):
                key = item.id
                self._reminder_counts[key] = self._reminder_counts.get(key, 0) + 1
                count = self._reminder_counts[key]
                msg = f"[{item.id}] {item.reminder}"
                if count >= item.max_reminders:
                    msg += (
                        f"\n⚠ Ignored {count} times. "
                        f"Either address this or declare override: "
                        f"[CHECKLIST_OVERRIDE: {item.id}] Reason: <justification>"
                    )
                reminders.append(msg)
        return reminders

    @classmethod
    def from_skill_constraints(cls, engine: ChecklistEngine, skill_meta: dict) -> "Checklist":
        """Build a Checklist from constraint declarations in a Skill's YAML frontmatter.

        This is the bridge: Skill markdown → runtime ChecklistItems.
        No Python code needed to add a constraint.
        """
        items = []
        for c in skill_meta.get("constraints", []):
            items.append(ChecklistItem(
                id=c["id"],
                description=c.get("description", c["id"]),
                phases=set(c.get("phases", ["*"])),
                trigger_on=c.get("trigger_on"),
                content_rules=c.get("content_rules", []),
                result_rules=c.get("result_rules", []),
                reminder=c.get("reminder", ""),
                severity=c.get("severity", "warning"),
                max_reminders=c.get("max_reminders", 3),
            ))
        return cls(engine=engine, items=items)
