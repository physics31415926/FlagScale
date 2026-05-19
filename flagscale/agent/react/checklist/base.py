"""Declarative ChecklistItem + ChecklistEngine + Checklist base.

All rule evaluation uses LLM classify() — no regex/keyword matching.

Rules are evaluated in a single batched LLM call per tool observation:
all matching ChecklistItems are packed into one checklist_rule_batch classify()
call, which returns only the violated constraint IDs. This keeps budget usage
at 1 per tool call regardless of how many constraints are loaded.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ChecklistAlert:
    """A single alert returned from Checklist.check()."""
    message: str
    severity: str = "warning"  # "error" | "warning" | "info"


@dataclass
class ChecklistItem:
    """A single domain check evaluated by the LLM judge.

    No evaluate callable, no regex rules. The prompt field describes
    the classification task, and the engine injects runtime context
    (driver version, shared storage paths, etc.) alongside the
    observation.
    """

    id: str
    description: str
    phases: set[str] = field(default_factory=lambda: {"*"})

    # ── Trigger ──
    trigger_on: dict | None = None  # {"tool": "shell"} or None for all tools

    # ── LLM judge prompt (the classification task) ──
    prompt: str = ""

    # ── Reminder ──
    reminder: str = ""
    severity: str = "warning"  # "warning" | "error" | "info"
    max_reminders: int = 3


class ChecklistEngine:
    """Holds auto-detected runtime facts injected into every classify call."""

    def __init__(self):
        self.facts: dict[str, str] = {}

    def evaluate_batch(self, items: list[ChecklistItem], obs) -> list[str]:
        """Evaluate multiple items in one LLM call. Returns violated item IDs.

        Filters items by trigger_on and prompt presence, packs them into
        a single checklist_rule_batch classify() call, and returns the
        IDs of items the LLM flagged as violated.
        """
        # Filter by trigger_on
        candidates = []
        for item in items:
            if item.trigger_on:
                trigger_tool = item.trigger_on.get("tool")
                if trigger_tool and obs.tool_name != trigger_tool:
                    continue
            if not item.prompt:
                continue
            candidates.append(item)

        if not candidates:
            return []

        classify = getattr(obs, "classify_fn", None)
        if not classify:
            return []

        # Build context
        context = {
            "tool_name": obs.tool_name,
            "tool_args": obs.tool_args or {},
            "tool_result": obs.tool_result or "",
        }
        facts = self.facts
        if facts:
            context["_facts"] = facts

        # Pack items for the batch prompt
        items_payload = [
            {"id": item.id, "description": item.description, "prompt": item.prompt}
            for item in candidates
        ]

        result = classify("checklist_rule_batch", {
            "items": items_payload,
            "context": context,
        }, default=[])

        # result should be [{"id": "...", "reason": "..."}, ...]
        if isinstance(result, list):
            return [v.get("id", "") for v in result if isinstance(v, dict) and v.get("id")]
        return []


class Checklist:
    """A collection of domain checklist items."""

    def __init__(self, engine: ChecklistEngine | None = None, items: list[ChecklistItem] | None = None):
        self.engine = engine or ChecklistEngine()
        self._items: list[ChecklistItem] = list(items) if items else []
        self._reminder_counts: dict[str, int] = {}
        # Track which violations were reported in the previous check() call,
        # so we only return new violations each time.
        self._previous_violated: set[str] = set()
        # Statistics
        self.total_checks: int = 0
        self.total_violations: int = 0
        self.violation_by_id: dict[str, int] = {}

    def add(self, item: ChecklistItem):
        self._items.append(item)

    # Regex to extract SCOPE from a constraint prompt of the form
    # "SCOPE: <condition>. CHECK: <condition>."
    _SCOPE_RE = re.compile(r'SCOPE:\s*(.+?)\.\s*(?:CHECK|DO NOT MATCH|MATCH)\b', re.IGNORECASE)

    def override(self, constraint_id: str) -> bool:
        """Permanently remove a constraint by id. Returns True if removed."""
        before = len(self._items)
        self._items = [item for item in self._items if item.id != constraint_id]
        removed = len(self._items) < before
        if removed:
            self._previous_violated.discard(constraint_id)
            self._reminder_counts.pop(constraint_id, None)
            self.violation_by_id.pop(constraint_id, None)
        return removed

    def pre_check_tool(self, tool_name: str, tool_args: dict) -> list[ChecklistAlert]:
        """Lightweight pre-exec check based on SCOPE matching.

        Returns alerts for constraints whose SCOPE clearly matches the
        about-to-be-executed tool call. This is a heads-up, not a violation
        verdict — the post-exec check via check() still has the final say.

        No LLM call — pure string matching on the SCOPE clause.
        """
        cmd = tool_args.get("command", "") if tool_name == "shell" else ""
        file_path = ""
        if tool_name in ("write_file", "edit_file"):
            file_path = tool_args.get("path", "") or tool_args.get("file_path", "")
        if tool_name == "read_file":
            file_path = tool_args.get("path", "") or tool_args.get("file_path", "")

        alerts = []
        for item in self._items:
            prompt = item.prompt
            if not prompt:
                continue
            m = self._SCOPE_RE.search(prompt)
            if not m:
                continue
            scope = m.group(1).lower()

            # Match scope against the specific tool call
            incoming = [tool_name]
            if cmd:
                incoming.append(cmd.lower())
            if file_path:
                incoming.append(file_path.lower())
            combined = " ".join(incoming)

            # Check if the scope condition is satisfied by the incoming command
            # Use the scope as a bag of significant keywords
            scope_keywords = [
                w for w in re.findall(r'[\"\'\`]([^\"\'\`]+)[\"\'\`]', scope)
            ]
            if not scope_keywords:
                # Fallback: extract significant multi-word phrases from scope
                scope_keywords = [w.strip() for w in scope.split(",") if len(w.strip()) > 5]

            has_match = False
            for kw in scope_keywords:
                if kw.lower() in combined:
                    has_match = True
                    break

            if has_match:
                alerts.append(ChecklistAlert(
                    f"⚠ Pre-check: [{item.id}] {item.reminder}",
                    severity="warning",
                ))

        return alerts

    def check(self, obs) -> list[ChecklistAlert]:
        """Evaluate all matching items in one batched LLM call.

        Returns ChecklistAlert objects for newly-violated constraints only
        (not repeats). Uses _previous_violated to suppress stale violations.
        """
        # Filter by phase
        matching = [
            item for item in self._items
            if obs.phase_name in item.phases or "*" in item.phases
        ]

        if not matching:
            return []

        # Single LLM call for all matching items
        self.total_checks += 1
        violated_ids = set(self.engine.evaluate_batch(matching, obs))

        if not violated_ids:
            # No violations this round — reset tracking
            self._previous_violated.clear()
            return []

        # Detect new violations (not in previous round)
        new_ids = violated_ids - self._previous_violated
        # Detect cleared violations (were in previous, not in current)
        cleared_ids = self._previous_violated - violated_ids
        self._previous_violated = violated_ids

        # Filter out IDs not in our items
        item_map = {item.id: item for item in self._items}
        new_ids = {vid for vid in new_ids if vid in item_map}

        if not new_ids and not cleared_ids:
            return []

        self.total_violations += 1
        alerts = []

        # Summary header
        if new_ids:
            parts = []
            for vid in sorted(new_ids):
                item = item_map.get(vid)
                desc = item.description if item else vid
                parts.append(f"[{vid}] {desc}")
            alerts.append(ChecklistAlert("📋 CHECKLIST: " + "; ".join(parts), severity="info"))
        if cleared_ids:
            cleared_parts = [f"[{vid}]" for vid in sorted(cleared_ids)]
            alerts.append(ChecklistAlert(f"  ✅ Resolved: {' '.join(cleared_parts)}", severity="info"))

        # Build detailed messages for each new violation
        for vid in new_ids:
            item = item_map.get(vid)
            if not item:
                continue
            key = item.id
            self._reminder_counts[key] = self._reminder_counts.get(key, 0) + 1
            self.violation_by_id[key] = self._reminder_counts[key]
            count = self._reminder_counts[key]
            severity = item.severity or "warning"
            msg = f"[{item.id}] {item.reminder} (×{count})"
            if count >= item.max_reminders:
                msg += (
                    f"\n⚠ Ignored {count} times. "
                    f"Either address this or declare override: "
                    f"[CHECKLIST_OVERRIDE: {item.id}] Reason: <justification>"
                )
            alerts.append(ChecklistAlert(msg, severity=severity))
        return alerts

    @classmethod
    def from_skill_constraints(cls, engine: ChecklistEngine, skill_meta: dict) -> "Checklist":
        """Build a Checklist from constraint declarations in a Skill's YAML frontmatter."""
        items = []
        for c in skill_meta.get("constraints", []):
            items.append(ChecklistItem(
                id=c["id"],
                description=c.get("description", c["id"]),
                phases=set(c.get("phases", ["*"])),
                trigger_on=c.get("trigger_on"),
                prompt=c.get("prompt", ""),
                reminder=c.get("reminder", ""),
                severity=c.get("severity", "warning"),
                max_reminders=c.get("max_reminders", 3),
            ))
        return cls(engine=engine, items=items)
