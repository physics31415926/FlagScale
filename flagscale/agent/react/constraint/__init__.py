"""Constraint system — auto-extract hard constraints from skills, compile to Guards.

Phase 3 refactoring: Skill hard constraints are extracted from prose via LLM,
compiled into structured Constraint objects, and enforced via ConstraintGuard.

Design principle (REFACTOR_TRACKER.md D3):
- Deterministic trigger: tool_name + keyword match (cheap, no LLM)
- Precise judgment: only when triggered, call Judge.classify() (LLM)
- Block behavior: violated constraints return block + correction, not just warning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConstraintTrigger:
    """Deterministic trigger condition for a constraint.

    Checked before calling LLM for precise judgment.
    """
    tool_names: set[str] = field(default_factory=set)  # Empty = all tools
    keywords: list[str] = field(default_factory=list)  # Match in tool_args or tool_result

    def matches(self, tool_name: str, tool_args: dict, tool_result: str | None = None) -> bool:
        """Check if this trigger condition is satisfied."""
        # Tool name filter
        if self.tool_names and tool_name not in self.tool_names:
            return False

        # Keyword filter (case-insensitive)
        if self.keywords:
            # Combine all searchable text
            search_text = " ".join(str(v) for v in tool_args.values())
            if tool_result:
                search_text += " " + tool_result
            search_text = search_text.lower()

            # At least one keyword must match
            if not any(kw.lower() in search_text for kw in self.keywords):
                return False

        return True


@dataclass
class Constraint:
    """A hard constraint extracted from a skill.

    Constraints are enforced via ConstraintGuard, which:
    1. Checks trigger condition (deterministic, cheap)
    2. If triggered, calls Judge.classify() for precise judgment (LLM)
    3. If violated, returns block + correction message
    """
    id: str
    description: str
    trigger: ConstraintTrigger
    severity: Literal["error", "warning"] = "error"

    # LLM judge prompt for precise violation detection
    prompt: str = ""

    # Correction message injected when violated
    correction: str = ""

    # Lifecycle: when to check (pre = before tool exec, post = after)
    check_phase: Literal["pre", "post"] = "pre"

    # Max times to warn before escalating
    max_violations: int = 3


@dataclass
class ConstraintViolation:
    """A detected constraint violation."""
    constraint_id: str
    reason: str
    correction: str
    severity: Literal["error", "warning"]
