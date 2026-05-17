"""Checklist — B类 gate (领域约束).

ChecklistItems are declarative rules, NOT imperative callables.
Agent can create/modify constraints by writing YAML in SKILL.md.

Key design:
- No evaluate callable — rule engine interprets declarative fields
- Constraints come from Skill YAML frontmatter, not Python code
- Reminders are injected into history; Agent decides whether to comply
- Repeatedly ignored items escalate after max_reminders
"""

from __future__ import annotations

from .base import ChecklistItem, Checklist, ChecklistEngine

__all__ = ["ChecklistItem", "Checklist", "ChecklistEngine"]
