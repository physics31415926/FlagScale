"""WarningGuard — soft reminders that inject messages without blocking.

Unlike ConstraintGuard (which blocks on violation), WarningGuard only
injects reminder messages. It respects max_reminders to avoid spam.

Design:
1. Deterministic trigger: tool_name + keyword match (cheap)
2. LLM judgment: classify_fn determines if warning applies
3. Inject behavior: reminder message injected, execution continues
"""

from __future__ import annotations

import logging
from typing import Any

from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale.agent.react.state_machine import AgentState

logger = logging.getLogger(__name__)


class WarningGuard(Guard):
    """Soft reminder Guard — injects messages without blocking.

    Each instance manages a list of warning definitions loaded from
    a Skill's frontmatter 'warnings' field.
    """

    name = "warning"
    priority = 26  # After constraint (25), before progress (30)
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING}

    def __init__(self, warnings: list[dict] | None = None):
        """Initialize with warning definitions.

        Each warning dict should have:
            id, description, severity, trigger, prompt, reminder, max_reminders
        """
        self._warnings: list[dict] = warnings or []
        self._reminder_counts: dict[str, int] = {}

    def add_warnings(self, warnings: list[dict]):
        """Add more warning definitions (e.g., after loading additional skills)."""
        self._warnings.extend(warnings)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check warnings before tool execution."""
        for warning in self._warnings:
            if not self._is_triggered(warning, ctx):
                continue

            wid = warning["id"]
            max_reminders = warning.get("max_reminders", 2)
            count = self._reminder_counts.get(wid, 0)

            if count >= max_reminders:
                continue

            # LLM judgment: should we actually warn?
            should_warn = self._judge_warning(warning, ctx)
            if not should_warn:
                continue

            self._reminder_counts[wid] = count + 1
            reminder = warning.get("reminder", warning.get("description", ""))

            logger.info("Warning triggered: %s (count=%d/%d)",
                        wid, count + 1, max_reminders)

            return GuardVerdict.inject(
                message=f"⚠️ {reminder}",
                reason=f"Warning [{wid}]: {warning.get('description', '')}",
            )

        return None

    def _is_triggered(self, warning: dict, ctx: GuardContext) -> bool:
        """Deterministic trigger check (no LLM cost).

        Returns True if the warning's trigger conditions match the context.
        """
        trigger = warning.get("trigger", {})
        if not isinstance(trigger, dict):
            return False

        # Tool filter
        tools = trigger.get("tools", [])
        if tools and ctx.tool_name not in tools:
            return False

        # Keyword filter
        keywords = trigger.get("keywords", [])
        if keywords:
            # Build searchable text from tool args
            search_text = " ".join(str(v) for v in ctx.tool_args.values())
            if ctx.tool_result:
                search_text += " " + ctx.tool_result
            search_text = search_text.lower()

            if not any(kw.lower() in search_text for kw in keywords):
                return False

        # If no trigger conditions defined, don't trigger
        if not tools and not keywords:
            return False

        return True

    def _judge_warning(self, warning: dict, ctx: GuardContext) -> bool:
        """Use LLM to judge if warning actually applies.

        Returns True if warning should fire, False otherwise.
        Without classify_fn, conservatively returns True.
        """
        if not ctx.classify_fn:
            return True

        prompt = warning.get("prompt", "")
        if not prompt:
            return True

        judge_context = {
            "warning": warning.get("description", ""),
            "prompt": prompt,
            "tool_name": ctx.tool_name,
            "tool_args": str(ctx.tool_args),
        }
        if ctx.tool_result:
            judge_context["tool_result"] = ctx.tool_result[:2000]

        try:
            result = ctx.classify_fn("is_warning_triggered", judge_context)
            return bool(result)
        except Exception as e:
            logger.warning("Warning judge failed for %s: %s",
                           warning.get("id", "unknown"), e)
            # Conservative: trigger warning on error
            return True

    @property
    def warnings(self) -> list[dict]:
        """Currently loaded warning definitions."""
        return list(self._warnings)

    @property
    def reminder_counts(self) -> dict[str, int]:
        """Current reminder counts per warning."""
        return dict(self._reminder_counts)

    def reset_turn(self):
        """No per-turn reset — reminder counts accumulate across turns."""
