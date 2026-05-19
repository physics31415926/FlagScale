"""ConstraintGuard — enforces compiled Constraints via Guard lifecycle.

Design (REFACTOR_TRACKER D3):
1. Deterministic trigger: tool_name + keyword match (cheap, no LLM)
2. Precise judgment: only when triggered, call classify_fn (LLM)
3. Block behavior: violated constraints return block + correction
"""

from __future__ import annotations

import logging
from typing import Any

from flagscale.agent.react.constraint import Constraint, ConstraintViolation
from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale.agent.react.state_machine import AgentState

logger = logging.getLogger(__name__)


class ConstraintGuard(Guard):
    """Enforces a set of compiled Constraints.

    Lifecycle:
    - check_pre: enforces constraints with check_phase="pre"
    - check_post: enforces constraints with check_phase="post"

    Trigger strategy:
    - First: deterministic trigger check (ConstraintTrigger.matches)
    - Then: LLM precise judgment via classify_fn
    - Finally: block + correction on violation
    """

    name = "constraint"
    priority = 25  # After safety (10), before progress (30)
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING}

    def __init__(self, constraints: list[Constraint] | None = None):
        self._constraints: list[Constraint] = constraints or []
        self._violations: dict[str, int] = {}  # constraint_id -> violation count

    def add_constraints(self, constraints: list[Constraint]):
        """Add constraints (e.g., after skill load)."""
        self._constraints.extend(constraints)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check pre-phase constraints."""
        return self._check(ctx, phase="pre")

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check post-phase constraints."""
        return self._check(ctx, phase="post")

    def _check(self, ctx: GuardContext, phase: str) -> GuardVerdict | None:
        """Core enforcement logic.

        1. Filter constraints by check_phase
        2. For each: deterministic trigger check
        3. If triggered: LLM precise judgment
        4. If violated: block (error) or inject (warning)
        """
        for constraint in self._constraints:
            if constraint.check_phase != phase:
                continue

            # Step 1: Deterministic trigger (cheap)
            if not constraint.trigger.matches(ctx.tool_name, ctx.tool_args, ctx.tool_result):
                continue

            # Step 2: Precise judgment via LLM
            violated = self._judge_violation(ctx, constraint)
            if not violated:
                continue

            # Step 3: Record and act
            count = self._violations.get(constraint.id, 0) + 1
            self._violations[constraint.id] = count

            logger.info(
                "Constraint violated: %s (count=%d, severity=%s)",
                constraint.id, count, constraint.severity,
            )

            if constraint.severity == "error":
                return GuardVerdict.block(
                    message=constraint.correction,
                    reason=f"Constraint [{constraint.id}]: {constraint.description}",
                )
            else:
                # Warning: inject correction but don't block
                return GuardVerdict.inject(
                    message=f"⚠️ {constraint.correction}",
                    reason=f"Constraint [{constraint.id}]: {constraint.description}",
                )

        return None

    def _judge_violation(self, ctx: GuardContext, constraint: Constraint) -> bool:
        """Use LLM to precisely judge if a constraint is violated.

        Returns True if violated, False otherwise.
        Falls back to True (block) if no classify_fn available.
        """
        if not ctx.classify_fn:
            # No LLM available — conservative: assume violated
            return True

        judge_context = {
            "constraint": constraint.description,
            "prompt": constraint.prompt,
            "tool_name": ctx.tool_name,
            "tool_args": str(ctx.tool_args),
        }
        if ctx.tool_result:
            judge_context["tool_result"] = ctx.tool_result[:2000]

        try:
            result = ctx.classify_fn("is_constraint_violated", judge_context)
            return bool(result)
        except Exception as e:
            logger.warning("Constraint judge failed for %s: %s", constraint.id, e)
            # Conservative: assume violated on error
            return True

    @property
    def violations(self) -> dict[str, int]:
        """Current violation counts per constraint."""
        return dict(self._violations)

    @property
    def constraints(self) -> list[Constraint]:
        """Currently loaded constraints."""
        return list(self._constraints)

    def reset_turn(self):
        """No per-turn reset needed — violations accumulate across turns."""
