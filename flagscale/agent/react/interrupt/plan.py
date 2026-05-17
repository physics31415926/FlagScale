"""PlanInterrupt — complex task without plan detection.

Mirrors v1's _check_plan_creation_gate and _check_plan_maintenance_gate.

Two activation modes:
1. Complexity judge fired → _complex_task_no_plan = True, hard block at _PLAN_GATE_MAX_EXPLORATORY
2. Independent: warn at _PLAN_GATE_INDEPENDENT_WARN, hard block at _PLAN_GATE_INDEPENDENT_BLOCK
"""

from __future__ import annotations

from .base import Interrupt, Intervention, Observation


class PlanInterrupt(Interrupt):
    """Detects complex tasks without a plan and prompts plan creation.

    activate_on: {"always"} — applies regardless of scene.
    """

    name = "plan"
    activate_on = {"always"}
    priority = 35

    # ── Self-owned state ──
    _complex_task_no_plan: bool = False  # Set by ComplexityJudge externally
    _pre_plan_tool_calls: int = 0
    _consecutive_reads: int = 0  # For independent mode detection

    # ── Thresholds (from v1) ──
    _PLAN_GATE_MAX_EXPLORATORY = 6  # Complexity mode: block after 6 calls
    _PLAN_GATE_INDEPENDENT_WARN = 8  # Independent mode: warn
    _PLAN_GATE_INDEPENDENT_BLOCK = 12  # Independent mode: block

    def mark_complex_task(self):
        """Called externally (by ComplexityJudge) when a task needs a plan."""
        self._complex_task_no_plan = True

    def reset_plan_state(self):
        """Called externally when a plan is created."""
        self._complex_task_no_plan = False
        self._pre_plan_tool_calls = 0

    def check_pre(self, obs: Observation) -> Intervention | None:
        # Skips if plan already exists (checked externally via task_plan.get_active())

        # Plan-related tools are always allowed
        if obs.tool_name in ("plan_create", "memory_write", "workspace_experiment"):
            return None

        self._consecutive_reads += 1
        self._pre_plan_tool_calls += 1

        # Mode 1: complexity judge fired → hard block at threshold
        if self._complex_task_no_plan:
            if self._pre_plan_tool_calls > self._PLAN_GATE_MAX_EXPLORATORY:
                return Intervention(
                    action="block",
                    message=(
                        f"⛔ [PLAN GATE — TOOL NOT EXECUTED] This task was flagged "
                        f"as complex. You've used {self._pre_plan_tool_calls} exploratory "
                        f"calls (limit: {self._PLAN_GATE_MAX_EXPLORATORY}) without creating "
                        f"a plan.\n"
                        f"This tool call was BLOCKED. You MUST call plan_create NOW.\n"
                        f"Use what you've gathered so far to create a concrete step-by-step plan."
                    ),
                    reason="complex task no plan exceeded",
                )

        # Mode 2: independent — soft warn, then hard block
        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_BLOCK:
            return Intervention(
                action="block",
                message=(
                    f"⛔ [PLAN GATE — TOOL NOT EXECUTED] You've made "
                    f"{self._consecutive_reads} consecutive exploratory calls "
                    f"without creating a plan.\n"
                    f"This tool call was BLOCKED. You MUST call plan_create NOW "
                    f"to organize your approach."
                ),
                reason="independent plan threshold exceeded",
            )

        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_WARN:
            return Intervention(
                action="inject_msg",
                message=(
                    f"\n[PLAN REMINDER] You've made {self._consecutive_reads} "
                    f"exploratory calls without a plan. Consider calling plan_create "
                    f"soon to organize your findings. "
                    f"You will be BLOCKED at {self._PLAN_GATE_INDEPENDENT_BLOCK} calls."
                ),
                reason="plan independent warn threshold",
            )

        return None

    def check_post(self, obs: Observation) -> Intervention | None:
        # Detect if a plan was created
        if obs.tool_name in ("plan_create",):
            self._complex_task_no_plan = False
            self._pre_plan_tool_calls = 0

        # Plan maintenance: remind agent to update stale plan steps
        # (requires access to task_plan — checked externally and injected via Observation)
        return None

    def check_plan_staleness(self, task_plan, turn_count: int) -> Intervention | None:
        """Check if plan's 'doing' step is stale (>8 turns without update)."""
        plan = task_plan.get_active() if task_plan else None
        if not plan:
            return None

        doing_steps = [s for s in plan.get("steps", []) if s.get("status") == "doing"]
        if not doing_steps:
            return None

        step = doing_steps[0]
        last_activity = step.get("_last_activity_turn", 0)
        turns_stale = turn_count - last_activity if last_activity else 0

        if turns_stale >= 8:
            return Intervention(
                action="inject_msg",
                message=(
                    f"\n[PLAN MAINTENANCE] Step {step['id']} "
                    f"('{step.get('title', '')[:40]}') has had no plan_update "
                    f"for {turns_stale} turns. "
                    f"If it's done, call plan_update(action='step_done'). "
                    f"If blocked, call plan_update(action='step_skip') and move on."
                ),
                reason=f"plan step stale: {turns_stale} turns",
            )
        return None

    def reset_turn(self):
        pass
