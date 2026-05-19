"""ContextPressureGuard — monitors context window pressure and triggers compaction."""

from __future__ import annotations

from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale.agent.react.state_machine import AgentState


class ContextPressureGuard(Guard):
    """Monitors context pressure and triggers warnings / forced compaction.

    Activates in all states — context pressure is always relevant.
    """

    name = "context_pressure"
    priority = 40
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING, AgentState.REVIEWING}

    # ── Thresholds ──
    _SOFT_THRESHOLD = 0.75
    _HARD_THRESHOLD = 0.85
    _FORCE_THRESHOLD = 0.95

    def __init__(self):
        self._soft_warned: bool = False
        self._hard_warned: bool = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        pressure = ctx.context_pressure

        if pressure >= self._FORCE_THRESHOLD:
            return GuardVerdict.compact(
                reason=f"pressure at {pressure:.0%}",
            )

        if pressure >= self._HARD_THRESHOLD and not self._hard_warned:
            self._hard_warned = True
            return GuardVerdict.inject(
                f"[ContextPressure] Context at {pressure:.0%}. "
                "Write key findings to memory and request compaction via /compact.",
                reason=f"hard threshold reached: {pressure:.0%}",
            )

        if pressure >= self._SOFT_THRESHOLD and not self._soft_warned:
            self._soft_warned = True
            return GuardVerdict.inject(
                f"[ContextPressure] Context at {pressure:.0%}. "
                "Consider writing intermediate results to memory.",
                reason=f"soft threshold reached: {pressure:.0%}",
            )

        return None

    def reset_turn(self):
        # Do NOT reset warned flags here — reset_turn is called per iteration.
        # Context pressure warnings should fire at most once per session threshold crossing.
        pass
