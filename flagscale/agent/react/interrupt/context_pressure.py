"""ContextPressureInterrupt — monitors context window pressure and triggers compaction."""

from __future__ import annotations

from .base import Interrupt, Intervention, Observation


class ContextPressureInterrupt(Interrupt):
    """Monitors context pressure and triggers warnings / forced compaction.

    activate_on: {"always"} — applies regardless of scene.
    """

    name = "context_pressure"
    activate_on = {"always"}
    priority = 40

    # ── Self-owned state ──
    _soft_warned: bool = False   # 75% threshold
    _hard_warned: bool = False   # 85% threshold

    _SOFT_THRESHOLD = 0.75
    _HARD_THRESHOLD = 0.85
    _FORCE_THRESHOLD = 0.95

    def check_post(self, obs: Observation) -> Intervention | None:
        pressure = obs.context_pressure

        if pressure >= self._FORCE_THRESHOLD:
            return Intervention(
                action="force_compact",
                message="[ContextPressure] > 95% — forcing compaction now.",
                reason=f"pressure at {pressure:.0%}",
            )

        if pressure >= self._HARD_THRESHOLD and not self._hard_warned:
            self._hard_warned = True
            return Intervention(
                action="inject_msg",
                message=(
                    f"[ContextPressure] Context at {pressure:.0%}. "
                    "Write key findings to memory and request compaction via /compact."
                ),
                reason=f"hard threshold reached: {pressure:.0%}",
            )

        if pressure >= self._SOFT_THRESHOLD and not self._soft_warned:
            self._soft_warned = True
            return Intervention(
                action="inject_msg",
                message=(
                    f"[ContextPressure] Context at {pressure:.0%}. "
                    "Consider writing intermediate results to memory."
                ),
                reason=f"soft threshold reached: {pressure:.0%}",
            )

        return None

    def check_pre(self, obs: Observation) -> Intervention | None:
        return None  # Pressure is checked post-exec

    def reset_turn(self):
        pass  # Pressure warnings persist across turns
