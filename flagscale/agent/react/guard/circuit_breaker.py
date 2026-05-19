"""CircuitBreakerGuard — prevents infinite retries by tripping on repeated errors."""

from __future__ import annotations

import time

from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale.agent.react.state_machine import AgentState


class CircuitBreakerGuard(Guard):
    """Circuit breaker: trips (blocks) when same error category repeats N times.

    States: closed (normal) → open (tripped) → half_open (probe) → closed/open.
    Activates in EXECUTING state with highest priority.
    """

    name = "circuit_breaker"
    priority = 8  # high priority, before safety(10)
    activate_on_states = {AgentState.EXECUTING}

    TRIP_THRESHOLD = 4       # consecutive same-category errors → trip
    COOLDOWN_ITERS = 3       # iterations to wait before half-open probe

    # Circuit states
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, trip_threshold: int = 4, cooldown_iters: int = 3):
        self._trip_threshold = trip_threshold
        self._cooldown_iters = cooldown_iters

        # Per-category state
        self._error_counts: dict[str, int] = {}  # category → consecutive count
        self._circuit_state: dict[str, str] = {}  # category → CLOSED/OPEN/HALF_OPEN
        self._trip_iteration: dict[str, int] = {}  # category → iteration when tripped
        self._current_iteration: int = 0
        self._last_error_category: str | None = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Skip pre-iteration check (no specific tool being attempted)
        if not ctx.tool_name:
            return None

        self._current_iteration += 1

        # Check if any circuit is open and would block this tool
        for category, state in self._circuit_state.items():
            if state == self.OPEN:
                trip_iter = self._trip_iteration.get(category, 0)
                elapsed = self._current_iteration - trip_iter

                if elapsed > self._cooldown_iters:
                    # Transition to half-open: allow one probe
                    self._circuit_state[category] = self.HALF_OPEN
                    return None
                else:
                    remaining = self._cooldown_iters - elapsed + 1
                    return GuardVerdict.block(
                        f"[CircuitBreaker] Tripped for '{category}' errors "
                        f"({self._error_counts.get(category, 0)} consecutive failures). "
                        f"Cooldown: {remaining} iteration(s) remaining. "
                        f"Change your approach or ask the user for guidance.",
                        reason=f"circuit_open_{category}",
                    )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_result:
            return None

        category = self._classify_error(ctx.tool_result)

        if category is None:
            # Success — close any half-open circuits
            for cat in list(self._circuit_state.keys()):
                if self._circuit_state[cat] == self.HALF_OPEN:
                    self._circuit_state[cat] = self.CLOSED
                    self._error_counts[cat] = 0
            self._last_error_category = None
            # Reset counts on success
            self._error_counts.clear()
            return None

        # Error detected
        self._last_error_category = category

        # If half-open and error recurs, re-trip
        if self._circuit_state.get(category) == self.HALF_OPEN:
            self._circuit_state[category] = self.OPEN
            self._trip_iteration[category] = self._current_iteration
            return GuardVerdict.inject(
                f"[CircuitBreaker] Half-open probe failed for '{category}'. "
                f"Circuit re-tripped. The same approach keeps failing.",
                reason=f"circuit_retrip_{category}",
            )

        # Increment consecutive count
        if category == self._last_error_category or not self._error_counts.get(category):
            self._error_counts[category] = self._error_counts.get(category, 0) + 1
        else:
            # Different category — reset this one
            self._error_counts[category] = 1

        # Check if threshold reached
        if self._error_counts.get(category, 0) >= self._trip_threshold:
            self._circuit_state[category] = self.OPEN
            self._trip_iteration[category] = self._current_iteration
            return GuardVerdict.inject(
                f"[CircuitBreaker] TRIPPED for '{category}' "
                f"({self._error_counts[category]} consecutive failures). "
                f"Blocking further attempts for {self._cooldown_iters} iterations. "
                f"You MUST try a different approach.",
                reason=f"circuit_trip_{category}",
            )

        return None

    def reset_turn(self):
        # Keep circuit state across turns (session-level)
        pass

    def _classify_error(self, result: str) -> str | None:
        """Classify error — delegates to shared classifier."""
        from flagscale.agent.react.guard.error_classifier import ErrorClassifierGuard
        return ErrorClassifierGuard._classify_static(result)

    @property
    def tripped_categories(self) -> list[str]:
        """Return list of currently tripped (open) categories."""
        return [cat for cat, state in self._circuit_state.items() if state == self.OPEN]

    @property
    def state_summary(self) -> dict[str, str]:
        """Return current state of all tracked categories."""
        return dict(self._circuit_state)
