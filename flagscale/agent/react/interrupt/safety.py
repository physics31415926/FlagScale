"""SafetyInterrupt — dangerous command detection, self-kill protection, error escalation.

Mirrors v1's _check_error_escalation and dangerous command guards.
"""

from __future__ import annotations

import time

from .base import Interrupt, Intervention, Observation


class SafetyInterrupt(Interrupt):
    """Detects dangerous commands and escalating error patterns.

    activate_on: {"always"} — applies regardless of scene.
    Checked first (priority=10).
    """

    name = "safety"
    activate_on = {"always"}
    priority = 10

    # ── Self-owned state ──
    _last_tool_had_error: bool = False
    _root_cause_recorded_since_error: bool = False
    _recent_shell_errors: list[str] = []
    _consecutive_errors: int = 0

    # ── Escalation thresholds ──
    _ERROR_ESCALATE_WARN = 3
    _ERROR_ESCALATE_HARD = 5

    # ── Dangerous command patterns ──
    _DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+~",
        r"chmod\s+777\s+/",
        r":\(\)\s*\{",  # fork bomb
        r"mkfs\.",
        r"dd\s+if=",
        r">\s*/dev/sd",
    ]

    def check_pre(self, obs: Observation) -> Intervention | None:
        # Check for dangerous commands in shell calls
        if obs.tool_name == "shell":
            cmd = obs.tool_args.get("command", "")
            for pat in self._DANGEROUS_PATTERNS:
                if self._matches(pat, cmd):
                    return Intervention(
                        action="block",
                        message=(
                            "[Safety] Dangerous command detected and blocked. "
                            "If this is intentional, explain why and use a "
                            "more targeted approach."
                        ),
                        reason=f"dangerous pattern: {pat}",
                    )
        return None

    def check_post(self, obs: Observation) -> Intervention | None:
        result = obs.tool_result or ""

        # Detect if this tool produced an error
        if self._is_error_output(result):
            self._last_tool_had_error = True
            self._consecutive_errors += 1
            self._recent_shell_errors.append(result[-300:])

            # Check if root cause has been recorded since error
            if obs.tool_name in ("memory_write",):
                self._root_cause_recorded_since_error = True

            # --- Escalation ---
            if self._consecutive_errors >= self._ERROR_ESCALATE_HARD:
                return Intervention(
                    action="escalate",
                    message=(
                        f"[Safety] {self._consecutive_errors} consecutive tool errors. "
                        "The current approach is not working. Stop, diagnose the root "
                        "cause, and reformulate your strategy before continuing."
                    ),
                    reason=f"hard escalation: {self._consecutive_errors} errors",
                )

            if self._consecutive_errors >= self._ERROR_ESCALATE_WARN:
                if not self._root_cause_recorded_since_error:
                    return Intervention(
                        action="inject_msg",
                        message=(
                            f"[Safety] {self._consecutive_errors} consecutive tool errors "
                            "without recording root cause. Use memory_write to document "
                            "what's failing and why before retrying."
                        ),
                        reason="error escalation warn: no root cause recorded",
                    )
        else:
            self._last_tool_had_error = False

        # Track recovery: if training succeeds after failures, reset tracking
        if obs.tool_name == "shell" and self._is_success(result):
            self._recovery_from_failures = self._consecutive_errors
            self._consecutive_errors = 0

        return None

    def reset_turn(self):
        pass  # Safety/error state persists across turns

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _matches(pattern: str, text: str) -> bool:
        import re
        return bool(re.search(pattern, text))

    @staticmethod
    def _is_error_output(result: str) -> bool:
        lower = result.lower()
        return any(kw in lower for kw in [
            "error", "traceback", "exception", "failed", "fatal",
            "exitcode=1", "exitcode 1", "killed", "segfault",
            "cannot", "no such file", "permission denied",
        ])

    @staticmethod
    def _is_success(result: str) -> bool:
        """Check if the output indicates successful completion."""
        lower = result.lower()
        return "exitcode=0" in lower or "completed" in lower
