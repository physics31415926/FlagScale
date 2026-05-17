"""LoopDetectInterrupt — detects repeated/looping tool calls."""

from __future__ import annotations

from .base import Interrupt, Intervention, Observation


class LoopDetectInterrupt(Interrupt):
    """Detects when the agent is looping on the same tool calls.

    activate_on: {"always"} — applies regardless of scene.
    """

    name = "loop_detect"
    activate_on = {"always"}
    priority = 20

    # ── Self-owned state ──
    _recent_tool_calls: list[tuple[str, str]] = []  # [(tool_name, key_args), ...]
    _tool_call_cache: dict[tuple[str, str], str] = {}  # cache within turn

    _MAX_RECENT = 10
    _LOOP_THRESHOLD = 3  # Same call 3+ times in recent window

    def check_pre(self, obs: Observation) -> Intervention | None:
        if not obs.tool_name:
            return None

        # Build a key from tool_name + key arguments
        key_args = self._extract_key_args(obs.tool_args)
        entry = (obs.tool_name, key_args)

        self._recent_tool_calls.append(entry)
        if len(self._recent_tool_calls) > self._MAX_RECENT:
            self._recent_tool_calls = self._recent_tool_calls[-self._MAX_RECENT:]

        # Check for looping
        recent_same = sum(
            1 for t in self._recent_tool_calls[-self._MAX_RECENT:]
            if t == entry
        )
        if recent_same >= self._LOOP_THRESHOLD:
            return Intervention(
                action="inject_msg",
                message=(
                    f"[LoopDetect] Same tool call repeated {recent_same} times. "
                    "The previous attempts did not produce the desired result. "
                    "Try a different approach."
                ),
                reason=f"looping on {obs.tool_name}",
            )

        return None

    def check_post(self, obs: Observation) -> Intervention | None:
        # Cache the result
        if obs.tool_name:
            key_args = self._extract_key_args(obs.tool_args)
            if obs.tool_result:
                self._tool_call_cache[(obs.tool_name, key_args)] = obs.tool_result
        return None

    def reset_turn(self):
        self._tool_call_cache.clear()

    @staticmethod
    def _extract_key_args(args: dict) -> str:
        """Extract meaningful key arguments for dedup, skipping transient values."""
        # Skip shell commands' timing-dependent args, keep path/file names
        skip_keys = {"timeout", "description", "run_in_background"}
        key_parts = []
        for k, v in sorted(args.items()):
            if k in skip_keys:
                continue
            val = str(v)[:80]
            key_parts.append(f"{k}={val}")
        return "|".join(key_parts)
