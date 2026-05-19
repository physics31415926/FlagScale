"""LoopDetectGuard — detects repeated/looping tool calls."""

from __future__ import annotations

from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale.agent.react.state_machine import AgentState


class LoopDetectGuard(Guard):
    """Detects when the agent is looping on the same tool calls.

    Activates in EXECUTING state.
    """

    name = "loop_detect"
    priority = 20
    activate_on_states = {AgentState.EXECUTING}

    _MAX_RECENT = 10
    _LOOP_THRESHOLD = 3

    def __init__(self):
        self._recent_tool_calls: list[tuple[str, str]] = []
        self._tool_call_cache: dict[tuple[str, str], str] = {}

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        key_args = self._extract_key_args(ctx.tool_args)
        entry = (ctx.tool_name, key_args)

        self._recent_tool_calls.append(entry)
        if len(self._recent_tool_calls) > self._MAX_RECENT:
            self._recent_tool_calls = self._recent_tool_calls[-self._MAX_RECENT:]

        recent_same = sum(
            1 for t in self._recent_tool_calls[-self._MAX_RECENT:]
            if t == entry
        )
        if recent_same >= self._LOOP_THRESHOLD:
            return GuardVerdict.inject(
                f"[LoopDetect] Same tool call repeated {recent_same} times. "
                "The previous attempts did not produce the desired result. "
                "Try a different approach.",
                reason=f"looping on {ctx.tool_name}",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name:
            key_args = self._extract_key_args(ctx.tool_args)
            if ctx.tool_result:
                self._tool_call_cache[(ctx.tool_name, key_args)] = ctx.tool_result
        return None

    def reset_turn(self):
        self._tool_call_cache.clear()

    @staticmethod
    def _extract_key_args(args: dict) -> str:
        """Extract meaningful key arguments for dedup, skipping transient values."""
        skip_keys = {"timeout", "description", "run_in_background"}
        key_parts = []
        for k, v in sorted(args.items()):
            if k in skip_keys:
                continue
            val = str(v)[:80]
            key_parts.append(f"{k}={val}")
        return "|".join(key_parts)
