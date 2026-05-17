"""Interrupt ABC + Observation + Intervention dataclasses."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Observation:
    """Read-only snapshot at interrupt check time.

    This is ALL the context an Interrupt gets. No access to agent internals.
    """

    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None  # None for pre-exec
    turn_count: int = 0
    phase_name: str = "idle"
    recent_tool_names: list[str] = field(default_factory=list)
    context_pressure: float = 0.0
    # Injected context for richer interrupt logic
    experiment_compare_fn: callable | None = None  # (name1, name2) -> dict
    experiment_diff_fn: callable | None = None  # (name) -> dict
    current_experiment_name: str = ""


@dataclass
class Intervention:
    """What the interrupt wants the agent to do."""

    action: Literal["none", "block", "inject_msg", "force_compact", "escalate"]
    message: str = ""
    reason: str = ""


class Interrupt(abc.ABC):
    """A self-contained behavioral guard.

    Each Interrupt OWNS its state — no agent._xxx scatter.
    """

    # Subclass must override
    name: str = "base"
    activate_on: set[str] = {"always"}  # {"always"} = activates regardless of scene
    priority: int = 50  # Lower = earlier in check order

    def check_pre(self, obs: Observation) -> Intervention | None:
        """Called BEFORE tool execution. Return Intervention to act."""
        return None

    def check_post(self, obs: Observation) -> Intervention | None:
        """Called AFTER tool execution. Return Intervention to act."""
        return None

    def reset_turn(self):
        """Called at the start of each turn. Override to reset per-turn state."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
