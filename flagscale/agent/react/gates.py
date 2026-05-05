"""Gate framework — extracted from agent.py for maintainability.

Gates are pre/post-tool checks that inject warnings or block execution.
This module provides the base class and the ProgressGate as a reference
implementation. Other gates can be migrated here incrementally.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Gate:
    """Base class for agent gates.

    A gate inspects tool calls and agent state, returning either:
    - ("", False): no intervention
    - (warning_text, False): soft warning (injected into result)
    - (block_text, True): hard block (tool not executed, block_text returned instead)
    """

    name: str = "base"

    def check(self, tool_name: str, arguments: dict, state: dict) -> Tuple[str, bool]:
        """Check gate condition. Override in subclasses.

        Args:
            tool_name: name of the tool being called
            arguments: tool arguments
            state: dict of relevant agent state (gate-specific keys)

        Returns:
            (message, is_hard_block)
        """
        return "", False

    def reset(self):
        """Reset gate state (called when productive work happens)."""
        pass


class ProgressGate(Gate):
    """Detects aimless exploration vs purposeful deep reading.

    Tracks staleness (re-reading without discovering new files) and
    adjusts thresholds based on mode (porting vs normal vs debugging).
    """

    name = "progress"

    PRODUCTIVE_TOOLS = {"write_file", "edit_file", "shell", "memory_write",
                        "workspace_experiment", "workspace_current", "plan_create",
                        "plan_update"}
    READ_ONLY_TOOLS = {"read_file", "find_latest_log", "memory_read", "memory_list",
                       "web_fetch", "plan_status"}

    def __init__(self):
        self.consecutive_reads = 0
        self.reads_since_last_new_file = 0
        self.last_unique_file_count = 0
        self.triggers = 0

    def reset(self):
        self.consecutive_reads = 0
        self.reads_since_last_new_file = 0
        self.triggers = 0

    def check(self, tool_name: str, arguments: dict, state: dict) -> Tuple[str, bool]:
        porting_mode = state.get("porting_mode", False)
        debugging = state.get("consecutive_train_failures", 0) >= 2
        has_plan = state.get("has_plan", False)
        files_read_count = state.get("files_read_count", 0)

        if tool_name in self.PRODUCTIVE_TOOLS:
            self.reset()
            return "", False

        if tool_name in self.READ_ONLY_TOOLS:
            self.consecutive_reads += 1

        # Track file discovery
        current_unique = files_read_count
        if current_unique > self.last_unique_file_count:
            self.reads_since_last_new_file = 0
            self.last_unique_file_count = current_unique
        else:
            self.reads_since_last_new_file += 1

        # Threshold selection based on mode
        stale_threshold = 12
        if porting_mode:
            stale_threshold = 30
        elif debugging:
            stale_threshold = 18

        # Pattern 1: Staleness
        if self.reads_since_last_new_file >= stale_threshold:
            self.triggers += 1
            if self.reads_since_last_new_file >= stale_threshold + 8:
                if not has_plan:
                    return (
                        f"⛔ [PROGRESS BLOCK — TOOL NOT EXECUTED] You've made "
                        f"{self.reads_since_last_new_file} calls without discovering "
                        f"any new files or producing output. This suggests you're stuck.\n"
                        "Create a plan (plan_create) to organize what you know and "
                        "identify what's missing, then continue with focused goals."
                    ), True
                else:
                    self.reads_since_last_new_file = 0
                    return (
                        "⛔ [PROGRESS BLOCK — TOOL NOT EXECUTED] Extended re-reading "
                        "without new discoveries. Record your current findings with "
                        "memory_write, then continue."
                    ), True
            else:
                return (
                    "\n\n[PROGRESS NOTE] You've been re-reading known files without "
                    "discovering new information. If you're looking for something specific, "
                    "consider: what exact question are you trying to answer? "
                    "A memory_write of current findings can help clarify next steps."
                ), False

        # Pattern 2: Very long exploration without checkpoint (safety net)
        reads_hard_cap = 60 if porting_mode else 40
        if self.consecutive_reads >= reads_hard_cap and self.triggers == 0:
            self.triggers += 1
            return (
                f"\n\n[CHECKPOINT SUGGESTION] You've done extensive exploration "
                f"({files_read_count} unique files read). "
                "Consider a memory_write to persist key findings — this protects "
                "against context compaction losing your discoveries."
            ), False

        return "", False


class GateRunner:
    """Runs a collection of gates and deduplicates warnings."""

    def __init__(self):
        self._gates: list = []
        self._last_warning = ""

    def register(self, gate: Gate):
        self._gates.append(gate)

    def reset_all(self):
        """Reset all gates (called on productive tool execution)."""
        for gate in self._gates:
            gate.reset()
        self._last_warning = ""

    def check_all(self, tool_name: str, arguments: dict, state: dict) -> Tuple[Optional[str], bool]:
        """Run all gates. Returns first hard block, or combined soft warnings.

        Deduplicates: same warning text won't be returned twice in a row.
        """
        warnings = []
        for gate in self._gates:
            msg, is_block = gate.check(tool_name, arguments, state)
            if is_block and msg:
                if msg != self._last_warning:
                    self._last_warning = msg
                    return msg, True
                return None, False  # Deduplicated block
            if msg and msg != self._last_warning:
                warnings.append(msg)

        if warnings:
            combined = "\n".join(warnings)
            self._last_warning = warnings[-1]
            return combined, False

        return None, False
