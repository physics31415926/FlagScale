"""ProgressInterrupt — detects read-only stalls and lack of productive output.

Mirrors v1's _check_progress_gate with adaptive thresholds and three
intervention patterns:
1. Re-reading known files without discovering new ones
2. Repeated similar shell errors
3. Very long exploration without checkpointing
"""

from __future__ import annotations

from collections import Counter

from .base import Interrupt, Intervention, Observation


# Tools that are "read-only" (no state changes)
READ_ONLY_TOOLS = {
    "read_file", "find", "grep", "ls", "list_files",
    "memory_read", "memory_list", "plan_status", "web_fetch",
}

# Tools that are "productive" (state changes)
PRODUCTIVE_TOOLS = {
    "write_file", "edit_file", "shell", "memory_write",
    "plan_create", "plan_update", "workspace_experiment",
}


class ProgressInterrupt(Interrupt):
    """Detects read-only stalls and nudges agent toward productive action.

    activate_on: {"always"} — applies regardless of scene.

    Adaptive thresholds:
    - Normal mode: stale_threshold = 25
    - Porting mode: stale_threshold = 40 (porting requires reading many sources)
    - Debugging mode: stale_threshold = 30 (more lenient during debugging)
    """

    name = "progress"
    activate_on = {"always"}
    priority = 30

    # ── Self-owned state ──
    _consecutive_reads: int = 0
    _reads_since_last_new_file: int = 0
    _last_unique_file_count: int = 0
    _rereads_without_save: int = 0
    _read_files: set[str] = set()  # Per-turn set, reset each turn
    _progress_triggers: int = 0
    _recent_shell_errors: list[str] = []

    # ── Mode flags (set externally by WorkerAgent / ScenePreset) ──
    is_porting_mode: bool = False
    consecutive_train_failures: int = 0

    # ── Thresholds ──
    _STALE_THRESHOLD_NORMAL = 25
    _STALE_THRESHOLD_PORTING = 40
    _STALE_THRESHOLD_DEBUG = 30
    _STALE_EXTRA_FOR_BLOCK = 8
    _READS_HARD_CAP_NORMAL = 60
    _READS_HARD_CAP_PORTING = 80

    def check_pre(self, obs: Observation) -> Intervention | None:
        return None  # Progress is checked post-exec

    def check_post(self, obs: Observation) -> Intervention | None:
        session_files_count = 0  # populated by _get_session_context call

        # --- Reset on productive action ---
        if obs.tool_name in PRODUCTIVE_TOOLS:
            self._consecutive_reads = 0
            self._reads_since_last_new_file = 0
            self._rereads_without_save = 0
            self._progress_triggers = 0
            return None

        # --- Track read-only calls ---
        if obs.tool_name in READ_ONLY_TOOLS:
            self._consecutive_reads += 1

            # Track new-file discovery
            if obs.tool_name == "read_file":
                path = obs.tool_args.get("path", "") or obs.tool_args.get("file_path", "")
                if path and path not in self._read_files:
                    self._read_files.add(path)
                    self._reads_since_last_new_file = 0
                elif path:
                    self._reads_since_last_new_file += 1
                    self._rereads_without_save += 1

        # --- Track repeated shell errors ---
        if obs.tool_name == "shell" and obs.tool_result:
            if self._is_error(obs.tool_result):
                self._recent_shell_errors.append(obs.tool_result[-300:])
                if len(self._recent_shell_errors) > 5:
                    self._recent_shell_errors = self._recent_shell_errors[-5:]

        # Shell commands are exploratory — don't count toward staleness
        # (already handled above: shell ∉ READ_ONLY_TOOLS)

        # --- Determine adaptive threshold ---
        stale_threshold = self._STALE_THRESHOLD_NORMAL
        if self.is_porting_mode:
            stale_threshold = self._STALE_THRESHOLD_PORTING
        elif self.consecutive_train_failures >= 2:
            stale_threshold = self._STALE_THRESHOLD_DEBUG

        # --- Pattern 1: Re-reading without discovering anything new ---
        if self._reads_since_last_new_file >= stale_threshold:
            self._progress_triggers += 1
            if self._reads_since_last_new_file >= stale_threshold + self._STALE_EXTRA_FOR_BLOCK:
                return Intervention(
                    action="block",
                    message=(
                        f"⛔ [PROGRESS BLOCK] You've made {self._reads_since_last_new_file} "
                        f"calls without discovering any new files or producing output. "
                        "This suggests you're stuck.\n"
                        "Create a plan (plan_create) to organize what you know and "
                        "identify what's missing, then continue with focused goals."
                    ),
                    reason=f"extended staleness: {self._reads_since_last_new_file} reads",
                )
            else:
                return Intervention(
                    action="inject_msg",
                    message=(
                        "\n[PROGRESS NOTE] You've been re-reading known files without "
                        "discovering new information. If you're looking for something specific, "
                        "consider: what exact question are you trying to answer? "
                        "A memory_write of current findings can help clarify next steps."
                    ),
                    reason="re-reading known files",
                )

        # --- Pattern 2: Repeated shell errors ---
        repeated_errors = self._count_repeated_recent_errors()
        if repeated_errors >= 3:
            return Intervention(
                action="inject_msg",
                message=(
                    "\n[PROGRESS NOTE] Similar errors appearing repeatedly. "
                    "Consider stepping back to understand the root cause rather than "
                    "retrying variations of the same approach."
                ),
                reason=f"repeated errors: {repeated_errors}",
            )

        # --- Pattern 3: Very long exploration without checkpoint ---
        reads_hard_cap = self._READS_HARD_CAP_PORTING if self.is_porting_mode else self._READS_HARD_CAP_NORMAL
        if self._consecutive_reads >= reads_hard_cap and self._progress_triggers == 0:
            self._progress_triggers += 1
            return Intervention(
                action="inject_msg",
                message=(
                    "\n[CHECKPOINT SUGGESTION] You've done extensive exploration. "
                    "Consider a memory_write to persist key findings — this protects "
                    "against context compaction loss."
                ),
                reason=f"extended exploration: {self._consecutive_reads} reads",
            )

        return None

    def reset_turn(self):
        self._read_files.clear()

    def _count_repeated_recent_errors(self) -> int:
        """Count how many recent shell calls produced similar errors."""
        if len(self._recent_shell_errors) < 2:
            return 0
        recent = self._recent_shell_errors[-5:]
        if len(recent) < 2:
            return 0
        last_words = set(recent[-1].lower().split()[:20])
        similar = sum(
            1 for e in recent[:-1]
            if len(set(e.lower().split()[:20]) & last_words) > len(last_words) * 0.5
        )
        return similar + 1  # Include the last one

    @staticmethod
    def _is_error(result: str) -> bool:
        lower = result.lower()
        return any(kw in lower for kw in ["error", "traceback", "exception", "failed", "fatal"])
