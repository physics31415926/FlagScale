"""Find latest FlagScale training logs in one step."""

import os
import re
import subprocess
import time

from flagscale.agent.react.tools.base import Tool


def _last_sorted_subdir(parent: str, key=None):
    """Return the last subdirectory under parent when sorted by key."""
    if not os.path.isdir(parent):
        return ""
    entries = [e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e))]
    if not entries:
        return ""
    entries.sort(key=key)
    return os.path.join(parent, entries[-1])


def _numeric_key(name: str):
    """Extract trailing number for sorting: 'attempt_2' → 2, '7' → 7."""
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 0


class FindLatestLogTool(Tool):
    name = "find_latest_log"
    description = (
        "Find and display the latest FlagScale training log for an experiment. "
        "Navigates: last node → latest timestamp → last attempt → last rank. "
        "Returns the tail of stdout and/or stderr, saving multiple rounds of manual navigation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment name, e.g. qwen3_0_6b_train",
            },
            "log_type": {
                "type": "string",
                "enum": ["stdout", "stderr", "both"],
                "description": "Which log to show. Default: both",
            },
            "lines": {
                "type": "integer",
                "description": "Number of tail lines per log file. Default: 50",
            },
        },
        "required": ["experiment"],
    }

    def __init__(self, outputs_dir: str = ""):
        self._outputs_dir = outputs_dir or os.path.join(os.getcwd(), "outputs")

    def execute(self, **kwargs) -> str:
        experiment = kwargs["experiment"]
        log_type = kwargs.get("log_type", "both")
        lines = kwargs.get("lines", 50)

        exp_dir = os.path.join(self._outputs_dir, experiment)
        if not os.path.isdir(exp_dir):
            return f"ERROR: Experiment directory not found: {exp_dir}"

        details_dir = os.path.join(exp_dir, "logs", "details")
        if not os.path.isdir(details_dir):
            return f"ERROR: No logs/details directory in {exp_dir}"

        rank_dir = self._resolve_rank_dir(details_dir)
        if not rank_dir:
            return f"ERROR: Could not locate log directory under {details_dir}"

        targets = []
        if log_type in ("stdout", "both"):
            targets.append("stdout.log")
        if log_type in ("stderr", "both"):
            targets.append("stderr.log")

        results = []
        for target in targets:
            path = os.path.join(rank_dir, target)
            if not os.path.isfile(path):
                results.append(f"=== {target} ===\nNot found at {path}")
                continue
            mtime = os.path.getmtime(path)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            size = os.path.getsize(path)

            header = f"=== {target} === (modified: {ts}, size: {size} bytes)\nPath: {path}"
            try:
                out = subprocess.run(
                    ["tail", f"-{lines}", path],
                    capture_output=True, text=True, timeout=10,
                )
                content = out.stdout or "(empty)"
            except Exception as e:
                content = f"ERROR reading: {e}"
            results.append(f"{header}\n{content}")

        return "\n\n".join(results)

    def _resolve_rank_dir(self, details_dir: str) -> str:
        """Navigate: last node → latest timestamp → run_id → last attempt → last rank."""
        # Last node (host_0_xxx, host_1_xxx, ... → take last)
        node_dir = _last_sorted_subdir(details_dir)
        if not node_dir:
            return ""
        # Latest timestamp dir (sorted lexicographically = chronologically)
        ts_dir = _last_sorted_subdir(node_dir)
        if not ts_dir:
            return ""
        # Run id (default_xxx — usually only one, take last)
        run_dir = _last_sorted_subdir(ts_dir)
        if not run_dir:
            return ""
        # Last attempt (attempt_0, attempt_1, ... → sort by trailing number)
        attempt_dir = _last_sorted_subdir(run_dir, key=_numeric_key)
        if not attempt_dir:
            return ""
        # Last rank (0, 1, 2, ... → sort by number)
        rank_dir = _last_sorted_subdir(attempt_dir, key=_numeric_key)
        return rank_dir
