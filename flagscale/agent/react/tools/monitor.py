"""Monitor tool — long-running local polling without LLM calls.

The agent calls this tool to declare "I want to watch this file/command
until something interesting happens." The system then polls locally,
returning to the LLM only when a meaningful change is detected or timeout.
"""

import glob
import os
import re
import subprocess
import time

from flagscale.agent.react.tools.base import Tool
from flagscale.agent.react.tools.find_log import _last_sorted_subdir, _numeric_key


_INTERESTING_RE = re.compile(
    r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
    r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
    r'torch\.cuda\.OutOfMemoryError|CUDA error|'
    r'saved\s+checkpoint|training\s+complete|finished|'
    r'gradient_accumulation_fusion|No such file.*TransformerEngine|'
    r'CUDA extension not found|flash_attn.*not.*installed|'
    r'AssertionError|ValueError.*config|FileNotFoundError',
    re.IGNORECASE,
)

_METRIC_RE = re.compile(
    r'step[=:\s]|iteration[=:\s]|loss[=:\s]|grad.norm|throughput|MFU',
    re.IGNORECASE,
)

_TRAIN_PROCESS_RE = re.compile(
    r'torchrun|python.*train|flagscale|megatron|deepspeed',
    re.IGNORECASE,
)


class MonitorTool(Tool):
    name = "monitor"
    description = (
        "Watch a file or command output locally WITHOUT calling the LLM. "
        "Use this when you need to wait for training progress, model loading, "
        "or any long-running process. The tool polls locally and only returns "
        "when: (1) an error/completion pattern is detected, (2) new training "
        "metrics appear, (3) the timeout is reached, (4) the target step is hit, "
        "or (5) the monitored process has died. "
        "IMPORTANT: For FlagScale training, use 'output_dir' to auto-scan all "
        "rank stderr.log files for errors — this catches crashes that don't "
        "appear in stdout. "
        "This saves tokens by avoiding repeated LLM calls during waiting."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Path to the log file to watch (e.g., results/log.txt or train.log).",
            },
            "command": {
                "type": "string",
                "description": (
                    "Shell command to poll instead of watching a file. "
                    "Use either 'file' or 'command', not both."
                ),
            },
            "output_dir": {
                "type": "string",
                "description": (
                    "FlagScale output directory (e.g., outputs/qwen3_0_6b_dp_tp). "
                    "When set, the monitor auto-discovers the latest run's log directory "
                    "and scans ALL rank stderr.log files for errors each poll cycle. "
                    "This is the recommended way to monitor FlagScale training."
                ),
            },
            "duration": {
                "type": "integer",
                "description": "Maximum monitoring duration in seconds. Default: 300 (5 min). Max: 1800 (30 min).",
            },
            "interval": {
                "type": "integer",
                "description": "Polling interval in seconds. Default: 30.",
            },
            "target_step": {
                "type": "integer",
                "description": "Stop and return when training reaches this step number.",
            },
            "success_pattern": {
                "type": "string",
                "description": "Regex pattern — return immediately when matched (e.g., 'step=0000100').",
            },
            "fail_pattern": {
                "type": "string",
                "description": "Regex pattern — return immediately on match, flagged as error.",
            },
            "process_pattern": {
                "type": "string",
                "description": (
                    "Regex to check process liveness via pgrep -f. "
                    "If the process dies and no new output appears, monitoring stops early. "
                    "Default: auto-detect from 'torchrun|python.*train|flagscale'."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, display_fn=None):
        self._display_fn = display_fn

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file", "")
        command = kwargs.get("command", "")
        output_dir = kwargs.get("output_dir", "")
        duration = min(kwargs.get("duration", 300), 1800)
        interval = max(kwargs.get("interval", 30), 5)
        target_step = kwargs.get("target_step")
        success_pattern = kwargs.get("success_pattern", "")
        fail_pattern = kwargs.get("fail_pattern", "")
        process_pattern = kwargs.get("process_pattern", "")

        # If output_dir is given, auto-discover the log file to watch
        if output_dir and not file_path and not command:
            # Wait up to 30s for logs to appear (handles nohup race condition)
            discovered = None
            for _wait in range(6):
                discovered = self._discover_flagscale_logs(output_dir)
                if not discovered.get("error"):
                    break
                time.sleep(5)
            if discovered.get("error"):
                return discovered["error"]
            file_path = discovered.get("stdout_log", "")
            if not file_path:
                return f"ERROR: No stdout log found in {output_dir}. Check if training has started."

        if not file_path and not command:
            return "ERROR: Provide 'file', 'command', or 'output_dir' to monitor."

        success_re = re.compile(success_pattern) if success_pattern else None
        fail_re = re.compile(fail_pattern) if fail_pattern else None

        start = time.time()
        poll_count = 0
        last_content = ""
        last_line_count = 0
        events = []
        no_change_cycles = 0
        stderr_checked = {}  # track stderr sizes to detect new errors

        # Discover stderr logs for FlagScale output_dir
        stderr_logs = []
        if output_dir:
            discovered = self._discover_flagscale_logs(output_dir)
            stderr_logs = discovered.get("stderr_logs", [])

        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                events.append(f"[timeout after {int(elapsed)}s, {poll_count} polls]")
                break

            # Get current output
            if file_path:
                current = self._read_file(file_path)
            else:
                current = self._run_command(command)

            poll_count += 1

            if current != last_content:
                no_change_cycles = 0
                new_lines = self._get_new_lines(last_content, current)

                # Check fail pattern
                if fail_re:
                    for line in new_lines:
                        if fail_re.search(line):
                            events.append(f"[FAIL pattern matched at {int(elapsed)}s]")
                            return self._format_result(
                                "error_detected", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check success pattern
                if success_re:
                    for line in new_lines:
                        if success_re.search(line):
                            events.append(f"[SUCCESS pattern matched at {int(elapsed)}s]")
                            return self._format_result(
                                "success", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check target step
                if target_step is not None:
                    for line in new_lines:
                        step_match = re.search(r'step[=:\s]*0*(\d+)', line, re.IGNORECASE)
                        if step_match and int(step_match.group(1)) >= target_step:
                            events.append(f"[target step {target_step} reached at {int(elapsed)}s]")
                            return self._format_result(
                                "target_reached", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check for errors
                error_lines = [l for l in new_lines if _INTERESTING_RE.search(l)]
                if error_lines:
                    events.append(f"[interesting change at {int(elapsed)}s: {len(error_lines)} lines]")
                    return self._format_result(
                        "interesting_change", poll_count, elapsed,
                        events, new_lines[-20:], current
                    )

                # Check for new metrics (record but don't break)
                metric_lines = [l for l in new_lines if _METRIC_RE.search(l)]
                if metric_lines:
                    events.append(f"[+{len(metric_lines)} metric lines at {int(elapsed)}s]")

                last_content = current
                current_lines = current.strip().splitlines()
                last_line_count = len(current_lines)
            else:
                no_change_cycles += 1

            # FlagScale stderr scan — check ALL rank stderr.log files for errors
            if stderr_logs and poll_count % 2 == 1:
                stderr_error = self._scan_stderr_logs(stderr_logs, stderr_checked, elapsed)
                if stderr_error:
                    events.append(stderr_error["event"])
                    return self._format_result(
                        "stderr_error", poll_count, elapsed,
                        events, stderr_error["lines"], current
                    )

            # Process liveness check — every 2nd poll cycle when no new output
            if no_change_cycles >= 2 and poll_count % 2 == 0:
                if not self._is_process_alive(process_pattern):
                    # Before declaring dead, do one final stderr scan
                    if stderr_logs:
                        stderr_error = self._scan_stderr_logs(stderr_logs, stderr_checked, elapsed)
                        if stderr_error:
                            events.append(stderr_error["event"])
                            return self._format_result(
                                "stderr_error", poll_count, elapsed,
                                events, stderr_error["lines"], current
                            )
                    events.append(f"[process DEAD at {int(elapsed)}s, no new output for {no_change_cycles} cycles]")
                    return self._format_result(
                        "process_dead", poll_count, elapsed,
                        events, self._tail_lines(current, 20), current
                    )

            # Display progress
            if self._display_fn:
                self._display_fn(poll_count, elapsed, last_line_count)

            time.sleep(interval)

        # Timeout — return final state with summary
        return self._format_result(
            "timeout", poll_count, time.time() - start,
            events, self._tail_lines(current, 20), current
        )

    def _discover_flagscale_logs(self, output_dir):
        """Discover the latest FlagScale run's log files.

        Reuses find_log utilities for directory traversal.
        FlagScale log structure:
          outputs/<exp>/logs/details/host_<N>_<hostname>/<timestamp>/<run_name>/attempt_<N>/<rank>/
            - stdout.log
            - stderr.log
        """
        result = {"stdout_log": "", "stderr_logs": [], "error": ""}
        logs_dir = os.path.join(output_dir, "logs", "details")
        if not os.path.isdir(logs_dir):
            result["error"] = f"ERROR: No logs directory at {logs_dir}. Training may not have started."
            return result

        # Find the most recent host/timestamp using _last_sorted_subdir
        host_dirs = sorted(glob.glob(os.path.join(logs_dir, "host_*")))
        if not host_dirs:
            result["error"] = f"ERROR: No host directories in {logs_dir}."
            return result

        # Find latest timestamp across all hosts
        latest_ts = ""
        for host_dir in host_dirs:
            ts_dir = _last_sorted_subdir(host_dir)
            if ts_dir and (not latest_ts or os.path.getmtime(ts_dir) > os.path.getmtime(latest_ts)):
                latest_ts = ts_dir

        if not latest_ts:
            result["error"] = f"ERROR: No run directories found in {logs_dir}."
            return result

        # Navigate: timestamp -> run_name -> latest attempt
        run_dir = _last_sorted_subdir(latest_ts)
        if not run_dir:
            result["error"] = f"ERROR: No run directory under {latest_ts}."
            return result

        attempt_dir = _last_sorted_subdir(run_dir, key=_numeric_key)
        if not attempt_dir:
            result["error"] = f"ERROR: No attempt directory under {run_dir}."
            return result

        # Collect all rank logs
        stderr_logs = []
        stdout_log = ""
        for entry in sorted(os.listdir(attempt_dir)):
            rank_dir = os.path.join(attempt_dir, entry)
            if not os.path.isdir(rank_dir):
                continue
            stderr_path = os.path.join(rank_dir, "stderr.log")
            stdout_path = os.path.join(rank_dir, "stdout.log")
            if os.path.isfile(stderr_path):
                stderr_logs.append(stderr_path)
            if os.path.isfile(stdout_path) and entry == "0":
                stdout_log = stdout_path
            elif os.path.isfile(stdout_path) and not stdout_log:
                stdout_log = stdout_path

        result["stdout_log"] = stdout_log
        result["stderr_logs"] = stderr_logs
        return result

    def _scan_stderr_logs(self, stderr_logs, checked_sizes, elapsed):
        """Scan all stderr.log files for new error content.

        Returns dict with 'event' and 'lines' if error found, else None.
        """
        for log_path in stderr_logs:
            try:
                size = os.path.getsize(log_path)
            except OSError:
                continue

            prev_size = checked_sizes.get(log_path, 0)
            if size <= prev_size:
                continue

            # New content in this stderr.log
            checked_sizes[log_path] = size
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(prev_size)
                    new_content = f.read(8192)  # Read up to 8KB of new content
            except Exception:
                continue

            if not new_content.strip():
                continue

            # Check for error patterns
            error_lines = [l for l in new_content.splitlines() if _INTERESTING_RE.search(l)]
            if error_lines:
                rank = self._extract_rank_from_path(log_path)
                return {
                    "event": f"[STDERR ERROR rank {rank} at {int(elapsed)}s: {error_lines[0][:80]}]",
                    "lines": new_content.strip().splitlines()[-30:],
                }

            # Even without regex match, non-trivial stderr content is suspicious
            lines = new_content.strip().splitlines()
            if len(lines) > 3:
                rank = self._extract_rank_from_path(log_path)
                return {
                    "event": f"[STDERR activity rank {rank} at {int(elapsed)}s: {len(lines)} lines, possible error]",
                    "lines": lines[-30:],
                }

        return None

    @staticmethod
    def _extract_rank_from_path(path):
        """Extract rank number from FlagScale log path like .../attempt_0/6/stderr.log"""
        parts = path.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p == "stderr.log" and i > 0:
                return parts[i - 1]
        return "?"

    def _is_process_alive(self, process_pattern):
        """Check if the training process is still running."""
        pattern = process_pattern or r'torchrun|python.*train_|flagscale.*run'
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return True

    def _read_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception as e:
            return f"[read error: {e}]"

    def _run_command(self, cmd):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "[command timed out]"
        except Exception as e:
            return f"[command error: {e}]"

    @staticmethod
    def _get_new_lines(old, new):
        old_lines = old.strip().splitlines()
        new_lines = new.strip().splitlines()
        if len(new_lines) > len(old_lines):
            return new_lines[len(old_lines):]
        elif new_lines != old_lines:
            return new_lines[-10:]
        return []

    @staticmethod
    def _tail_lines(content, n=20):
        lines = content.strip().splitlines()
        return lines[-n:] if len(lines) > n else lines

    @staticmethod
    def _format_result(reason, poll_count, elapsed, events, recent_lines, full_content):
        parts = [f"Monitor result: {reason} ({poll_count} polls, {int(elapsed)}s)"]

        if events:
            parts.append("Events:")
            for e in events[-10:]:
                parts.append(f"  {e}")

        if recent_lines:
            parts.append("Recent output:")
            for line in recent_lines:
                parts.append(f"  {line}")

        # Extract latest metrics for quick reference
        metric_lines = [l for l in (full_content or "").splitlines() if _METRIC_RE.search(l)]
        if metric_lines:
            parts.append("Latest metrics:")
            for line in metric_lines[-3:]:
                parts.append(f"  {line.strip()}")

        return "\n".join(parts)
