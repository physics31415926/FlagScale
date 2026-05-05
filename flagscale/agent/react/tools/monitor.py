"""Monitor tool — long-running local polling without LLM calls.

The agent calls this tool to declare "I want to watch this file/command
until something interesting happens." The system then polls locally,
returning to the LLM only when a meaningful change is detected or timeout.
"""

import re
import subprocess
import time

from flagscale.agent.react.tools.base import Tool


_INTERESTING_RE = re.compile(
    r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
    r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
    r'torch\.cuda\.OutOfMemoryError|CUDA error|'
    r'saved\s+checkpoint|training\s+complete|finished',
    re.IGNORECASE,
)

_METRIC_RE = re.compile(
    r'step[=:\s]|iteration[=:\s]|loss[=:\s]|grad.norm|throughput|MFU',
    re.IGNORECASE,
)


class MonitorTool(Tool):
    name = "monitor"
    description = (
        "Watch a file or command output locally WITHOUT calling the LLM. "
        "Use this when you need to wait for training progress, model loading, "
        "or any long-running process. The tool polls locally and only returns "
        "when: (1) an error/completion pattern is detected, (2) new training "
        "metrics appear, (3) the timeout is reached, or (4) the target step is hit. "
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
        },
        "required": [],
    }

    def __init__(self, display_fn=None):
        self._display_fn = display_fn

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file", "")
        command = kwargs.get("command", "")
        duration = min(kwargs.get("duration") or 300, 1800)
        interval = max(kwargs.get("interval") or 30, 5)
        target_step = kwargs.get("target_step")
        success_pattern = kwargs.get("success_pattern", "")
        fail_pattern = kwargs.get("fail_pattern", "")

        if not file_path and not command:
            return "ERROR: Must provide either 'file' or 'command' parameter."

        if file_path and command:
            return "ERROR: Provide either 'file' or 'command', not both."

        success_re = re.compile(success_pattern) if success_pattern else None
        fail_re = re.compile(fail_pattern) if fail_pattern else None

        start = time.time()
        poll_count = 0
        last_content = ""
        last_line_count = 0
        events = []

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

            # Display progress
            if self._display_fn:
                self._display_fn(poll_count, elapsed, last_line_count)

            time.sleep(interval)

        # Timeout — return final state with summary
        return self._format_result(
            "timeout", poll_count, time.time() - start,
            events, self._tail_lines(current, 20), current
        )

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
