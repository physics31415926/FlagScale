"""Tool execution engine for FlagScale Agent.

Handles single and batch tool execution with:
- Deduplication of identical calls
- Batch size capping
- Guard pre-checks (block/inject)
- Checklist pre-checks
- Shell command confirmation
- Parallel execution with display
- Tool result caching within a turn
- File read/write tracking
- Skill auto-loading on load_skill
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from flagscale.agent.react import display
from flagscale.agent.react.constants import (
    READ_ONLY_TOOLS,
    READ_FILE_SUMMARY_THRESHOLD,
    READ_FILE_SUMMARY_THRESHOLD_PORTING,
)

if TYPE_CHECKING:
    from flagscale.agent.react.agent import WorkerAgent

logger = logging.getLogger(__name__)


def tool_display_summary(tool_name: str, arguments: dict) -> str:
    """Short human-readable summary for a tool call display."""
    if tool_name == "shell":
        cmd = arguments.get("command", "")
        s = cmd.replace("\n", " ").replace("\r", "").strip()
        return s[:120] + ("..." if len(s) > 120 else "")
    if tool_name == "read_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        return os.path.basename(path) if path else ""
    if tool_name == "edit_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        old = arguments.get("old_string", "")
        new = arguments.get("new_string", "")
        if old and new:
            old_val = old.split(":")[-1].strip() if ":" in old else ""
            new_val = new.split(":")[-1].strip() if ":" in new else ""
            if old_val and new_val and len(old_val) < 30 and len(new_val) < 30:
                return f"{os.path.basename(path)}: {old_val} → {new_val}"
            old_line = old.strip().split("\n")[0][:60]
            return f"{os.path.basename(path)}: {old_line}..."
        return os.path.basename(path) if path else ""
    if tool_name == "write_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        return os.path.basename(path) if path else ""
    if tool_name == "web_fetch":
        url = arguments.get("url", "")
        return url[:60] + ("..." if len(url) > 60 else "")
    if tool_name == "load_skill":
        return arguments.get("name", "")
    if tool_name == "workspace_experiment":
        action = arguments.get("action", "")
        name = arguments.get("name", "")
        return f"{action} {name}" if name else action
    if tool_name == "memory_write":
        return arguments.get("key", "")
    if tool_name == "plan_create":
        return arguments.get("title", "")
    if tool_name == "plan_update":
        action = arguments.get("action", "")
        step_id = arguments.get("step_id", "")
        return f"{action} step_{step_id}" if step_id else action
    if tool_name == "plan_status":
        return ""
    if tool_name == "monitor":
        tag = arguments.get("action", "") or arguments.get("tag", "")
        return tag or "check"
    if tool_name == "memory_read":
        return arguments.get("key", "")
    if tool_name == "grep":
        pattern = arguments.get("pattern", "")
        return pattern[:60] + ("..." if len(pattern) > 60 else "")
    return ""


def shell_display_summary(cmd: str, max_len: int = 90) -> str:
    """Short shell command display summary."""
    s = cmd.replace("\n", " ").replace("\r", "").strip()
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


# Keywords that indicate low-value verbose output (install/build logs)
_LOW_VALUE_KEYWORDS = (
    "installing", "collecting", "downloading", "successfully installed",
    "requirement already", "building wheel", "running setup", "compiling",
    "cloning into", "resolving deltas", "receiving objects",
)


def _compress_shell_result(result: str) -> str:
    """Compress large shell output at return time to save tokens.

    Only compresses clearly low-value output (install/build logs).
    Error output and code/config content are never compressed.
    """
    lines = result.splitlines()
    num_lines = len(lines)

    # Never compress if output has errors
    if any(kw in result[:500] for kw in ("Error", "ERROR", "Traceback", "FAILED", "Exception")):
        return result

    # Compress install/build/clone logs: keep command + outcome
    lower_head = result[:600].lower()
    if any(kw in lower_head for kw in _LOW_VALUE_KEYWORDS):
        head = "\n".join(lines[:3])
        tail = "\n".join(lines[-5:])
        return f"{head}\n[... {num_lines - 8} lines of output compressed ...]\n{tail}"

    # For other long output (>100 lines), keep head + tail
    if num_lines > 100:
        head = "\n".join(lines[:15])
        tail = "\n".join(lines[-15:])
        return f"{head}\n[... {num_lines - 30} lines omitted ({len(result)} chars total) ...]\n{tail}"

    return result


class ToolExecutor:
    """Executes tools with batching, dedup, guards, and parallel dispatch.

    This class encapsulates the tool execution lifecycle:
    1. Pre-checks (guards, checklist, confirmation)
    2. Deduplication and batch capping
    3. Parallel execution with display
    4. Post-execution tracking (files, skills, cache)
    """

    def __init__(self, agent: "WorkerAgent"):
        self._agent = agent

    def execute_batch(self, tool_calls: list[dict]) -> list[str]:
        """Execute a batch of tool calls with full pre-check pipeline.

        Single-call batches get efficiency reminders if read-only.
        Multi-call batches get dedup, guard checks, and parallel execution.
        """
        agent = self._agent

        if len(tool_calls) == 1:
            result = self.execute_single(tool_calls[0])
            tool_name = tool_calls[0]["name"]
            if tool_name in READ_ONLY_TOOLS:
                agent._consecutive_single_tool_calls += 1
                if 2 <= agent._consecutive_single_tool_calls <= 4:
                    result += (
                        "\n\n[EFFICIENCY REMINDER: You have made "
                        f"{agent._consecutive_single_tool_calls} consecutive single-tool responses. "
                        "Batch independent tool calls in ONE response to reduce round-trips.]"
                    )
            else:
                agent._consecutive_single_tool_calls = 0
            return [result]

        agent._consecutive_single_tool_calls = 0
        return self._execute_parallel(tool_calls)

    def execute_single(self, tool_call: dict, skip_confirm: bool = False) -> str:
        """Execute a single tool call with display, caching, and tracking."""
        agent = self._agent
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # Check turn cache
        cached_key = (tool_name, json.dumps(arguments, sort_keys=True))
        if cached_key in agent._tool_call_cache:
            logger.info("Cache hit for %s, skipping execution", tool_name)
            return agent._tool_call_cache[cached_key] + "\n[Cached result from earlier in this turn]"

        detail = tool_display_summary(tool_name, arguments)
        display.tool_start(tool_name, detail)
        t0 = time.time()
        try:
            if skip_confirm and tool_name == "shell":
                result = agent.tool_registry.execute(tool_name, _skip_confirm=True, **arguments)
            else:
                result = agent.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            result = f"ERROR: {e}"
        elapsed = time.time() - t0

        logger.info("Tool %s: %.1fs, result %d chars", tool_name, elapsed, len(result))
        error = False
        err_detail = ""
        if result:
            first_line = result.split('\n')[0] if '\n' in result else result[:120]
            if (first_line.startswith(("ERROR:", "FATAL:", "STALLED:", "TERMINATED:", "DENIED:"))
                    or "Traceback (most recent call last)" in result[:200]):
                error = True
                err_detail = first_line.split(":", 1)[-1].strip()[:80] if ":" in first_line else first_line[:80]
        display.tool_done(tool_name, elapsed, detail=err_detail, error=error)

        # Post-execution tracking
        result = self._post_execute(tool_name, arguments, result, error)

        # Cache for this turn
        agent._tool_call_cache[cached_key] = result
        return result

    def _post_execute(self, tool_name: str, arguments: dict, result: str, error: bool) -> str:
        """Handle post-execution side effects (tracking, skill loading, etc.)."""
        agent = self._agent

        # Immediate compression for large shell output (saves tokens before entering history)
        if tool_name == "shell" and not error and len(result) > 3000:
            result = _compress_shell_result(result)

        # Track file reads
        if tool_name == "read_file" and not error:
            path = arguments.get("path", "")
            if path:
                agent._files_read_this_session.add(path)
                threshold = READ_FILE_SUMMARY_THRESHOLD_PORTING if agent.modes.has("porting") else READ_FILE_SUMMARY_THRESHOLD
                if len(result) > threshold:
                    result = agent._summarize_file_content(result, path)

        # Track writes
        if tool_name in ("write_file", "edit_file") and not error:
            agent._last_write_turn = agent.turn_count
            agent._code_written = True
            path = arguments.get("path", "") or arguments.get("file_path", "")
            if path:
                agent._files_written_this_session.add(path)

        # Track load_skill side effects
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if skill_name not in agent._loaded_skills:
                agent._loaded_skills.add(skill_name)
                skill_content = result
                prefix_end = result.find("\n\n")
                if prefix_end != -1 and result.startswith("SUCCESS:"):
                    skill_content = result[prefix_end + 2:]
                agent._active_skill_content[skill_name] = skill_content
                agent._skill_load_iterations[skill_name] = agent._total_iterations
                agent._apply_skill_effects(skill_name)
                agent._on_skill_loaded(skill_name, skill_content)
                result = f"[Skill '{skill_name}' loaded — content available in system context]"

        return result

    def _execute_parallel(self, tool_calls: list[dict]) -> list[str]:
        """Execute multiple tool calls with dedup, guards, and parallel dispatch."""
        agent = self._agent

        # Dedup
        seen_calls = {}
        dedup_indices = set()
        for i, tc in enumerate(tool_calls):
            key = (tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True))
            if key in seen_calls:
                dedup_indices.add(i)
            else:
                seen_calls[key] = i

        _MAX_BATCH = 20
        capped_indices = set()
        if len(tool_calls) > _MAX_BATCH:
            logger.warning("Batch has %d tool calls, capping to %d", len(tool_calls), _MAX_BATCH)
            for i in range(_MAX_BATCH, len(tool_calls)):
                capped_indices.add(i)

        # Pre-exec: Guard + Checklist checks
        skip_indices: set[int] = set()
        results = [None] * len(tool_calls)

        for i, tc in enumerate(tool_calls):
            # Checklist pre-check
            blocked_by_checklist = False
            if agent.checklist:
                for alert in agent.checklist.pre_check_tool(tc["name"], tc.get("arguments", {})):
                    if alert.severity == "error":
                        blocked_by_checklist = True
                        print(display.red(f"  🛡 ⛔ BLOCKED by checklist: {alert.message}"))
                        agent._inject_message(
                            f"[CHECKLIST BLOCK] {alert.message}\n"
                            "This command was BLOCKED by a checklist constraint. "
                            "You must either:\n"
                            "  1. Change the command to comply with the constraint\n"
                            "  2. Declare an override if you believe this is a false positive: "
                            "[CHECKLIST_OVERRIDE: <constraint_id>] Reason: <justification>"
                        )
                    else:
                        print(display.yellow(f"  🛡 {alert.message}"))
                        agent._inject_message(alert.message)

            if blocked_by_checklist:
                skip_indices.add(i)
                continue

            # Guard pre-checks
            from flagscale.agent.react.guard import GuardContext
            from flagscale.agent.react.tools.base import ToolEffect
            tool_effects = ToolEffect()
            try:
                tool = agent.tool_registry.get(tc["name"])
                tool_effects = tool.effects
            except (KeyError, AttributeError):
                pass
            guard_ctx = GuardContext(
                tool_name=tc["name"],
                tool_args=tc.get("arguments", {}),
                tool_effects=tool_effects,
                turn_count=agent.turn_count,
                context_pressure=agent.history.get_context_pressure() if agent.history else 0.0,
                current_state=agent._kernel.fsm.current_state,
                transitions_count=len(agent._kernel.fsm.history),
                classify_fn=agent.judge.classify,
            )
            verdict = agent._kernel.deps.guard_registry.check_pre(guard_ctx)
            if verdict and verdict.action == "block":
                skip_indices.add(i)
                results[i] = f"⛔ TOOL NOT EXECUTED — blocked: {verdict.message}"
                break

        # Pre-confirm shell commands
        shell_tool = agent.tool_registry.get("shell")
        denied = set()
        if shell_tool:
            for i, tc in enumerate(tool_calls):
                if i in skip_indices:
                    continue
                if tc["name"] == "shell":
                    cmd = tc["arguments"].get("command", "")
                    if shell_tool.needs_confirm(cmd):
                        if not shell_tool.pre_confirm(cmd):
                            denied.add(i)

        skip_indices |= denied | dedup_indices | capped_indices

        # Serialize non-read shell commands
        write_shell_indices = []
        for i, tc in enumerate(tool_calls):
            if i in skip_indices:
                continue
            if tc["name"] == "shell":
                cmd = tc["arguments"].get("command", "")
                if not agent.judge.classify("is_read_only_shell", {"command": cmd}, default=False):
                    write_shell_indices.append(i)
        if len(write_shell_indices) > 1:
            for idx in write_shell_indices[1:]:
                skip_indices.add(idx)
                results[idx] = (
                    "[PARALLEL WRITE BLOCK — COMMAND NOT EXECUTED]\n\n"
                    "Non-read shell commands cannot run in parallel. "
                    "Issue them sequentially in separate responses.\n"
                )

        for i in denied:
            results[i] = "DENIED: User declined to execute this command."
        for i in dedup_indices:
            orig = seen_calls[(tool_calls[i]["name"], json.dumps(tool_calls[i].get("arguments", {}), sort_keys=True))]
            results[i] = f"[DEDUP: identical to call #{orig + 1} in this batch, skipped]"
        for i in capped_indices:
            results[i] = f"[BATCH CAPPED — TOOL NOT EXECUTED] Only {_MAX_BATCH} tool calls allowed per response."

        to_run = [(i, tc) for i, tc in enumerate(tool_calls) if i not in skip_indices]

        # Start parallel display
        summaries = [(tc["name"], tool_display_summary(tc["name"], tc.get("arguments", {})))
                     for _, tc in to_run]
        display.parallel_tools_start(summaries)

        idx_to_line = {orig_i: line_i for line_i, (orig_i, _) in enumerate(to_run)}

        def _run_quiet(idx, tc):
            tool_name = tc["name"]
            arguments = tc["arguments"]
            t0 = time.time()
            try:
                if tool_name == "shell":
                    result = agent.tool_registry.execute(
                        tool_name, _skip_confirm=True,
                        _parallel_index=idx_to_line[idx], **arguments)
                else:
                    result = agent.tool_registry.execute(tool_name, **arguments)
            except Exception as e:
                result = f"ERROR: {e}"
            elapsed = time.time() - t0
            logger.info("Tool %s: %.1fs, result %d chars", tool_name, elapsed, len(result))
            error = "ERROR" in result[:20] if result else False
            detail = ""
            if error and result:
                raw = result.split('\n')[0].replace("ERROR:", "").strip()
                detail = (raw[:57] + "...") if len(raw) > 60 else raw
            display.parallel_tool_update(idx_to_line[idx], elapsed, error, detail)
            return result

        if not to_run:
            display.parallel_tools_finish()
            return results

        with ThreadPoolExecutor(max_workers=min(len(to_run), 4)) as pool:
            futures = {pool.submit(_run_quiet, i, tc): i for i, tc in to_run}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        display.parallel_tools_finish()
        return results
