"""Poll mode and error classification mixin."""

import json
import logging
import re
import time

from flagscale.agent.react import display

logger = logging.getLogger(__name__)


class PollMixin:
    """Token-saving poll mode: re-runs monitoring commands locally without LLM calls."""

    _MONITOR_CMD_RE = re.compile(
        r'^(tail|head|cat|wc|grep|ls)\b.*'
        r'(log|output|stdout|stderr|nohup\.out|train.*\.log)',
        re.IGNORECASE,
    )

    _INTERESTING_CHANGE_RE = re.compile(
        r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
        r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
        r'torch\.cuda\.OutOfMemoryError|CUDA error|'
        r'loss[=:\s]|grad.norm|throughput|step\s+\d|iteration\s+\d|'
        r'training\s+complete|finished|saved\s+checkpoint',
        re.IGNORECASE,
    )

    _TRAIN_CMD_RE = re.compile(
        r'flagscale\s+train|torchrun|python.*(?:train|verify|dryrun|test_model)|'
        r'python\s+.*run\.py\s+.*action\s*=\s*(?:run|dryrun)',
    )
    _TRAIN_LAUNCH_RE = re.compile(
        r'flagscale\s+train|torchrun\s|deepspeed\s|'
        r'python\s+.*(?:pretrain|finetune|train).*\.py|'
        r'python\s+.*run\.py\s+.*action\s*=\s*run|'
        r'nohup\s+.*(?:flagscale\s+train|torchrun|deepspeed)|'
        r'bash\s+-c\s+.*(?:flagscale\s+train|torchrun|deepspeed)',
    )
    _TRAIN_FAIL_RE = re.compile(
        r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
        r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
        r'torch\.cuda\.OutOfMemoryError|CUDA error',
        re.IGNORECASE,
    )

    # ── Error pattern classification ───────────────────────────────────

    _ERROR_PATTERNS = {
        "guardrail_warning": re.compile(r'\[GUARDRAIL\]|\[WARN:.*processes.*running\]', re.I),
        "import_error": re.compile(r'ModuleNotFoundError|ImportError|No module named', re.I),
        "runtime_extension_error": re.compile(
            r'RuntimeError.*(?:extension|fused_|apex|transformer.engine|gradient_accumulation_fusion|'
            r'flash_attn|CUDA.*not.*(?:compiled|available|built))', re.I),
        "oom": re.compile(r'OutOfMemoryError|CUDA out of memory|torch\.cuda\.OutOfMemoryError|'
                         r'\bOOM\b.*(?:killed|error|failed|crash)', re.I),
        "nccl_timeout": re.compile(r'NCCL.*timeout|NCCL.*hang|NCCL.*error', re.I),
        "shape_mismatch": re.compile(r'size mismatch|shape.*does not match|dimension.*mismatch', re.I),
        "checkpoint_load": re.compile(r'checkpoint.*not found|Failed to load|IncompatibleKeys|unexpected key', re.I),
        "data_pipeline": re.compile(r'DataLoader|dataset.*error|StopIteration|IndexError.*batch', re.I),
        "config_error": re.compile(r'invalid.*argument|unrecognized.*argument|TypeError.*__init__|'
                                   r'TP.*PP.*must|world_size|ValueError.*config', re.I),
        "cuda_error": re.compile(r'CUDA error|CUDA.*invalid|device-side assert', re.I),
        "device_mismatch": re.compile(r'Expected.*tensor.*on.*cuda|Expected.*device|'
                                      r'expected.*cuda.*got.*cpu|expected.*cpu.*got.*cuda', re.I),
        "permission_error": re.compile(r'PermissionError|Permission denied|EACCES', re.I),
        "disk_space": re.compile(r'No space left|disk quota|OSError.*28', re.I),
        "timeout": re.compile(r'TimeoutError|timed out|deadline exceeded', re.I),
        "key_error": re.compile(r"KeyError|missing key|key.*not found", re.I),
        "type_error": re.compile(r'TypeError.*argument|TypeError.*expected|got.*instead of', re.I),
    }

    _ERROR_SKILL_MAP = {
        "nccl_timeout": "parallel-strategy",
        "oom": "parallel-strategy",
        "runtime_extension_error": "env-setup",
        "shape_mismatch": "model-porter",
        "checkpoint_load": "model-porter",
        "config_error": "train-run",
        "data_pipeline": "data-prep",
        "import_error": "env-setup",
    }

    _ERROR_CLASSIFY_PROMPT = (
        "Classify this training error into ONE category. Reply with ONLY the category name.\n"
        "Categories: import_error, runtime_extension_error, oom, nccl_timeout, shape_mismatch, "
        "checkpoint_load, data_pipeline, config_error, cuda_error, device_mismatch, "
        "permission_error, disk_space, timeout, key_error, type_error, unknown\n\n"
        "Error:\n{error_text}\n\nCategory:"
    )

    # ── String constants for hints/warnings ────────────────────────────

    _CHECKPOINT_LOAD_RE = re.compile(
        r'--resume[_-]from|--finetune[_-]from|--load\s|--pretrained[_-]model|'
        r'--init[_-]checkpoint|--restore[_-]file',
    )

    _DRY_RUN_WARNING = (
        "\n⚠️ PRE-LAUNCH CHECK: This command loads a checkpoint but no validation run was done first.\n"
        "Principle: validate cheap things before expensive things. Run with --train-iters=20 first "
        "to verify the pipeline works before committing to a full run.\n"
        "NOTE: FlagScale --dryrun only generates scripts — it does NOT validate model/data loading.\n"
    )

    _EXPERIMENT_GATE_WARNING = (
        "\n⚠️ EXPERIMENT REGISTRY GATE: You launched a training run without "
        "creating an experiment entry first.\n"
        "This is a HARD REQUIREMENT. You MUST now:\n"
        "1. Call workspace_experiment(action='create', name='<exp_name>', purpose='...', config={...})\n"
        "2. Record the result when training completes or fails.\n"
    )

    _EXPERIMENT_UPDATE_REMINDER = (
        "\n⚠️ EXPERIMENT REGISTRY: Training ended (completed or failed). "
        "Update the experiment with workspace_experiment:\n"
        "1. Use add_attempt or update_last_attempt to record the result.\n"
        "2. If this experiment is done, use finalize to set status, root_cause, and learnings.\n"
    )

    _KNOWLEDGE_CAPTURE_HINT = (
        "\n\U0001f9e0 KNOWLEDGE CAPTURE: Training SUCCEEDED after {n} failure(s). "
        "Write the reusable fix to memory NOW: memory_write(key='<topic>', content='<root cause + fix>')\n"
    )

    _WORKAROUND_MEMORY_HINT = (
        "\n\U0001f9e0 WORKAROUND DETECTED: Previous call failed, this one succeeded. "
        "Write the fix to memory NOW: memory_write(key='<topic>', content='<root cause + fix>')\n"
    )

    _SESSION_MEMORY_REVIEW = (
        "\n\U0001f4dd SESSION REVIEW: Save any env quirks, version constraints, or workarounds "
        "discovered this session with memory_write.\n"
    )

    _TRAINING_MEMORY_HINT = (
        "\n\U0001f9e0 TRAINING LAUNCHED: If this succeeds after prior failures, "
        "remember to capture the fix with memory_write.\n"
    )

    _POST_LAUNCH_STDERR_HINT = (
        "\n⚠️ POST-LAUNCH: Wait 10-15s, then use monitor(output_dir='<exp_dir>', duration=120) "
        "to auto-discover logs and scan stderr. Do NOT use raw find commands (they find old logs). "
        "If stderr shows errors, training has ALREADY FAILED.\n"
    )

    _STALE_MEMORY_WARNING_TEMPLATE = (
        "\n⚠️ STALE MEMORIES: {count} memory entries are older than {days} days: {keys}. "
        "When you encounter these during work, verify they still hold. "
        "If outdated, update or delete them with memory_write / memory_read.\n"
    )

    # ── Poll mode methods ──────────────────────────────────────────────

    def _record_iteration(self, tool_calls, results, llm_output_tokens, tool_elapsed_list):
        """Record iteration metadata for poll pattern detection and error tracking."""
        if len(tool_calls) == 1 and tool_calls[0]["name"] == "shell":
            entry = {
                "tool_name": "shell",
                "command": tool_calls[0]["arguments"].get("command", "").strip(),
                "output": results[0] if results else "",
                "llm_output_tokens": llm_output_tokens,
                "tool_elapsed": tool_elapsed_list[0] if tool_elapsed_list else 0,
            }
        else:
            entry = None
        self._recent_iters.append(entry)
        if len(self._recent_iters) > self.config.poll_detect_window:
            self._recent_iters = self._recent_iters[-self.config.poll_detect_window:]

        if not hasattr(self, '_recent_shell_errors'):
            self._recent_shell_errors = []
        for tc, result in zip(tool_calls, results):
            if tc["name"] == "shell" and isinstance(result, str):
                if any(kw in result for kw in ("Error", "ERROR", "Traceback", "FAILED", "No such file")):
                    error_line = result.strip().splitlines()[-1][:150] if result.strip() else ""
                    if error_line:
                        self._recent_shell_errors.append(error_line)
                        if len(self._recent_shell_errors) > 10:
                            self._recent_shell_errors = self._recent_shell_errors[-10:]

    @staticmethod
    def _normalize_monitor_cmd(cmd):
        """Extract the target file from a monitoring command for fuzzy matching."""
        import shlex
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
        files = [p for p in parts if '/' in p or p.endswith(('.log', '.out', '.txt'))]
        return files[0] if files else cmd

    def _detect_poll_pattern(self):
        """Check if recent iterations form a polling pattern."""
        window = self.config.poll_detect_window
        if len(self._recent_iters) < window:
            return False
        recent = self._recent_iters[-window:]
        if any(r is None for r in recent):
            return False
        for r in recent:
            if r["llm_output_tokens"] > 200:
                return False
            if r["tool_elapsed"] > 5:
                return False
        commands = [r["command"] for r in recent]
        if len(set(commands)) == 1 and commands[0]:
            return True
        if all(self._MONITOR_CMD_RE.match(c) for c in commands):
            targets = [self._normalize_monitor_cmd(c) for c in commands]
            if len(set(targets)) == 1:
                return True
        return False

    @staticmethod
    def _poll_output_changed(old, new):
        """Compare two shell outputs, ignoring timestamp-like noise."""
        if old == new:
            return False
        old_lines = set(old.strip().splitlines())
        new_lines = set(new.strip().splitlines())
        if old_lines == new_lines:
            return False
        len_old = len(old.strip())
        len_new = len(new.strip())
        if len_old > 0 and abs(len_new - len_old) / max(len_old, 1) > 0.10:
            return True
        if new_lines - old_lines:
            return True
        return False

    @classmethod
    def _poll_output_interesting(cls, old, new):
        """Check if the change is interesting enough to return to LLM."""
        if not cls._poll_output_changed(old, new):
            return False
        old_lines = set(old.strip().splitlines())
        new_lines = new.strip().splitlines()
        added = [l for l in new_lines if l not in old_lines]
        for line in added:
            if cls._INTERESTING_CHANGE_RE.search(line):
                return True
        len_old = len(old.strip())
        len_new = len(new.strip())
        if len_old > 0 and abs(len_new - len_old) / max(len_old, 1) > 0.30:
            return True
        return False

    def _classify_error_pattern(self, error_text):
        """Classify error into pattern categories. Regex first, LLM fallback for unknown."""
        for pattern_name, pattern_re in self._ERROR_PATTERNS.items():
            if pattern_re.search(error_text):
                return pattern_name
        return self._llm_classify_error(error_text)

    def _llm_classify_error(self, error_text):
        """LLM fallback for error classification when regex patterns don't match."""
        try:
            prompt = self._ERROR_CLASSIFY_PROMPT.format(error_text=error_text[:500])
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            raw = response.get("content", "") if isinstance(response, dict) else ""
            category = raw.strip().lower().replace(" ", "_")
            if category in self._ERROR_PATTERNS or category == "unknown":
                return category
        except Exception as e:
            logger.debug("LLM error classification failed: %s", e)
        return "unknown"

    def _run_poll_mode(self, command, last_output, tool_call_id):
        """Execute poll loop locally without LLM calls."""
        interval = self.config.poll_interval
        max_duration = self.config.poll_max_duration
        cmd_display = self._shell_display_summary(command, max_len=70)
        display.poll_mode_start(cmd_display, interval)

        poll_count = 0
        routine_changes = 0
        start = time.time()
        current_output = last_output

        try:
            while True:
                elapsed = time.time() - start
                if elapsed >= max_duration:
                    return current_output, poll_count, elapsed, "timeout", routine_changes

                time.sleep(interval)
                poll_count += 1
                elapsed = time.time() - start

                try:
                    new_output = self.tool_registry.execute("shell", command=command, _skip_confirm=True)
                except Exception as e:
                    new_output = f"ERROR: {e}"

                if self._poll_output_interesting(current_output, new_output):
                    current_output = new_output
                    return current_output, poll_count, elapsed, "changed", routine_changes

                if self._poll_output_changed(current_output, new_output):
                    routine_changes += 1
                    current_output = new_output
                    display.poll_check(poll_count, elapsed, routine_change=True)
                else:
                    display.poll_check(poll_count, elapsed)
                    current_output = new_output

        except KeyboardInterrupt:
            elapsed = time.time() - start
            return current_output, poll_count, elapsed, "interrupted", routine_changes
