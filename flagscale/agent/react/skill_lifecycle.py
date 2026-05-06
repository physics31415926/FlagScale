"""Skill lifecycle management and error-to-skill auto-loading mixin."""

import logging
import re
import uuid

from flagscale.agent.react import display

logger = logging.getLogger(__name__)


class SkillLifecycleMixin:
    """Manages skill loading/unloading, phase transitions, and training failure tracking."""

    # ── Phase transitions ──────────────────────────────────────────────

    _PHASE_TRANSITIONS = {
        "analysis": {
            "exit_conditions": ["read_count >= 8", "categories >= 3", "analysis_persisted"],
            "next_phase": "implementation",
            "message": (
                "\n\n[PHASE TRANSITION] Analysis phase complete. Before moving to implementation:\n"
                "1. Have you read enough files (≥8) across categories (≥3/4)?\n"
                "2. Have you persisted your analysis (component mapping, memory budget)?\n"
                "3. Are you ready to write code?\n"
                "If yes, proceed. If no, continue analysis."
            ),
        },
        "implementation": {
            "exit_conditions": ["code_written", "dry_run_passed"],
            "next_phase": "verification",
            "message": (
                "\n\n[PHASE TRANSITION] Implementation phase complete. Before moving to verification:\n"
                "1. Have you written the porting code?\n"
                "2. Have you run a short validation training (--train-iters 20) successfully?\n"
                "   NOTE: FlagScale --dryrun only generates scripts — it does NOT validate the pipeline.\n"
                "If yes, proceed to verification. If no, continue implementation."
            ),
        },
    }

    def _check_skill_lifecycle(self):
        """Unload skills that are no longer needed to save tokens."""
        if not self._active_skill_content:
            return
        to_unload = []
        for skill_name, loaded_at_iter in list(self._skill_load_iterations.items()):
            age = self._total_iterations - loaded_at_iter
            if self._porting_mode and "model-porter" in skill_name:
                continue
            if self._data_prep_mode and "data-prep" in skill_name:
                continue
            if skill_name == "train-run" and self._training_started:
                to_unload.append(skill_name)
            elif skill_name == "env-setup" and self._env_verified:
                to_unload.append(skill_name)
            elif age > 30 and skill_name not in self._recently_referenced_skills:
                to_unload.append(skill_name)
        for name in to_unload:
            self._active_skill_content.pop(name, None)
            self._skill_load_iterations.pop(name, None)
            logger.info("Unloaded skill '%s' (no longer needed)", name)
        if to_unload:
            self._refresh_system_prompt()
        self._recently_referenced_skills.clear()

    def _auto_load_skill_for_error(self, error_text):
        """Auto-load relevant skill when a known error pattern is detected."""
        pattern = self._classify_error_pattern(error_text)
        skill_name = self._ERROR_SKILL_MAP.get(pattern)
        if not skill_name or skill_name in self._loaded_skills:
            return ""
        try:
            content = self.skill_manager.load(skill_name)
            content = self._maybe_summarize_skill(skill_name, content)
            tool_call_id = f"auto_err_{uuid.uuid4().hex[:8]}"
            fake_response = {
                "content": None,
                "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
            }
            self.history.append(self.provider.format_assistant_message(fake_response))
            self.history.append(self.provider.format_tool_result(
                tool_call_id, f"[Skill '{skill_name}' loaded — content available in system context]"))
            self._loaded_skills.add(skill_name)
            self._active_skill_content[skill_name] = content
            self._skill_load_iterations[skill_name] = self._total_iterations
            self._refresh_system_prompt()
            display.skill_auto_loaded(skill_name)
            return f"\n[Auto-loaded skill '{skill_name}' for {pattern} error]\n"
        except Exception:
            return ""

    def _check_phase_transition(self, tool_name):
        """Check if the agent should transition to the next phase."""
        if not self._porting_mode:
            return ""

        if tool_name in ("write_file", "edit_file"):
            self._code_written = True
        if tool_name == "shell":
            pass

        current_phase = getattr(self, "_current_phase", "analysis")
        transition = self._PHASE_TRANSITIONS.get(current_phase)
        if not transition:
            return ""

        if current_phase == "analysis":
            files_read = len(self._files_read_this_session)
            if files_read >= 8:
                self._current_phase = transition["next_phase"]
                return transition["message"]
        elif current_phase == "implementation":
            if self._code_written and getattr(self, "_dry_run_passed", False):
                self._current_phase = transition["next_phase"]
                return transition["message"]

        return ""

    def _maybe_summarize_skill(self, skill_name, content):
        """Summarize skill content if it's too long for the context window."""
        if len(content) <= 8000:
            return content
        if skill_name in getattr(self, "_skill_summaries", {}):
            return self._skill_summaries[skill_name]
        summary = self._auto_generate_skill_summary(content)
        if not hasattr(self, "_skill_summaries"):
            self._skill_summaries = {}
        self._skill_summaries[skill_name] = summary
        return summary

    @staticmethod
    def _auto_generate_skill_summary(content):
        """Extract key sections from skill content as a summary."""
        lines = content.split("\n")
        summary_lines = []
        in_section = False
        for line in lines:
            if line.startswith("#") or line.startswith("##"):
                in_section = True
                summary_lines.append(line)
            elif in_section and line.strip():
                summary_lines.append(line)
                if len(summary_lines) > 100:
                    break
            elif not line.strip():
                in_section = False
        return "\n".join(summary_lines[:100])

    def _summarize_file_content(self, content, path):
        """Summarize file content for context-constrained situations."""
        lines = content.split("\n")
        if len(lines) <= 50:
            return content
        header = lines[:5]
        imports = [l for l in lines[:30] if l.startswith(("import ", "from "))]
        classes = [l for l in lines if l.startswith("class ")]
        functions = [l for l in lines if l.startswith("def ") or l.startswith("    def ")][:20]
        summary = header + ["", "# --- Imports ---"] + imports
        if classes:
            summary += ["", "# --- Classes ---"] + classes
        if functions:
            summary += ["", "# --- Functions ---"] + functions
        summary += [f"\n# Total: {len(lines)} lines"]
        return "\n".join(summary)

    def _is_context_limit_error(self, e):
        """Check if an exception is a context length limit error."""
        msg = str(e).lower()
        return any(kw in msg for kw in ("context length", "maximum context", "token limit", "too many tokens", "400"))

    # ── Training failure tracking ──────────────────────────────────────

    def _track_training_failures(self, tool_calls, results):
        """Track consecutive training failures for escalation."""
        for tc, result in zip(tool_calls, results):
            tool_name = tc["name"]
            if not isinstance(result, str):
                continue

            if tool_name == "shell":
                cmd = tc["arguments"].get("command", "")
                if not self._TRAIN_CMD_RE.search(cmd):
                    continue
                is_verification = bool(re.search(r'verify|dryrun|test_model', cmd, re.I))
                if self._TRAIN_FAIL_RE.search(result[:2000]):
                    # LLM confirm: is this a real training failure?
                    if not self._regex_judge_confirm(
                            "is_training_failure", result[:500],
                            result[:2000]):
                        continue
                    self._consecutive_train_failures += 1
                    reason = result[:200].split('\n')[0]
                    self._last_train_failure_reasons.append(reason)
                    pattern = self._classify_error_pattern(result[:2000])
                    self._error_pattern_history.append(pattern)
                    self._record_and_escalate_failure(cmd, result, pattern, is_verification)
                else:
                    self._recovery_from_failures = self._consecutive_train_failures
                    self._consecutive_train_failures = 0
                    self._last_train_failure_reasons.clear()
                    self._error_pattern_history.clear()

            elif tool_name == "monitor":
                if any(kw in result[:500] for kw in ("process_dead", "stderr_error", "process DEAD")):
                    self._consecutive_train_failures += 1
                    reason = result[:200].split('\n')[0]
                    self._last_train_failure_reasons.append(reason)
                    pattern = self._classify_error_pattern(result[:2000])
                    self._error_pattern_history.append(pattern)
                    cmd = tc["arguments"].get("command", "") or tc["arguments"].get("output_dir", "monitor")
                    self._record_and_escalate_failure(cmd, result, pattern, False)
                elif any(kw in result[:500] for kw in ("target_reached", "success")) and self._consecutive_train_failures > 0:
                    self._recovery_from_failures = self._consecutive_train_failures
                    self._consecutive_train_failures = 0
                    self._last_train_failure_reasons.clear()
                    self._error_pattern_history.clear()

    def _record_and_escalate_failure(self, cmd, result, pattern, is_verification):
        """Record a training failure and escalate if pattern repeats."""
        if is_verification and self._consecutive_train_failures >= 2:
            escalation = (
                f"\n\n[VERIFICATION FAILURE AUDIT] {self._consecutive_train_failures} consecutive "
                f"verification script failures detected.\n"
                f"STOP incremental fixes. Perform systematic audit:\n"
                f"1. Read the COMPLETE API contract (init signature, forward kwargs, return types)\n"
                f"2. List ALL shape/dtype/key assumptions in your code\n"
                f"3. Verify EACH assumption against the framework code\n"
                f"4. Fix ALL issues at once, then run\n"
                f"Recent failures:\n"
            )
            for i, r in enumerate(self._last_train_failure_reasons[-3:], 1):
                escalation += f"  {i}. {r}\n"
            self.history.append({"role": "user", "content": escalation})

        elif (len(self._error_pattern_history) >= 2
                and self._error_pattern_history[-1] == self._error_pattern_history[-2]
                and self._error_pattern_history[-1] != "unknown"):
            escalation = (
                f"\n\n[ERROR PATTERN REPEAT] Same error pattern '{pattern}' occurred twice consecutively.\n"
                f"STOP incremental fixes. Required actions:\n"
                f"1. State the ROOT CAUSE in one sentence (not the symptom)\n"
                f"2. List ALL assumptions that led to this approach\n"
                f"3. Propose a FUNDAMENTALLY DIFFERENT approach\n"
                f"4. If you can't identify root cause, ASK the user\n"
                f"Recent failures:\n"
            )
            for i, r in enumerate(self._last_train_failure_reasons[-3:], 1):
                escalation += f"  {i}. {r}\n"
            self.history.append({"role": "user", "content": escalation})

        elif self._consecutive_train_failures >= 3:
            escalation = (
                f"\n\n[ESCALATION] {self._consecutive_train_failures} consecutive training failures detected. "
                f"STOP and report to the user. Summarize all attempts and failures before continuing.\n"
                f"Recent failure reasons:\n"
            )
            for i, r in enumerate(self._last_train_failure_reasons[-5:], 1):
                escalation += f"  {i}. {r}\n"
            self.history.append({"role": "user", "content": escalation})
