"""Skill lifecycle management and error-to-skill auto-loading mixin."""

import logging
import re
import uuid

from flagscale.agent.react import display

logger = logging.getLogger(__name__)


class SkillLifecycleMixin:
    """Manages skill loading/unloading, phase transitions, and training failure tracking."""

    # ── Phase transitions ──────────────────────────────────────────────
    # Strict ordering: each phase has prerequisites that MUST be met before advancing.
    # This prevents the common failure of doing checkpoint conversion before model
    # structure is complete, or connecting data pipelines without considering parallelism.

    _PHASE_TRANSITIONS = {
        "analysis": {
            "exit_conditions": ["read_count >= 8", "categories >= 3", "analysis_persisted"],
            "next_phase": "structure_implementation",
            "message": (
                "\n\n[PHASE TRANSITION → STRUCTURE IMPLEMENTATION]\n"
                "Analysis phase complete. Moving to model structure implementation.\n"
                "Requirements before proceeding:\n"
                "1. Component enumeration persisted (ALL source modules listed)\n"
                "2. Component diff table completed\n"
                "3. Parallelism strategy decided\n"
                "Next: Implement ALL model components. Do NOT start checkpoint conversion yet."
            ),
        },
        "structure_implementation": {
            "exit_conditions": ["code_written", "structure_completeness_verified"],
            "next_phase": "structure_verification",
            "message": (
                "\n\n[PHASE TRANSITION → STRUCTURE VERIFICATION]\n"
                "Model code written. Before checkpoint conversion, verify structure completeness:\n"
                "1. Compare target module tree against source enumeration\n"
                "2. Verify ALL components from checklist are implemented\n"
                "3. Parameter count matches source (within 1% tolerance)\n"
                "Do NOT proceed to checkpoint conversion until structure is verified complete."
            ),
        },
        "structure_verification": {
            "exit_conditions": ["structure_completeness_verified"],
            "next_phase": "data_pipeline",
            "message": (
                "\n\n[PHASE TRANSITION → DATA PIPELINE]\n"
                "Structure verified complete. Now implement data pipeline.\n"
                "CRITICAL: Data pipeline MUST consider parallelism strategy:\n"
                "- TP: All TP ranks receive IDENTICAL input (use broadcast_data)\n"
                "- PP: Only first stage needs tokens, only last needs labels\n"
                "- DP: Different micro-batch per rank (handled by sampler)\n"
                "Implement get_batch with real data, not dummy data."
            ),
        },
        "data_pipeline": {
            "exit_conditions": ["data_pipeline_implemented"],
            "next_phase": "checkpoint_conversion",
            "message": (
                "\n\n[PHASE TRANSITION → CHECKPOINT CONVERSION]\n"
                "Data pipeline implemented. Now convert checkpoints.\n"
                "1. Convert ALL weights in one pass (not incrementally)\n"
                "2. Verify: key count, tensor shapes, numerical statistics\n"
                "3. Load with strict=True to catch missing/unexpected keys"
            ),
        },
        "checkpoint_conversion": {
            "exit_conditions": ["checkpoint_converted"],
            "next_phase": "training_verification",
            "message": (
                "\n\n[PHASE TRANSITION → TRAINING VERIFICATION]\n"
                "Checkpoint converted and verified. Run short training verification:\n"
                "1. Run --train-iters 20 with real data\n"
                "2. Verify loss is finite and decreasing\n"
                "3. Check all components receive gradients\n"
                "4. Verify parallelism works (TP/PP if configured)"
            ),
        },
        "training_verification": {
            "exit_conditions": ["training_verified"],
            "next_phase": "complete",
            "message": (
                "\n\n[PHASE TRANSITION → COMPLETE]\n"
                "Training verification passed. Porting is complete.\n"
                "Deliverables: model code, checkpoint conversion, get_batch, training config."
            ),
        },
    }

    # Phase ordering for enforcement — later phases cannot be entered without
    # completing earlier ones
    _PHASE_ORDER = [
        "analysis",
        "structure_implementation",
        "structure_verification",
        "data_pipeline",
        "checkpoint_conversion",
        "training_verification",
        "complete",
    ]

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
        """Check if the agent should transition to the next phase.

        Enforces strict phase ordering — each phase has prerequisites that must
        be met before advancing. This prevents premature checkpoint conversion
        or data pipeline work before model structure is complete.
        """
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
            if files_read >= 8 and self._analysis_persisted:
                self._current_phase = transition["next_phase"]
                return transition["message"]

        elif current_phase == "structure_implementation":
            if self._code_written and getattr(self, '_structure_completeness_verified', False):
                self._current_phase = transition["next_phase"]
                return transition["message"]

        elif current_phase == "structure_verification":
            if getattr(self, '_structure_completeness_verified', False):
                self._current_phase = transition["next_phase"]
                return transition["message"]

        elif current_phase == "data_pipeline":
            if getattr(self, '_data_pipeline_understood', False) and self._code_written:
                self._current_phase = transition["next_phase"]
                return transition["message"]

        elif current_phase == "checkpoint_conversion":
            # Detect checkpoint conversion completion from recent tool results
            recent_tools = list(self._recent_tool_calls)[-10:]
            has_conversion = any(
                "convert" in str(rest).lower() and "success" in str(rest).lower()
                for _, *rest in recent_tools
            )
            if has_conversion:
                self._current_phase = transition["next_phase"]
                return transition["message"]

        elif current_phase == "training_verification":
            if self._training_started and self._consecutive_train_failures == 0:
                self._current_phase = transition["next_phase"]
                return transition["message"]

        return ""

    def _get_current_phase_guidance(self):
        """Return guidance about what the current phase allows and prohibits."""
        current_phase = getattr(self, "_current_phase", "analysis")
        phase_idx = self._PHASE_ORDER.index(current_phase) if current_phase in self._PHASE_ORDER else 0

        prohibitions = []
        if phase_idx < self._PHASE_ORDER.index("checkpoint_conversion"):
            prohibitions.append("Checkpoint conversion is NOT allowed yet — complete model structure first")
        if phase_idx < self._PHASE_ORDER.index("data_pipeline"):
            prohibitions.append("Data pipeline implementation is NOT allowed yet")
        if phase_idx < self._PHASE_ORDER.index("training_verification"):
            prohibitions.append("Full training launch is NOT allowed yet")

        if not prohibitions:
            return ""

        return (
            f"\n[CURRENT PHASE: {current_phase}] "
            f"Prohibited actions: {'; '.join(prohibitions)}\n"
        )

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

    def _extract_fix_fingerprint(self):
        """Extract a human-readable fingerprint of the most recent fix actions.

        Scans recent tool calls to find edit_file/write_file/shell commands that
        constitute the fix attempt. Returns a short string like:
          "edited megatron/model/qwen3.py (changed import path)"
        """
        recent = list(self._recent_tool_calls)[-8:]
        fix_parts = []
        for entry in reversed(recent):
            if not isinstance(entry, tuple) or len(entry) < 2:
                continue
            tool_name = entry[0]
            if tool_name == "edit_file" and len(entry) >= 4:
                path = entry[1]
                old_str = entry[2][:60] if len(entry) > 2 else ""
                new_str = entry[3][:60] if len(entry) > 3 else ""
                fname = path.rsplit("/", 1)[-1] if "/" in path else path
                fix_parts.append(f"edited {fname} ({old_str!r} → {new_str!r})")
            elif tool_name == "write_file" and len(entry) >= 2:
                path = entry[1]
                fname = path.rsplit("/", 1)[-1] if "/" in path else path
                fix_parts.append(f"wrote {fname}")
            elif tool_name == "shell" and len(entry) >= 2:
                cmd = entry[1]
                if any(kw in cmd for kw in ("pip install", "sed ", "mv ", "cp ", "ln ")):
                    fix_parts.append(f"shell: {cmd[:80]}")
            if len(fix_parts) >= 3:
                break
        return "; ".join(fix_parts) if fix_parts else "(no edits detected before failure)"

    def _llm_judge_is_same_fix(self, prev_fingerprint, current_fingerprint):
        """Use LLM to judge whether two fix fingerprints are semantically the same approach."""
        if not prev_fingerprint or not current_fingerprint:
            return False
        if prev_fingerprint == current_fingerprint:
            return True
        try:
            prompt = (
                "Are these two fix attempts essentially the same approach (just with minor variations)?\n\n"
                f"Previous: {prev_fingerprint}\n"
                f"Current: {current_fingerprint}\n\n"
                "Answer ONLY 'yes' or 'no'."
            )
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            answer = response.get("content", "").strip().lower()
            return answer.startswith("yes")
        except Exception:
            return False

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
                    fix_fp = self._extract_fix_fingerprint()
                    self._error_pattern_history.append((pattern, fix_fp))
                    self._source_reads_since_last_failure = 0
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
                    fix_fp = self._extract_fix_fingerprint()
                    self._error_pattern_history.append((pattern, fix_fp))
                    self._source_reads_since_last_failure = 0
                    cmd = tc["arguments"].get("command", "") or tc["arguments"].get("output_dir", "monitor")
                    self._record_and_escalate_failure(cmd, result, pattern, False)
                elif any(kw in result[:500] for kw in ("target_reached", "success")) and self._consecutive_train_failures > 0:
                    self._recovery_from_failures = self._consecutive_train_failures
                    self._consecutive_train_failures = 0
                    self._last_train_failure_reasons.clear()
                    self._error_pattern_history.clear()

    def _record_and_escalate_failure(self, cmd, result, pattern, is_verification):
        """Record a training failure and escalate if pattern repeats."""
        # Build failed-attempts summary from structured history
        failed_attempts_summary = ""
        if len(self._error_pattern_history) >= 2:
            lines = []
            for i, entry in enumerate(self._error_pattern_history[-5:], 1):
                p, fp = entry if isinstance(entry, tuple) else (entry, "")
                lines.append(f"  #{i}. [{p}] {fp}")
            failed_attempts_summary = "\nPrevious fix attempts (ALL FAILED — do NOT repeat):\n" + "\n".join(lines) + "\n"

        if is_verification and self._consecutive_train_failures >= 2:
            escalation = (
                f"\n\n[VERIFICATION FAILURE AUDIT] {self._consecutive_train_failures} consecutive "
                f"verification script failures detected.\n"
                f"STOP incremental fixes. You are likely missing something in the framework implementation.\n"
                f"Mandatory diagnosis steps:\n"
                f"1. Read the ACTUAL upstream source code — Megatron-LM-FL / TransformerEngine-FL / FlagScale "
                f"implementation of the function that's failing. Not your code — the FRAMEWORK code.\n"
                f"2. List ALL shape/dtype/key assumptions in your code and verify EACH against framework source\n"
                f"3. Check API contract: init signature, forward kwargs, return types in the actual installed code\n"
                f"4. Fix ALL issues at once based on what you read, then run\n"
                f"{failed_attempts_summary}"
                f"Recent failures:\n"
            )
            for i, r in enumerate(self._last_train_failure_reasons[-3:], 1):
                escalation += f"  {i}. {r}\n"
            self.history.append({"role": "user", "content": escalation})

        elif len(self._error_pattern_history) >= 2:
            prev_entry = self._error_pattern_history[-2]
            curr_entry = self._error_pattern_history[-1]
            prev_pattern = prev_entry[0] if isinstance(prev_entry, tuple) else prev_entry
            prev_fp = prev_entry[1] if isinstance(prev_entry, tuple) else ""
            curr_fp = curr_entry[1] if isinstance(curr_entry, tuple) else ""

            same_pattern = (prev_pattern == pattern and pattern != "unknown")
            same_fix = self._llm_judge_is_same_fix(prev_fp, curr_fp) if (prev_fp and curr_fp) else False

            if same_pattern or same_fix:
                reason_tag = "same error pattern" if same_pattern else "same fix approach on different error"
                escalation = (
                    f"\n\n[REPEATED FIX DETECTED] {reason_tag} — '{pattern}' occurred again.\n"
                    f"Your previous fix: {prev_fp}\n"
                    f"Your current fix: {curr_fp}\n"
                    f"{'These are the SAME approach — it already FAILED.' if same_fix else ''}\n"
                    f"STOP. The problem is almost certainly incomplete understanding of the framework.\n"
                    f"Required actions:\n"
                    f"1. READ the upstream implementation that's involved in this error "
                    f"(Megatron-LM-FL / TransformerEngine-FL / FlagScale source — not your code)\n"
                    f"2. State the ROOT CAUSE in one sentence (not the symptom)\n"
                    f"3. Explain WHY the previous fix did not work — what did you misunderstand?\n"
                    f"4. Propose a FUNDAMENTALLY DIFFERENT approach based on what you read\n"
                    f"5. If stuck after 2+ fundamentally different attempts, ASK the user\n"
                    f"{failed_attempts_summary}"
                )
                self.history.append({"role": "user", "content": escalation})

        elif self._consecutive_train_failures >= 3:
            escalation = (
                f"\n\n[ESCALATION] {self._consecutive_train_failures} consecutive training failures detected. "
                f"STOP and report to the user. Summarize all attempts and failures before continuing.\n"
                f"{failed_attempts_summary}"
                f"Recent failure reasons:\n"
            )
            for i, r in enumerate(self._last_train_failure_reasons[-5:], 1):
                escalation += f"  {i}. {r}\n"
            self.history.append({"role": "user", "content": escalation})
