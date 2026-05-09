"""Context compaction logic for ReactAgent — summarizer, scorer, and pre-compaction memory dump."""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)


class CompactionMixin:
    """Mixin providing context compaction capabilities."""

    def _summarize_for_compaction(self, text: str) -> str:
        """Call LLM to summarize conversation segment being dropped during compaction.

        Preserves critical state that must survive compaction.
        """
        state_snapshot = []

        if self._original_user_task:
            state_snapshot.append(f"Original user task: {self._original_user_task[:200]}")

        mode_flags = []
        if self._porting_mode:
            mode_flags.append("porting")
        if self._data_prep_mode:
            mode_flags.append("data_prep")
        if self._analysis_persisted:
            mode_flags.append("analysis_persisted")
        if self._porting_path_confirmed:
            mode_flags.append("path_confirmed")
        if self._training_started:
            mode_flags.append("training_started")
        if self._understanding_verified:
            mode_flags.append("understanding_verified")
        if getattr(self, '_data_pipeline_understood', False):
            mode_flags.append("data_pipeline_understood")
        if getattr(self, '_component_plan_created', False):
            mode_flags.append("component_plan_created")
        if mode_flags:
            state_snapshot.append(f"Mode flags: {', '.join(mode_flags)}")

        if self._error_pattern_history:
            recent_errors = self._error_pattern_history[-3:]
            state_snapshot.append(f"Error patterns: {', '.join(recent_errors)}")

        if self._verification_stage != "none":
            state_snapshot.append(f"Verification stage: {self._verification_stage}")

        if self._current_phase != "idle":
            state_snapshot.append(f"Current phase: {self._current_phase}")

        if self._reading_categories:
            state_snapshot.append(f"Reading categories covered: {', '.join(sorted(self._reading_categories))}")

        if self._files_read_this_session:
            recent_files = list(self._files_read_this_session)[-10:]
            state_snapshot.append(f"Recent files read: {', '.join(recent_files)}")

        current_exp = self._experiment_manager.get_current_experiment()
        if current_exp:
            state_snapshot.append(f"Current experiment: {current_exp}")

        if self._consecutive_train_failures > 0:
            state_snapshot.append(f"consecutive_train_failures: {self._consecutive_train_failures}")

        state_block = "\n".join(state_snapshot) if state_snapshot else "(no critical state)"

        messages = [
            {"role": "system", "content": "You are a concise summarizer. Output only the summary, no preamble."},
            {"role": "user", "content": f"{text}\n\n--- CRITICAL STATE (preserve in summary) ---\n{state_block}"},
        ]
        response = self.provider.chat(messages, tools=[])
        summary = response.get("content", "").strip()

        return f"{summary}\n\n[State at compaction: {state_block}]"

    def _score_messages_for_compaction(self, messages):
        """Call LLM to score messages by value for compaction drop priority.

        Returns list of scores 0-10 (higher = more valuable to keep).
        Only called during full compaction when there are enough candidates.
        """
        descriptions = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            parts.append(f"[tool_use: {block.get('name', '?')}({str(block.get('input', ''))[:80]})]")
                        elif block.get("type") == "tool_result":
                            text = str(block.get("content", ""))[:150]
                            parts.append(f"[tool_result: {text}]")
                        elif block.get("type") == "text":
                            parts.append(block.get("text", "")[:150])
                content_str = " ".join(parts)
            else:
                content_str = str(content)[:200]
            descriptions.append(f"{i}: [{role}] {content_str}")

        batch_text = "\n".join(descriptions)

        prompt = (
            "Score each message by its VALUE for an AI agent continuing a task. "
            "Consider RE-READ COST: if this info is lost, will the agent need to re-execute "
            "a command or re-read a file? High re-read cost = high value.\n\n"
            "Score 0-10:\n"
            "- 9-10: Errors, decisions, file content that's expensive to re-read\n"
            "- 6-8: Useful context, reasoning, short conclusions\n"
            "- 3-5: Repetitive monitoring, directory listings\n"
            "- 0-2: Install logs, verbose build output, redundant checks\n\n"
            f"Messages:\n{batch_text}\n\n"
            "Reply with ONLY a comma-separated list of integer scores, one per message. "
            "Example: 7,3,8,2,5"
        )

        scoring_messages = [
            {"role": "system", "content": "You are a concise scorer. Output only the scores."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.provider.chat(scoring_messages, tools=[])
            score_text = response.get("content", "").strip()
            # Strip preamble text (e.g., "Here are the scores: 7,3,...")
            # Find the first digit and start parsing from there
            first_digit = next((i for i, c in enumerate(score_text) if c.isdigit()), -1)
            if first_digit > 0:
                score_text = score_text[first_digit:]
            # Split on comma, semicolon, or newline — LLMs vary in separator choice
            tokens = re.split(r'[,;\n]+', score_text)
            scores = []
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    scores.append(max(0, min(10, int(float(tok)))))
                except (ValueError, TypeError):
                    continue
            if len(scores) == len(messages):
                return scores
            # Tolerate small mismatches (±10%): align by truncating or padding with median
            diff = len(scores) - len(messages)
            if abs(diff) <= max(1, len(messages) // 10):
                if diff > 0:
                    scores = scores[:len(messages)]
                else:
                    median = sorted(scores)[len(scores) // 2] if scores else 5
                    scores.extend([median] * (-diff))
                logger.warning("Scorer returned %d scores for %d messages, aligned by %s",
                               len(scores) + diff, len(messages), "truncation" if diff > 0 else "padding")
                return scores
            logger.warning("Scorer returned %d scores for %d messages, falling back", len(scores), len(messages))
        except Exception as e:
            logger.warning("LLM scorer failed: %s", e)

        return []

    def _pre_compaction_memory_dump(self, droppable_messages=None):
        """Auto-save critical context before compaction destroys it.

        Scans droppable messages (or recent 20 if not provided) for errors,
        solutions, attempted fixes, and modified files. Injects a warning
        message into history if the dump fails so the agent knows context was lost.
        """
        try:
            scan_msgs = droppable_messages if droppable_messages else self.history.messages[-20:]
            errors_found = []
            solutions_found = []
            attempted_fixes = []
            files_modified = []
            current_approach = ""

            for msg in scan_msgs:
                text = ""
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                elif msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                r = block.get("content", "")
                                if isinstance(r, str) and ("Error" in r or "Traceback" in r):
                                    errors_found.append(r[:200])

                if not text:
                    continue

                if "error" in text.lower() or "failed" in text.lower():
                    for line in text.split("\n"):
                        if any(kw in line.lower() for kw in ("error:", "failed:", "traceback", "fix:")):
                            errors_found.append(line.strip()[:150])
                if any(kw in text.lower() for kw in ("fixed by", "solution:", "workaround:", "resolved")):
                    for line in text.split("\n"):
                        if any(kw in line.lower() for kw in ("fixed", "solution", "workaround", "resolved")):
                            solutions_found.append(line.strip()[:150])
                # Track attempted fixes (tried X but Y pattern)
                if any(kw in text.lower() for kw in ("tried", "attempted", "but failed", "didn't work", "still fails")):
                    for line in text.split("\n"):
                        if any(kw in line.lower() for kw in ("tried", "attempted", "but", "didn't work")):
                            attempted_fixes.append(line.strip()[:150])
                if "write_file" in text or "edit_file" in text:
                    paths = re.findall(r'["\']([/\w._-]+\.(py|yaml|sh))["\']', text)
                    files_modified.extend(p[0] for p in paths[:5])
                if "approach" in text.lower() or "strategy" in text.lower() or "plan" in text.lower():
                    if len(text) < 500:
                        current_approach = text[:300]

            parts = []
            if errors_found:
                parts.append(f"Errors: {'; '.join(errors_found[:5])}")
            if solutions_found:
                parts.append(f"Solutions: {'; '.join(solutions_found[:5])}")
            if attempted_fixes:
                parts.append(f"Attempted fixes (FAILED): {'; '.join(attempted_fixes[:5])}")
            if files_modified:
                parts.append(f"Files modified: {', '.join(set(files_modified)[:10])}")
            if current_approach:
                parts.append(f"Approach: {current_approach}")

            parts.append(f"Files read this session: {len(self._files_read_this_session)}")
            parts.append(f"Phase: {self._current_phase}, Verification: {self._verification_stage}")
            parts.append(f"Turn: {self._turn_count}")

            checkpoint_content = "\n".join(parts)

            key = f"compaction_checkpoint_{int(time.time())}"
            self.session_memory.put(
                key, "context", checkpoint_content,
                self._session_id,
                priority="critical",
            )

            # Sync attempted fixes to ExperimentManager so they survive compaction
            if attempted_fixes and hasattr(self, '_experiment_manager'):
                exp_name = self._experiment_manager.get_current_experiment()
                if exp_name:
                    exp = self._experiment_manager.read(exp_name)
                    if exp:
                        existing_learnings = exp.get("learnings", [])
                        for fix in attempted_fixes[:5]:
                            entry = f"[FAILED] {fix}"
                            if entry not in existing_learnings:
                                existing_learnings.append(entry)
                        exp["learnings"] = existing_learnings[-10:]
                        self._experiment_manager._save(exp_name, exp)

            logger.info("Pre-compaction memory dump: saved checkpoint with %d errors, %d solutions, %d attempted fixes",
                        len(errors_found), len(solutions_found), len(attempted_fixes))
        except Exception as e:
            logger.warning("Pre-compaction memory dump failed: %s", e)
            # Inject warning so agent knows context was lost without backup
            try:
                self.history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM WARNING] Pre-compaction memory dump FAILED. "
                        "Critical context from dropped messages may be permanently lost. "
                        "Re-read relevant files and re-check state before proceeding."
                    ),
                })
            except Exception:
                pass

    def _get_compaction_anchors(self) -> list:
        """Extract mandatory anchors for summary preservation."""
        anchors = []
        try:
            if self._original_user_task:
                anchors.append(f"ORIGINAL USER TASK: {self._original_user_task[:200]}")

            plan = self.task_plan.get_active() if hasattr(self, 'task_plan') else None
            if plan:
                doing = [s for s in plan.get("steps", []) if s.get("status") == "doing"]
                if doing:
                    anchors.append(f"Current plan step: {doing[0].get('title', '')[:80]}")

            entries = self.session_memory.list_entries()
            high_pri = [e for e in entries if e.get("priority") == "high"]
            for e in high_pri[-3:]:
                anchors.append(f"Key [{e['key']}]: {e['content'][:80]}")

            if hasattr(self, '_files_read_this_session') and self._files_read_this_session:
                files_list = list(self._files_read_this_session)[:8]
                anchors.append(f"Files already read: {', '.join(os.path.basename(f) for f in files_list)}")

            if self._porting_mode:
                if hasattr(self, '_verification_stage') and self._verification_stage != "none":
                    anchors.append(f"Verification stage: {self._verification_stage}")
                porting_entries = [
                    e for e in entries
                    if any(kw in (e.get("content") or "").lower()
                           for kw in ("mapping", "component", "architecture", "porting", "model structure"))
                ]
                for e in porting_entries[-2:]:
                    anchors.append(f"Porting [{e['key']}]: {e['content'][:100]}")
                exp_name = self._experiment_manager.get_current_experiment()
                if exp_name:
                    exp = self._experiment_manager.read(exp_name)
                    if exp:
                        attempts = exp.get("attempts", [])
                        if attempts:
                            last = attempts[-1]
                            anchors.append(f"Last experiment attempt: {last.get('change', '')[:60]} → {last.get('result', '')[:60]}")
        except Exception:
            pass
        return anchors
