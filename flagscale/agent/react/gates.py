"""Enforcement gates for ReactAgent — progress, plan, dry-run, training, and phase gates."""

import logging
import re
import time

logger = logging.getLogger(__name__)


class GatesMixin:
    """Mixin providing enforcement gate logic."""

    # Progress gate is now staleness-based (see _check_progress_gate)

    _PRODUCTIVE_TOOLS = frozenset({
        "memory_write", "write_file", "edit_file",
        "workspace_experiment",
        "plan_update", "plan_create",
    })

    _READ_ONLY_TOOLS = frozenset({
        "read_file", "shell", "web_fetch", "find_latest_log",
        "parse_training_metrics", "memory_read", "memory_list",
    })

    def _check_progress_gate(self, tool_name):
        """Detect aimless exploration vs purposeful deep reading.

        Instead of counting consecutive reads, detect actual stuck patterns:
        - Re-reading the same files without new information
        - Repeated similar shell commands with same errors
        - Long stretches with no new unique files discovered

        Only intervenes when the agent is genuinely stuck, not when doing
        purposeful deep source code reading across different files.

        Returns: (warning_text, is_hard_block)
        """
        if tool_name in self._PRODUCTIVE_TOOLS:
            self._consecutive_reads = 0
            self._progress_gate_triggers = 0
            self._reads_since_last_new_file = 0
            self._last_gate_warning = ""
            return "", False
        if tool_name in self._READ_ONLY_TOOLS:
            self._consecutive_reads += 1

        # Track whether we're discovering new files (purposeful exploration)
        # vs re-reading known files (potentially stuck)
        if not hasattr(self, '_reads_since_last_new_file'):
            self._reads_since_last_new_file = 0
        if not hasattr(self, '_last_unique_file_count'):
            self._last_unique_file_count = len(self._files_read_this_session)

        current_unique = len(self._files_read_this_session)
        if current_unique > self._last_unique_file_count:
            # New file discovered — reset staleness counter
            self._reads_since_last_new_file = 0
            self._last_unique_file_count = current_unique
        elif tool_name == "shell":
            # Shell commands are exploratory — don't count toward staleness
            pass
        else:
            self._reads_since_last_new_file += 1

        # Detect repeated shell errors (same error appearing multiple times)
        repeated_errors = self._count_repeated_recent_errors()

        # === Intervention logic ===

        # Pattern 1: Re-reading without discovering anything new for a long time
        stale_threshold = 25
        if self._porting_mode:
            stale_threshold = 40  # Porting requires reading many source files
        elif self._consecutive_train_failures >= 2:
            stale_threshold = 30  # More lenient during debugging

        if self._reads_since_last_new_file >= stale_threshold:
            self._progress_gate_triggers += 1
            if self._reads_since_last_new_file >= stale_threshold + 8:
                # Hard block only after extended staleness
                has_plan = self.task_plan.get_active() is not None
                if not has_plan:
                    return (
                        f"⛔ [PROGRESS BLOCK — TOOL NOT EXECUTED] You've made "
                        f"{self._reads_since_last_new_file} calls without discovering "
                        f"any new files or producing output. This suggests you're stuck.\n"
                        "Create a plan (plan_create) to organize what you know and "
                        "identify what's missing, then continue with focused goals."
                    ), True
                else:
                    self._reads_since_last_new_file = 0
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

        # Pattern 2: Repeated shell errors suggest a different approach is needed
        if repeated_errors >= 3:
            return (
                "\n\n[PROGRESS NOTE] Similar errors appearing repeatedly. "
                "Consider stepping back to understand the root cause rather than "
                "retrying variations of the same approach."
            ), False

        # Pattern 3: Very long exploration without any checkpoint (safety net)
        # Only triggers if reading many unique files but never recording anything
        reads_hard_cap = 80 if self._porting_mode else 60
        if self._consecutive_reads >= reads_hard_cap and self._progress_gate_triggers == 0:
            self._progress_gate_triggers += 1
            return (
                "\n\n[CHECKPOINT SUGGESTION] You've done extensive exploration "
                f"({len(self._files_read_this_session)} unique files read). "
                "Consider a memory_write to persist key findings — this protects "
                "against context compaction loss. Not mandatory, just good practice."
            ), False

        return "", False

    def _count_repeated_recent_errors(self):
        """Count how many recent shell calls produced similar errors."""
        if not hasattr(self, '_recent_shell_errors'):
            self._recent_shell_errors = []
        # This is populated in _record_iteration; here we just count
        if len(self._recent_shell_errors) < 2:
            return 0
        # Check last 5 errors for similarity
        recent = self._recent_shell_errors[-5:]
        if len(recent) < 2:
            return 0
        # Simple heuristic: if error messages share >50% of words, they're "similar"
        from collections import Counter
        last_words = set(recent[-1].lower().split()[:20])
        similar = sum(1 for e in recent[:-1]
                      if len(set(e.lower().split()[:20]) & last_words) > len(last_words) * 0.5)
        return similar + 1  # Include the last one itself

    _PLAN_GATE_MAX_EXPLORATORY = 120
    _PLAN_GATE_INDEPENDENT_WARN = 80
    _PLAN_GATE_INDEPENDENT_BLOCK = 120

    def _check_plan_creation_gate(self, tool_name):
        """Gate: encourage plan creation for sustained exploration.

        Two activation modes:
        1. Complexity judge fired → _complex_task_no_plan = True, hard block at _PLAN_GATE_MAX_EXPLORATORY
        2. Independent: warn at _PLAN_GATE_INDEPENDENT_WARN, hard block at _PLAN_GATE_INDEPENDENT_BLOCK

        Returns block/warning message or empty string.
        Hard block (non-empty + "TOOL NOT EXECUTED") means tool must NOT execute.
        """
        # Plan already exists — no gate needed
        if self.task_plan.get_active() is not None:
            self._complex_task_no_plan = False
            return ""

        # Productive tools are always allowed
        if tool_name in ("plan_create", "memory_write", "workspace_experiment"):
            return ""

        # Mode 1: complexity judge fired — hard block at 6
        if self._complex_task_no_plan:
            self._pre_plan_tool_calls += 1
            if self._pre_plan_tool_calls > self._PLAN_GATE_MAX_EXPLORATORY:
                task_reminder = ""
                if self._original_user_task:
                    task_reminder = (
                        f"\n\nORIGINAL USER REQUEST: {self._original_user_task[:200]}\n"
                        f"Your plan MUST address THIS request."
                    )
                return (
                    f"⛔ [PLAN GATE — TOOL NOT EXECUTED] This task was flagged as complex. "
                    f"You've used {self._pre_plan_tool_calls} exploratory calls "
                    f"(limit: {self._PLAN_GATE_MAX_EXPLORATORY}) without creating a plan.\n"
                    f"This tool call was BLOCKED. You MUST call plan_create NOW.\n"
                    f"Use what you've gathered so far to create a concrete step-by-step plan."
                    + task_reminder
                )

        # Mode 2: independent — soft warn at 8, hard block at 12
        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_BLOCK:
            task_reminder = ""
            if self._original_user_task:
                task_reminder = (
                    f"\n\nORIGINAL USER REQUEST: {self._original_user_task[:200]}\n"
                    f"Your plan MUST address THIS request."
                )
            return (
                f"⛔ [PLAN GATE — TOOL NOT EXECUTED] You've made {self._consecutive_reads} "
                f"consecutive exploratory calls without creating a plan.\n"
                f"This tool call was BLOCKED. You MUST call plan_create NOW to organize your approach."
                + task_reminder
            )
        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_WARN:
            return (
                f"\n\n[PLAN REMINDER] You've made {self._consecutive_reads} exploratory calls "
                f"without a plan. Consider calling plan_create soon to organize your findings. "
                f"You will be BLOCKED at {self._PLAN_GATE_INDEPENDENT_BLOCK} calls."
            )

        return ""

    def _check_plan_maintenance_gate(self, tool_name):
        """Gate: remind agent to update plan when it's going stale.

        If a plan exists but the current 'doing' step hasn't been updated for 8+ turns,
        inject a reminder to update the plan.
        Returns a soft warning (never hard blocks).
        """
        plan = self.task_plan.get_active()
        if not plan:
            return ""

        # Don't nag when agent is already updating the plan
        if tool_name in ("plan_update", "plan_create", "plan_status"):
            return ""

        doing_steps = [s for s in plan.get("steps", []) if s.get("status") == "doing"]
        if not doing_steps:
            return ""

        step = doing_steps[0]
        last_activity = step.get("_last_activity_turn", 0)
        turns_stale = self._turn_count - last_activity if last_activity else 0

        if turns_stale >= 8:
            return (
                f"\n\n[PLAN MAINTENANCE] Step {step['id']} ('{step.get('title', '')[:40]}') "
                f"has had no plan_update for {turns_stale} turns. "
                f"If it's done, call plan_update(action='step_done'). "
                f"If blocked, call plan_update(action='step_skip') and move on."
            )
        return ""

    def _check_config_validation_hint(self, tool_name):
        """Soft hint: suggest validate_config after writing a YAML config file.

        Includes full schema reference so the agent sees correct structure
        only when it's actually editing configs (not every turn).
        """
        if tool_name == "validate_config":
            return ""
        if not hasattr(self, '_recent_tool_calls') or not self._recent_tool_calls:
            return ""
        last_call = self._recent_tool_calls[-1] if self._recent_tool_calls else None
        if not last_call:
            return ""
        last_name = last_call[0] if isinstance(last_call, (list, tuple)) else ""
        if last_name not in ("write_file", "edit_file"):
            return ""
        last_args = last_call[1] if len(last_call) > 1 else ""
        path = str(last_args) if last_args else ""
        if not path.endswith(".yaml") and not path.endswith(".yml"):
            return ""
        if "conf/" not in path and "conf\\" not in path:
            return ""
        return (
            "\n[CONFIG HINT] You just wrote/edited a YAML config. "
            "Call validate_config(path='...') to check for structural errors.\n"
            "\n"
            "FlagScale config structure reference:\n"
            "  Top-level (conf/train.yaml): defaults, experiment{exp_name,task{type,backend,entrypoint},runner,cmds,envs}, action, hydra\n"
            "  Model-level (conf/train/<model>.yaml):\n"
            "    system: {tensor_model_parallel_size, pipeline_model_parallel_size, precision:{bf16}, logging:{log_interval,wandb_project}, checkpoint:{save_interval}}\n"
            "    model: {num_layers, hidden_size, num_attention_heads, seq_length, micro_batch_size, global_batch_size, optimizer:{lr_scheduler:{lr,min_lr}}}\n"
            "    data: {data_path, tokenizer:{tokenizer_type, tokenizer_path, vocab_size}}\n"
            "  Common mistakes: bf16→system.precision NOT model | tp/pp→system NOT model | save_interval→system.checkpoint NOT model | data_path→data NOT top-level"
        )

    def _check_source_reading_gate(self, tool_name, arguments):
        """Soft gate: after 2+ failures, require reading framework source before writing fixes."""
        if self._consecutive_train_failures < 2:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        if self._source_reads_since_last_failure >= 2:
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not target or any(ext in target for ext in (".yaml", ".yml", ".md", ".txt", ".json")):
            return ""
        return (
            f"\n\n[SOURCE READING REQUIRED] You have {self._consecutive_train_failures} consecutive "
            f"failures but have only read {self._source_reads_since_last_failure} framework source "
            f"files since the last failure.\n"
            "Before writing another fix, read the UPSTREAM implementation:\n"
            "- Find the actual Megatron-LM-FL / TransformerEngine-FL / FlagScale code path involved\n"
            "- Understand what the framework expects (args, shapes, dtypes, return values)\n"
            "- Then write a fix based on what you learned, not on guessing"
        )

    def _check_dry_run_gate(self, cmd, result):
        """Post-execution annotation: distinguish FlagScale dryrun from validation runs."""
        if not self._TRAIN_LAUNCH_RE.search(cmd):
            return result
        if not self._regex_judge_confirm("is_training_launch", cmd):
            return result

        # FlagScale dryrun — script generation only, NO training executed
        if self._is_flagscale_dryrun(cmd):
            return result + (
                "\n\n[FLAGSCALE DRYRUN COMPLETE] This generated launch scripts only — "
                "no training was executed, no GPU was used.\n"
                "Next steps:\n"
                "1. Inspect the generated scripts: cat {exp_dir}/logs/scripts/host_*_run.sh\n"
                "2. Verify: correct GPU count, correct entrypoint, expected CLI flags\n"
                "3. Run a short validation training (--train-iters 20) to verify the pipeline works\n"
                "NOTE: Dryrun passing does NOT validate model loading, data pipeline, or GPU compatibility."
            )

        # Short validation run (--train-iters 20 etc.) — real training happened
        if self._is_quick_test_command(cmd):
            self._dry_run_passed = True
            uses_synthetic = bool(re.search(r'synthetic|/dev/null|mock.data|fake', cmd, re.I))
            synthetic_note = ""
            if uses_synthetic:
                synthetic_note = (
                    "\n\n[SYNTHETIC DATA NOTE] This validation used synthetic/mock data. "
                    "Before full training with real data, you MUST verify:\n"
                    "1. Real data loads without error (1 batch)\n"
                    "2. Batch shapes/dtypes match what model expects\n"
                    "3. Tokenization/preprocessing produces expected output\n"
                    "Do NOT skip this — synthetic validation passing does NOT guarantee real data works."
                )
            return result + (
                "\n\n[VALIDATION RUN COMPLETE] Verify: model loaded? data flowing? "
                "no crashes? If OK, proceed to full run."
            ) + synthetic_note

        # Full training run without prior validation
        if not self._dry_run_passed:
            return result + (
                "\n\n[WARNING: NO VALIDATION RUN] This is a full training run without "
                "prior short validation (--train-iters 20). Issues like unloaded checkpoints, "
                "broken data pipelines, or config errors will waste GPU hours. "
                "Consider stopping and running with --train-iters=20 first."
            )
        return result

    def _check_kill_retry_loop(self, cmd):
        """Detect kill+relaunch cycles. 3 kills in 20 minutes = forced audit."""
        now = time.time()
        is_kill = bool(re.search(r'pkill|kill\s+-9|killall', cmd))
        is_launch = bool(self._TRAIN_LAUNCH_RE.search(cmd))
        if is_launch:
            is_launch = self._regex_judge_confirm("is_training_launch", cmd)
        if is_kill:
            self._kill_retry_timestamps.append(now)
        if is_launch:
            self._training_launch_timestamps.append(now)

        # Keep only last 20 minutes
        cutoff = now - 1200
        self._kill_retry_timestamps = [t for t in self._kill_retry_timestamps if t > cutoff]
        self._training_launch_timestamps = [t for t in self._training_launch_timestamps if t > cutoff]

        if len(self._kill_retry_timestamps) >= 3 and len(self._training_launch_timestamps) >= 3:
            self._kill_retry_timestamps.clear()
            self._training_launch_timestamps.clear()
            return (
                "\n\n[KILL-RETRY LOOP DETECTED] You've killed and relaunched training 3+ times "
                "in 20 minutes. This pattern wastes tokens and GPU time.\n"
                "MANDATORY before next launch:\n"
                "1. memory_write: record what's failing and why each attempt failed\n"
                "2. Identify the SYSTEMIC issue (not just the symptom)\n"
                "3. Consider: is the approach fundamentally wrong? Should you try:\n"
                "   - Different parallelism (TP=1 instead of TP=2)?\n"
                "   - Direct torchrun instead of wrapper?\n"
                "   - Simpler model config first?\n"
                "4. Only relaunch after addressing the root cause."
            )
        return ""

    def _check_training_hang(self, cmd, result, elapsed):
        """Detect training hang: launched but no output progress after timeout."""
        if not self._TRAIN_LAUNCH_RE.search(cmd):
            return ""
        if not self._regex_judge_confirm("is_training_launch", cmd):
            return ""
        if elapsed < 120:
            return ""
        # Require at least 2 independent signals to avoid false positives
        hang_score = 0
        if elapsed > 180 and "iteration" not in result[-500:]:
            hang_score += 1
        if re.search(r'utilization.*\b0\s*%|\b0\s*%.*utilization', result, re.I):
            hang_score += 1
        if "before the start of training step" in result and not re.search(r'iteration\s+\d+', result):
            hang_score += 1
        if hang_score >= 2:
            return (
                "\n\n[TRAINING HANG DETECTED] Training appears hung:\n"
                f"- Elapsed: {elapsed:.0f}s with no iteration progress\n"
                "Likely causes:\n"
                "1. NCCL deadlock (TP/PP communication issue) — try TP=1 first\n"
                "2. Zombie CUDA contexts from previous runs — check nvidia-smi\n"
                "3. rerun_state_machine all_reduce hang — known issue with TP>1\n"
                "ACTION: Kill the process, then diagnose before relaunching.\n"
                "Use: nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
            )
        return ""

    def _check_error_escalation(self, tool_name, arguments):
        """After error, require layered diagnosis before big changes.

        Enforces: environment check → dependency check → source reading → fix.
        The most common failure mode is skipping source reading and jumping to code changes.
        """
        if not self._last_tool_had_error:
            return ""

        big_change_tools = {"write_file", "edit_file"}
        if tool_name not in big_change_tools:
            return ""

        target = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")
        is_config_change = any(ext in target for ext in (".yaml", ".yml", ".json", ".cfg", ".conf"))
        is_large_change = len(content) > 500

        if (is_config_change or is_large_change) and not self._root_cause_recorded_since_error:
            failures = getattr(self, '_consecutive_train_failures', 0)
            if failures >= 2:
                return (
                    "\n\n[DIAGNOSIS REQUIRED — LAYERED RECOVERY]\n"
                    f"You have {failures} consecutive failures and are attempting another code change "
                    "without recording root cause. Follow this diagnosis order:\n"
                    "1. ENVIRONMENT: verify Python env, CUDA version, installed packages (pip show, nvidia-smi)\n"
                    "2. DEPENDENCIES: check FlagScale/Megatron-LM-FL/TransformerEngine-FL versions match\n"
                    "3. SOURCE READING: read the ACTUAL framework implementation that's failing — "
                    "not just your code, but the upstream code path (Megatron/TE/FlagScale source)\n"
                    "4. Only THEN write a fix based on what you learned from the source\n\n"
                    "Record your diagnosis with memory_write or workspace_experiment before proceeding."
                )
            return (
                "\n\n[ROOT CAUSE CHECK] You're making a significant change after an error, "
                "but haven't recorded the root cause yet. Before proceeding:\n"
                "1. What exactly caused the error?\n"
                "2. Why does this change fix the root cause (not just the symptom)?\n"
                "3. Record your diagnosis with memory_write or workspace_experiment."
            )
        return ""

    def _check_context_pressure(self):
        """Force memory dump when context is getting full. Three-tier: 60%/75%/85%."""
        ratio = self.history.get_context_pressure()

        if ratio > 0.85:
            self._pre_compaction_memory_dump()
            # Set anchors so the summary preserves critical info
            anchors = self._get_compaction_anchors()
            if anchors:
                self.history.set_compaction_anchors(anchors)
            self.history.force_compact(target_ratio=0.50)
            self._context_pressure_soft_warned = False
            self._context_pressure_hard_warned = False
            return ""

        if ratio > 0.75 and not self._context_pressure_hard_warned:
            self._context_pressure_hard_warned = True
            return (
                "\n\n[CONTEXT PRESSURE — CRITICAL] You've used ~75% of context budget.\n"
                "You MUST write ALL key findings to memory NOW:\n"
                "1. memory_write: errors encountered and solutions found\n"
                "2. memory_write: current approach and what phase you're in\n"
                "3. workspace_experiment: add_attempt with current progress\n"
                "After that, history will be auto-compacted. Anything not saved WILL BE LOST."
            )

        if ratio > 0.60 and not self._context_pressure_soft_warned:
            self._context_pressure_soft_warned = True
            return (
                "\n\n[CONTEXT BUDGET — 60%] Start persisting findings now.\n"
                "Use memory_write for: discoveries, workarounds, decisions.\n"
                "Use workspace_experiment for: attempt results.\n"
                "Avoid re-reading files you've already analyzed."
            )

        return ""



    _PORTING_WRITE_PATHS = re.compile(
        r'(tools/checkpoint/|flagscale/models/|megatron/.*model|model_provider|spec\.py|'
        r'pretrain_|train_.*\.py|data/.*dataset)'
    )
    _MIN_READS_BEFORE_PORTING_WRITE = 8

    _PORTING_PATH_EARLY_READ_LIMIT = 8

    def _check_porting_path_gate(self, tool_name, arguments):
        """Block until user confirms porting path (Mode B/C).

        Triggers EARLY — not just on write, but also on:
        - plan_create (making a plan = committing to a direction)
        - read_file beyond threshold (prevent deep exploration without direction)
        This avoids wasting tokens on the wrong path before user confirmation.
        """
        if not self._porting_mode or self._porting_path_confirmed:
            return ""

        triggered = False

        if tool_name in ("write_file", "edit_file"):
            target = arguments.get("path", "") or arguments.get("file_path", "")
            if self._PORTING_WRITE_PATHS.search(target):
                triggered = True

        elif tool_name == "plan_create":
            triggered = True

        elif tool_name == "read_file":
            if len(self._files_read_this_session) >= self._PORTING_PATH_EARLY_READ_LIMIT:
                triggered = True

        if not triggered:
            return ""

        return (
            "\n\n[PORTING PATH GATE] STOP. You must confirm the porting path with the user "
            "BEFORE going deeper.\n\n"
            "You've done initial exploration — now present the trade-offs and get a decision:\n"
            "- Mode B (Megatron Native): Full parallelism (TP/PP/EP/CP), best performance, "
            "higher implementation effort. Rewrite model using Megatron layer_spec system.\n"
            "- Mode C (HuggingFace Wrapper): FSDP2 distribution, HF model as-is, "
            "fastest to implement, limited parallelism. Wrap existing model code.\n\n"
            "Based on what you've read about the model, explain the complexity and "
            "recommend a path WITH REASONING. Then WAIT for user's explicit choice.\n"
            "Do NOT create a plan, do NOT read more files, do NOT write code until confirmed.\n"
            "\nIMPORTANT: End your response with [NEED_USER_INPUT] to ensure the user sees your question.\n"
        )

    # Mode-specific signal patterns for deviation detection
    _MODE_B_SIGNALS = re.compile(
        r'megatron.*native|layer_spec|ColumnParallelLinear|RowParallelLinear|'
        r'tensor.model.parallel|pipeline.model.parallel|'
        r'get_gpt_layer_with_transformer_engine|MegatronModule|'
        r'mode\s*b\b',
        re.IGNORECASE,
    )
    _MODE_C_SIGNALS = re.compile(
        r'HuggingFace\s*(?:Module|Wrapper)|FSDP2?|hf_module|'
        r'wrap.*existing.*model|from_pretrained.*wrapper|'
        r'mode\s*c\b',
        re.IGNORECASE,
    )

    def _check_porting_path_deviation_gate(self, tool_name, arguments):
        """Hard-block if agent tries to implement the WRONG porting path after user confirmed one."""
        if not self._confirmed_porting_path or self._confirmed_porting_path == "unknown":
            return ""

        if tool_name not in ("memory_write", "plan_create", "plan_update", "write_file", "edit_file"):
            return ""

        content = ""
        if tool_name == "memory_write":
            content = arguments.get("content", "") + " " + arguments.get("key", "")
        elif tool_name in ("plan_create", "plan_update"):
            content = arguments.get("content", "") + " " + arguments.get("plan", "")
        elif tool_name in ("write_file", "edit_file"):
            content = arguments.get("content", "") or arguments.get("new_content", "")
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = content + " " + path

        if not content.strip():
            return ""

        if self._confirmed_porting_path == "mode_b":
            if self._MODE_C_SIGNALS.search(content) and not self._MODE_B_SIGNALS.search(content):
                return (
                    "\n\n[PORTING PATH DEVIATION] BLOCKED. The user confirmed Mode B "
                    "(Megatron Native) but you are writing Mode C (HuggingFace Wrapper) content.\n\n"
                    "You MUST NOT switch architectural approach without explicit user permission.\n"
                    "If you believe Mode B is infeasible, explain WHY to the user and ask for "
                    "permission to change. Do NOT proceed until the user explicitly agrees.\n"
                    "\n[NEED_USER_INPUT]\n"
                )
        elif self._confirmed_porting_path == "mode_c":
            if self._MODE_B_SIGNALS.search(content) and not self._MODE_C_SIGNALS.search(content):
                return (
                    "\n\n[PORTING PATH DEVIATION] BLOCKED. The user confirmed Mode C "
                    "(HuggingFace Wrapper) but you are writing Mode B (Megatron Native) content.\n\n"
                    "You MUST NOT switch architectural approach without explicit user permission.\n"
                    "If you believe Mode C is infeasible, explain WHY to the user and ask for "
                    "permission to change. Do NOT proceed until the user explicitly agrees.\n"
                    "\n[NEED_USER_INPUT]\n"
                )

        return ""

    def _check_reading_depth_gate(self, tool_name, arguments):
        """Warn when writing porting code with insufficient reading."""
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""

        read_count = len(self._files_read_this_session)
        if read_count >= self._MIN_READS_BEFORE_PORTING_WRITE:
            return ""

        return (
            f"\n\n[READING DEPTH WARNING] You're writing porting code but have only read "
            f"{read_count} files this session (minimum recommended: {self._MIN_READS_BEFORE_PORTING_WRITE}). "
            f"Model porting failures are almost always caused by incomplete understanding. "
            f"Consider reading source model code, target base classes, and similar implementations first."
        )

    def _check_diagnostic_print_hint(self, tool_name, arguments):
        """Suggest diagnostic prints when writing model/training code for the first time.

        Fires once per file: when writing a .py file that contains forward/init/get_batch
        and doesn't already include shape-printing statements.
        """
        if tool_name != "write_file":
            return ""
        if not self._porting_mode:
            return ""
        path = arguments.get("path", "")
        if not path.endswith(".py"):
            return ""
        content = arguments.get("content", "")
        if not content:
            return ""
        # Only trigger for model/training files
        has_model_code = any(kw in content for kw in (
            "def forward(", "def __init__(", "def get_batch(", "def model_provider(",
        ))
        if not has_model_code:
            return ""
        # Skip if already has diagnostic prints
        has_prints = any(kw in content for kw in (
            "print(", ".shape", "f\"shape", "f\"dtype", "print_rank_0(",
        ))
        if has_prints:
            return ""
        # Only fire once per file
        if not hasattr(self, '_diagnostic_print_hinted_files'):
            self._diagnostic_print_hinted_files = set()
        if path in self._diagnostic_print_hinted_files:
            return ""
        self._diagnostic_print_hinted_files.add(path)
        return (
            "\n[DIAGNOSTIC PRINT HINT] You're writing model/training code without diagnostic prints.\n"
            "Add temporary prints at key boundaries to verify shapes/dtypes on first run:\n"
            "- forward() entry: input shapes, dtypes, device\n"
            "- After each major op: intermediate tensor shapes\n"
            "- get_batch(): output keys, shapes, dtypes\n"
            "One diagnostic run that confirms all shapes saves multiple blind training attempts.\n"
            "Remove prints after verification passes."
        )

    def _check_analysis_persistence(self, tool_name, arguments):
        """Remind to persist analysis before writing porting code."""
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""
        if self._analysis_persisted:
            return ""
        read_count = len(self._files_read_this_session)
        if read_count < self._MIN_READS_BEFORE_PORTING_WRITE:
            return ""  # Reading depth gate fires instead

        return (
            "\n\n[ANALYSIS PERSISTENCE] You've read enough code, but haven't persisted "
            "your analysis yet. Before writing porting code:\n"
            "1. Write your component mapping / architecture analysis to workspace_experiment "
            "or memory_write (>200 chars)\n"
            "2. This ensures the next session can pick up without re-reading everything.\n"
            "After persisting, this gate won't fire again."
        )

    _VERIFICATION_STAGES = ["none", "analysis", "init_ok", "forward_aligned", "backward_ok", "distributed_ok", "full_training"]

    def _check_verification_ladder(self, tool_name, arguments):
        """Enforce incremental verification for porting tasks."""
        if not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        is_full_training = self._TRAIN_LAUNCH_RE.search(cmd) and not self._is_quick_test_command(cmd)
        if is_full_training:
            is_full_training = self._regex_judge_confirm("is_training_launch", cmd)
        if not is_full_training:
            return ""
        if self._verification_stage in ("distributed_ok", "full_training"):
            return ""
        return (
            f"\n\n[VERIFICATION LADDER] Stage is '{self._verification_stage}'. "
            f"Verify the whole model before full training:\n"
            f"1. Weights: load_state_dict(strict=True) — zero missing/unexpected keys\n"
            f"2. Forward: real data batch → finite loss\n"
            f"3. Backward: --train-iters 20, verify loss decreases\n"
            f"4. Distributed: target TP/PP, check no hang\n"
            f"Record each stage with workspace_experiment update."
        )

    _PORTING_READ_CATEGORIES = {
        "source_model": re.compile(r'modeling_|model\.py|config\.json|configuration_'),
        "megatron_base": re.compile(r'megatron/.*(transformer|attention|mlp|language_model|gpt_model|spec)'),
        "te_attention": re.compile(r'transformer_engine.*attention|TEDotProductAttention|dot_product_attention'),
        "existing_impl": re.compile(r'flagscale/models/'),
        "checkpoint": re.compile(r'tools/checkpoint/|checkpoint_loader|convert'),
    }
    _MIN_CATEGORIES_BEFORE_WRITE = 3

    # ── Pipeline knowledge acquisition gate ────────────────────────────

    _PIPELINE_KNOWLEDGE_DIMENSIONS = {
        "train_script_anatomy": re.compile(
            r'train_gpt\.py|train_qwen3_vl\.py|train_qwen2_5_vl\.py'
        ),
        "model_provider_and_builder": re.compile(
            r'model_provider\.py|gpt_builders\.py'
        ),
        "megatron_layer_spec_system": re.compile(
            r'spec_utils\.py|gpt_layer_specs\.py|TransformerLayerSubmodules'
        ),
        "training_loop": re.compile(
            r'training/training\.py'
        ),
        "flagscale_custom_models": re.compile(
            r'flagscale/models/megatron/.*/layer_specs\.py|flagscale/models/megatron/.*_model\.py'
        ),
        "parallelism_system": re.compile(
            r'parallel_state\.py|tensor_parallel.*layers\.py|pipeline_parallel.*schedules\.py'
        ),
        "te_attention_system": re.compile(
            r'transformer_engine.*attention.*dot_product|transformer_engine.*backends'
            r'|megatron.*extensions.*transformer_engine'
        ),
    }
    _MIN_PIPELINE_DIMENSIONS = 6

    _PIPELINE_KNOWLEDGE_KEYWORDS = (
        "get_batch", "forward_step", "loss_func", "model_provider",
        "layer_spec", "ModuleSpec", "TransformerLayerSubmodules",
        "pretrain", "transformer_engine",
        "tensor_parallel", "pipeline_parallel", "context_parallel",
        "initialize_model_parallel", "ColumnParallelLinear",
    )
    _MIN_PIPELINE_KEYWORDS_IN_MEMORY = 10
    _PIPELINE_KNOWLEDGE_MEMORY_KEY = "megatron_pipeline_knowledge"

    def _check_pipeline_comprehension_gate(self, tool_name, arguments):
        """Hard block: require pipeline knowledge acquisition AND persistence before porting writes."""
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""

        # Phase 1: Check reading coverage of pipeline knowledge dimensions
        covered = set()
        for path in self._files_read_this_session:
            for dim, pattern in self._PIPELINE_KNOWLEDGE_DIMENSIONS.items():
                if pattern.search(path):
                    covered.add(dim)

        if len(covered) < self._MIN_PIPELINE_DIMENSIONS:
            missing = set(self._PIPELINE_KNOWLEDGE_DIMENSIONS.keys()) - covered
            dim_descs = {
                "train_script_anatomy": "Training script (get_batch, loss_func, forward_step): read train_gpt.py or a VL train script",
                "model_provider_and_builder": "Model construction: read model_provider.py or gpt_builders.py",
                "megatron_layer_spec_system": "Megatron layer_spec system: read Megatron-LM-FL spec_utils.py or gpt_layer_specs.py (ModuleSpec, TransformerLayerSubmodules, how specs compose)",
                "training_loop": "Training loop (pretrain): read flagscale/train/megatron/training/training.py",
                "flagscale_custom_models": "FlagScale custom models: read flagscale/models/megatron/<model>/layer_specs.py or *_model.py",
                "parallelism_system": "Megatron-LM-FL parallelism: read parallel_state.py (initialize_model_parallel, process groups), tensor_parallel/layers.py (ColumnParallelLinear), or pipeline_parallel/schedules.py",
                "te_attention_system": "TransformerEngine attention: read TEDotProductAttention (megatron/core/extensions/transformer_engine.py) AND TE's DotProductAttention/backends (transformer_engine/pytorch/attention/) — understand attn_mask_type, qkv_format, backend selection, CP integration",
            }
            missing_list = "\n".join(f"  - {dim_descs[m]}" for m in sorted(missing))
            return (
                f"\n\n[PIPELINE COMPREHENSION GATE] You're writing porting code but haven't "
                f"studied the training pipeline deeply enough. Covered {len(covered)}/{self._MIN_PIPELINE_DIMENSIONS} "
                f"required dimensions.\n\n"
                f"Missing knowledge:\n{missing_list}\n\n"
                f"Model porting requires understanding the FULL end-to-end flow: "
                f"data loading → model construction (layer_spec) → forward/loss → training loop → parallelism.\n"
                f"The layer_spec system (ModuleSpec + TransformerLayerSubmodules in Megatron-LM-FL) "
                f"is the backbone — it defines how TE layers, attention, MLP are composed.\n"
                f"The parallelism system (initialize_model_parallel, TP/PP/DP/CP/EP/SP) is the "
                f"infrastructure — model modules use TP primitives internally.\n"
                f"Read these files FIRST, then write code."
            )

        # Phase 2: Check knowledge persistence
        if not self._pipeline_knowledge_persisted:
            return (
                "\n\n[KNOWLEDGE PERSISTENCE GATE] You've read the pipeline code — good. "
                "Now PERSIST your understanding before writing.\n\n"
                "Call memory_write(key='megatron_pipeline_knowledge', content='...') with a "
                "structured summary covering:\n"
                "- How data flows (get_batch → model input)\n"
                "- How models are constructed (model_provider → layer_spec → TE layers)\n"
                "- The layer_spec system: ModuleSpec, TransformerLayerSubmodules, how FlagScale extends it\n"
                "- Parallelism: initialize_model_parallel, TP (ColumnParallelLinear), PP (schedules), CP/EP/SP\n"
                "- How to add a new model (what files to create, what to register)\n\n"
                "KEY MUST BE exactly 'megatron_pipeline_knowledge'.\n"
                "Your memory must mention at least 10 of the 14 required keywords. "
                "This knowledge must survive context compaction. Write it NOW."
            )

        # Phase 3: LLM self-confirmation for THIS specific porting task
        # Auto-pass after 3 blocks to prevent deadlock (the agent has read and persisted,
        # which is the real gate — confirmation is nice-to-have)
        if not self._pipeline_knowledge_confirmed:
            if not hasattr(self, '_pipeline_confirm_block_count'):
                self._pipeline_confirm_block_count = 0
            self._pipeline_confirm_block_count += 1
            if self._pipeline_confirm_block_count >= 3:
                self._pipeline_knowledge_confirmed = True
                logger.info("Pipeline knowledge confirmation auto-passed after 3 blocks")
                return ""
            return (
                "\n\n[KNOWLEDGE CONFIRMATION GATE] You've read the code and persisted knowledge. "
                "Now CONFIRM: for THIS specific porting task, is your knowledge sufficient?\n\n"
                "You MUST include in your response ONE of these markers:\n"
                "[PIPELINE_KNOWLEDGE_CONFIRMED: YES]\n"
                "Task: <what model you're porting>\n"
                "Covered: <key knowledge points you have>\n\n"
                "OR:\n"
                "[PIPELINE_KNOWLEDGE_CONFIRMED: NO]\n"
                "Missing: <what you still need to learn>\n\n"
                "If NO, go read the missing files first. If YES, you may proceed with writing code."
            )

        return ""

    def _check_reading_quality(self, tool_name, arguments):
        """Ensure agent reads the RIGHT files, not just enough files."""
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""
        read_count = len(self._files_read_this_session)
        if read_count < self._MIN_READS_BEFORE_PORTING_WRITE:
            return ""  # Depth gate handles this
        covered = set()
        for path in self._files_read_this_session:
            for cat, pattern in self._PORTING_READ_CATEGORIES.items():
                if pattern.search(path):
                    covered.add(cat)
        if len(covered) >= self._MIN_CATEGORIES_BEFORE_WRITE:
            return ""
        missing = set(self._PORTING_READ_CATEGORIES.keys()) - covered
        descs = {
            "source_model": "Source model code (modeling_*.py, config.json)",
            "megatron_base": "Megatron base classes (transformer, attention, MLP, spec)",
            "te_attention": "TE attention stack (TEDotProductAttention, DotProductAttention, backends.py)",
            "existing_impl": "Existing FlagScale implementations (flagscale/models/)",
            "checkpoint": "Checkpoint conversion code (tools/checkpoint/)",
        }
        missing_list = "\n".join(f"  - {descs[m]}" for m in sorted(missing))
        return (
            f"\n\n[READING QUALITY] You've read {read_count} files but missed "
            f"critical categories:\n{missing_list}\n"
            f"Read at least one file from each missing category before writing porting code."
        )

    # ── Data pipeline comprehension gate ────────────────────────────────

    _DATA_WRITE_PATHS = re.compile(
        r'(preprocess_data|dataset|dataloader|data_utils|task_encoder|'
        r'energon.*config|data.*pipeline|tokenize|data/.*\.py)'
    )
    _DATA_READ_CATEGORIES = {
        "source_format": re.compile(r'\.jsonl|\.json|\.parquet|\.csv|\.tar|webdataset|raw.*data|sample'),
        "processing": re.compile(r'preprocess|tokeniz|encode|transform|convert|pipeline|task_encoder'),
        "model_input": re.compile(r'get_batch|data_provider|dataloader|dataset.*\.py|collat'),
    }
    _MIN_DATA_READS = 5
    _MIN_DATA_CATEGORIES = 2

    def _check_data_pipeline_gate(self, tool_name, arguments):
        """Block data processing code writes until agent understands the full pipeline."""
        if not self._data_prep_mode or self._data_pipeline_understood:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._DATA_WRITE_PATHS.search(target):
            return ""

        data_reads = 0
        covered = set()
        for path in self._files_read_this_session:
            for cat, pattern in self._DATA_READ_CATEGORIES.items():
                if pattern.search(path):
                    covered.add(cat)
                    data_reads += 1

        issues = []
        if data_reads < self._MIN_DATA_READS:
            issues.append(
                f"Read only {data_reads} data-related files (minimum: {self._MIN_DATA_READS})"
            )
        if len(covered) < self._MIN_DATA_CATEGORIES:
            missing = set(self._DATA_READ_CATEGORIES.keys()) - covered
            descs = {
                "source_format": "Source data format (raw files, JSONL, tar, samples)",
                "processing": "Processing code (preprocess, tokenize, encode, TaskEncoder)",
                "model_input": "Model input interface (get_batch, dataloader, dataset class)",
            }
            issues.append(
                "Missing categories:\n" +
                "\n".join(f"    - {descs[m]}" for m in sorted(missing))
            )

        if not issues:
            return ""

        return (
            "\n\n[DATA PIPELINE GATE] You must understand the full data pipeline before "
            "writing data processing code.\n\n"
            "Trace the chain: source format → processing operations → model input.\n"
            + "\n".join(f"  - {i}" for i in issues) + "\n\n"
            "Read the relevant source code, then persist your findings to memory with "
            "memory_write (include: source format, key transformations, and how data "
            "enters the model). The gate clears after you persist pipeline understanding."
        )

    def _check_data_parallelism_gate(self, tool_name, arguments):
        """HARD BLOCK: Enforce parallelism strategy awareness when implementing data pipeline.

        Data pipeline implementation MUST consider ALL parallelism dimensions from the start.
        This prevents the common failure of implementing data pipeline without considering
        parallelism, then having to rewrite it.
        """
        if not self._porting_mode and not self._data_prep_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""

        target = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")

        # Detect data pipeline implementation
        is_data_pipeline = (
            "get_batch" in content or
            "dataset" in target.lower() or
            re.search(r'class.*Dataset|def get_batch|DataLoader|data_provider', content)
        )

        if not is_data_pipeline:
            return ""

        # Check if parallelism strategy has been documented
        entries = self.session_memory.list_entries()
        has_parallelism_doc = any(
            any(kw in (e.get("content") or "").lower()
                for kw in ("tp=", "pp=", "dp=", "ep=", "cp=", "sp=",
                           "tensor parallel", "pipeline parallel", "data parallel",
                           "expert parallel", "context parallel", "sequence parallel",
                           "broadcast_data", "parallelism strategy", "parallel strategy"))
            for e in entries
        )

        # Check if content mentions parallelism handling
        content_mentions_parallelism = any(
            kw in content.lower()
            for kw in ("broadcast_data", "tensor_model_parallel", "pipeline_model_parallel",
                       "data_parallel_rank", "expert_parallel", "context_parallel",
                       "tp_rank", "pp_rank", "dp_rank", "ep_rank", "cp_rank",
                       "pre_process", "post_process", "get_data_parallel_group",
                       "get_tensor_model_parallel_group", "get_pipeline_model_parallel_group")
        )

        if has_parallelism_doc or content_mentions_parallelism:
            return ""  # Parallelism is considered

        return (
            "[DATA PARALLELISM GATE — BLOCKED]\n\n"
            "Data pipeline implementation BLOCKED. Parallelism strategy not documented.\n\n"
            "In distributed training, data pipeline MUST be designed with ALL parallelism dimensions:\n"
            "- TP (Tensor Parallel): All TP ranks receive IDENTICAL input\n"
            "  → Use broadcast_data() from megatron.training.utils\n"
            "- PP (Pipeline Parallel): Only first stage needs tokens, only last needs labels\n"
            "  → Guard with pre_process/post_process flags\n"
            "- DP (Data Parallel): Different micro-batch per rank\n"
            "  → Handled by sampler, don't break with global indexing\n"
            "- EP (Expert Parallel): Data routing to experts must account for expert sharding\n"
            "  → Token-to-expert assignment must be consistent across EP ranks\n"
            "- CP (Context Parallel): Sequence split across ranks\n"
            "  → Correct position IDs and attention masks per rank\n"
            "- SP (Sequence Parallel): Activation memory distributed along sequence dimension\n"
            "  → Automatically handled by framework when enabled with TP\n\n"
            "Before implementing get_batch:\n"
            "1. Document current parallelism strategy (TP/PP/DP/EP/CP/SP values)\n"
            "2. Explain how data will be distributed across ALL parallelism dimensions\n"
            "3. Identify which ranks need which data and in what format\n"
            "4. Consider special cases: MoE routing, long sequences, packed samples\n"
            "5. Save to memory (workspace_experiment or memory_write)\n\n"
            "Data pipeline and parallelism are NOT separable — design them together.\n"
        )

    # ── Lightweight understanding check (non-porting tasks) ─────────────

    _CONFIG_WRITE_PATTERNS = re.compile(
        r'(\.yaml|\.yml|config.*\.py|\.toml|\.cfg)$'
    )
    _CONFIG_CONTEXT_CATEGORIES = {
        "docs": re.compile(r'(getting.started|readme|doc|guide|tutorial)', re.IGNORECASE),
        "example_config": re.compile(r'(example|sample|template|default).*\.(yaml|yml|toml)', re.IGNORECASE),
        "existing_config": re.compile(r'(config|conf).*\.(yaml|yml|toml|py)', re.IGNORECASE),
    }

    def _check_config_understanding(self, tool_name, arguments):
        """Soft reminder: read docs/examples before writing configs from scratch.

        Only fires once per session, only for write_file (not edit_file),
        and only when the agent hasn't read any relevant context yet.
        Not a hard block — just a nudge toward better practice.
        """
        if getattr(self, '_config_understanding_fired', False):
            return ""
        if tool_name != "write_file":
            return ""
        target = arguments.get("path", "")
        if not self._CONFIG_WRITE_PATTERNS.search(target):
            return ""
        # Check if agent has read any relevant context
        covered = set()
        for path in self._files_read_this_session:
            for cat, pattern in self._CONFIG_CONTEXT_CATEGORIES.items():
                if pattern.search(path):
                    covered.add(cat)
        if covered:
            return ""  # Agent has done some reading, trust its judgment
        # Only fire if truly zero context reading
        if len(self._files_read_this_session) >= 3:
            return ""  # Agent has read files, just not matching our patterns
        self._config_understanding_fired = True
        return (
            "\n\n[UNDERSTANDING NOTE] You're writing a config file but haven't read "
            "any documentation or example configs yet. For complex configurations "
            "(parallelism, model architecture, data pipelines), reading an existing "
            "example first significantly reduces trial-and-error iterations."
        )

    # ── Checkpoint verification gate ────────────────────────────────────

    def _check_phase_ordering_gate(self, tool_name, arguments):
        """HARD BLOCK: Enforce strict phase ordering in porting workflow.

        Prevents:
        - Checkpoint conversion before model structure is complete
        - Data pipeline work before checkpoint is converted
        - Training launch before data pipeline is ready

        This addresses the common failure of doing work out of order, which
        wastes time and must be redone.
        """
        if not self._porting_mode:
            return ""

        current_phase = getattr(self, "_current_phase", "analysis")
        if current_phase == "complete":
            return ""  # All phases complete, no restrictions

        # Define phase order indices
        phase_order = [
            "analysis",
            "structure_implementation",
            "structure_verification",
            "data_pipeline",
            "checkpoint_conversion",
            "training_verification",
            "complete",
        ]
        current_idx = phase_order.index(current_phase) if current_phase in phase_order else 0

        # Detect checkpoint conversion attempts
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            if re.search(r'convert.*checkpoint|checkpoint.*convert|ckpt.*convert|convert.*ckpt', cmd, re.I):
                if current_idx < phase_order.index("checkpoint_conversion"):
                    return (
                        f"[PHASE ORDERING GATE — BLOCKED]\n\n"
                        f"Current phase: {current_phase}\n"
                        f"Checkpoint conversion is NOT allowed yet.\n\n"
                        f"Required order:\n"
                        f"1. Complete model structure implementation\n"
                        f"2. Verify structure completeness (all components present)\n"
                        f"3. Implement data pipeline with parallelism support\n"
                        f"4. THEN convert checkpoint\n\n"
                        f"Converting checkpoint before data pipeline is ready wastes time — "
                        f"data pipeline design may reveal missing model interfaces.\n"
                    )

        # Detect data pipeline implementation
        if tool_name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            is_data_pipeline = (
                "get_batch" in content or
                "dataset" in path.lower() or
                "data_prep" in path.lower() or
                re.search(r'class.*Dataset|def get_batch', content)
            )
            if is_data_pipeline and current_idx < phase_order.index("data_pipeline"):
                return (
                    f"[PHASE ORDERING GATE — BLOCKED]\n\n"
                    f"Current phase: {current_phase}\n"
                    f"Data pipeline implementation is NOT allowed yet.\n\n"
                    f"Required order:\n"
                    f"1. Complete model structure\n"
                    f"2. Verify structure completeness\n"
                    f"3. THEN implement data pipeline\n\n"
                    f"Data pipeline must be designed with the final model structure in mind.\n"
                )

        # Detect training launch
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            if self._is_training_launch(cmd) and not self._is_quick_test_command(cmd):
                if current_idx < phase_order.index("training_verification"):
                    return (
                        f"[PHASE ORDERING GATE — BLOCKED]\n\n"
                        f"Current phase: {current_phase}\n"
                        f"Full training launch is NOT allowed yet.\n\n"
                        f"Complete these phases first:\n"
                        f"- Model structure implementation and verification\n"
                        f"- Data pipeline with parallelism support\n"
                        f"- Checkpoint conversion\n"
                    )

        return ""

    def _check_structure_completeness_gate(self, tool_name, arguments):
        """HARD BLOCK: Prevent checkpoint conversion before model structure is verified complete.

        Enforces the correct porting order:
        1. Enumerate all source model components
        2. Implement all components in target
        3. Verify structure completeness
        4. THEN do checkpoint conversion

        This prevents the common failure of converting checkpoints for incomplete models,
        which wastes time and must be redone after adding missing components.
        """
        if not self._porting_mode:
            return ""
        if tool_name != "shell" and tool_name != "write_file":
            return ""

        # Detect checkpoint conversion attempts
        is_ckpt_conversion = False
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            if re.search(r'convert.*checkpoint|checkpoint.*convert|ckpt.*convert|convert.*ckpt|convert.*weight|weight.*convert', cmd, re.I):
                is_ckpt_conversion = True
        elif tool_name == "write_file":
            path = arguments.get("path", "")
            if re.search(r'convert.*ckpt|ckpt.*convert|convert.*checkpoint|checkpoint.*convert', path, re.I):
                is_ckpt_conversion = True

        if not is_ckpt_conversion:
            return ""

        # Check if structure completeness has been verified
        if getattr(self, '_structure_completeness_verified', False):
            return ""

        # Check memory for structure enumeration evidence
        entries = self.session_memory.list_entries()
        has_enumeration = any(
            any(kw in (e.get("content") or "").lower()
                for kw in ("component checklist", "structure enumeration", "all components",
                           "module tree", "total parameters", "porting checklist"))
            for e in entries
        )

        if has_enumeration:
            # Enumeration exists but not explicitly verified — soft warning
            return (
                "[STRUCTURE COMPLETENESS CHECK]\n"
                "You have a component enumeration in memory. Before converting checkpoints, "
                "confirm ALL components from the checklist are implemented:\n"
                "1. Compare your implemented module tree against the source enumeration\n"
                "2. Verify parameter count matches (within 1% tolerance)\n"
                "3. If all components are present, proceed with conversion\n"
                "4. If any are missing, implement them FIRST — partial conversion wastes time\n"
            )

        # No enumeration at all — hard block
        return (
            "[STRUCTURE COMPLETENESS GATE — BLOCKED]\n\n"
            "Checkpoint conversion BLOCKED. Model structure completeness not verified.\n\n"
            "You MUST complete these steps before checkpoint conversion:\n"
            "1. Enumerate ALL source model components (run model.named_modules() or read __init__)\n"
            "2. Create a porting checklist with every component and its parameter count\n"
            "3. Implement ALL components in the target model\n"
            "4. Verify: target module count and parameter count match source\n\n"
            "Save the checklist to memory (workspace_experiment or memory_write).\n"
            "Checkpoint conversion on an incomplete model wastes time — it must be redone "
            "after adding missing components.\n"
        )

    def _check_checkpoint_verified_gate(self, tool_name, arguments):
        """Verify checkpoint after conversion before using it in training."""
        if tool_name != "shell":
            return ""

        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Check if checkpoint conversion happened recently (within last 20 tool calls)
        recent_tools = list(self._recent_tool_calls)[-20:]
        has_conversion = any(
            "convert" in str(name).lower() and "checkpoint" in str(rest).lower()
            for name, *rest in recent_tools
        )

        if not has_conversion:
            return ""

        # Check if deep verification was done (inspect_checkpoint or torch.load with shape check)
        conversion_indices = [idx for idx, (n, *_r) in enumerate(recent_tools)
                             if "convert" in str(n).lower()]
        last_conversion_idx = conversion_indices[-1] if conversion_indices else -1
        has_deep_verification = any(
            ("inspect_checkpoint" in str(name) or
             ("torch.load" in str(rest) and "shape" in str(rest)) or
             "verify" in str(name).lower())
            and i > last_conversion_idx
            for i, (name, *rest) in enumerate(recent_tools)
        )

        if has_deep_verification:
            return ""

        ckpt_match = re.search(r'--load[=\s]+([^\s]+)', cmd)
        ckpt_path = ckpt_match.group(1) if ckpt_match else "<checkpoint_path>"

        return (
            "[CHECKPOINT VERIFICATION GATE]\n"
            f"Recently converted checkpoint ({ckpt_path}) not verified before training.\n"
            "Use `inspect_checkpoint(path=...)` to verify:\n"
            "- Shape/dtype correctness for all tensors\n"
            "- No all-zero, NaN, or Inf anomalies\n"
            "- Key count matches model expectation\n"
            "- Optionally compare vs reference with reference_path=...\n"
            "This catches 90% of conversion bugs in seconds — much cheaper than a failed training run.\n"
        )

    # ── Distributed training prerequisite gate ──────────────────────────

    def _check_distributed_prerequisite_gate(self, tool_name, arguments):
        """Inform about distributed training prerequisites, let LLM decide."""
        if tool_name != "shell":
            return ""

        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Extract parallelism
        tp = self._extract_arg_value(cmd, r'--tensor-model-parallel-size[=\s]+(\d+)')
        pp = self._extract_arg_value(cmd, r'--pipeline-model-parallel-size[=\s]+(\d+)')

        if tp <= 1 and pp <= 1:
            return ""  # Single GPU, no gate

        # Check if single-GPU verification exists in recent history
        recent_cmds = [rest[0] if rest else "" for name, *rest in list(self._recent_tool_calls)[-30:] if name == "shell"]
        has_single_gpu = any(
            self._is_training_launch(c) and
            self._extract_arg_value(c, r'--tensor-model-parallel-size[=\s]+(\d+)') <= 1 and
            self._extract_arg_value(c, r'--pipeline-model-parallel-size[=\s]+(\d+)') <= 1
            for c in recent_cmds
        )

        if has_single_gpu:
            return ""  # Already verified single-GPU

        return (
            "[DISTRIBUTED PREREQUISITE GATE]\n"
            f"Launching distributed (TP={tp}, PP={pp}) without single-GPU verification.\n"
            "Options: (1) Run single-GPU first (recommended for <30B, new architectures), "
            "(2) Skip if MoE/very large model, (3) Test custom layers separately.\n"
        )

    # ── Phase 1 Gates: Pre-Implementation ──────────────────────────────

    def _check_understanding_verification_gate(self, tool_name, arguments):
        """A1: Before writing training code, verify agent understands data flow, model signature, parallelism."""
        if self._understanding_verified or not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        path = arguments.get("path", "") or arguments.get("file_path", "")
        if not re.search(r'train_.*\.py|get_batch|forward_step|pretrain_', path):
            return ""
        return (
            "[UNDERSTANDING VERIFICATION GATE]\n"
            "Writing training/model code without demonstrating understanding.\n"
            "Answer in workspace_experiment: (1) Data flow — input format, get_batch output, "
            "(2) Model signature — forward() args/returns, loss fn, "
            "(3) Parallelism — which layers get TP, how DP handles data.\n"
        )

    def _check_component_isolation_gate(self, tool_name, arguments):
        """A3: For multi-component models, require whole-model implementation plan."""
        if self._component_plan_created or not self._porting_mode:
            return ""
        if tool_name != "write_file":
            return ""
        path = arguments.get("path", "")
        if not re.search(r'train_.*\.py|pretrain_', path):
            return ""

        # Detect multi-component model from recent reads
        recent_content = " ".join(str(rest) for _, *rest in list(self._recent_tool_calls)[-30:])
        multimodal_signals = sum(1 for kw in (
            "vision", "vit", "clip", "siglip", "encoder", "decoder",
            "vae", "diffusion", "moe", "expert", "router", "multimodal",
            "image_processor", "visual", "audio", "speech"
        ) if kw in recent_content.lower())

        if multimodal_signals < 2:
            self._component_plan_created = True  # Single-component model, skip
            return ""

        return (
            "[WHOLE-MODEL PLAN GATE]\n"
            "Multi-component model detected. Create a plan (plan_create) that builds the "
            "COMPLETE nested Module first — one top-level class that owns all sub-modules "
            "(vision, language, generation, projections). Analysis can be per-component, "
            "but implementation must be whole-model.\n"
            "Plan should cover: (1) complete Module structure with all sub-modules, "
            "(2) checkpoint conversion for ALL weights in one pass, "
            "(3) real data get_batch implementation, "
            "(4) attention mechanism mapping (source attention → TE or native fallback).\n"
        )

    _FLAGSCALE_RUNPY_RE = re.compile(
        r'python\s+.*run\.py\s+.*(?:action\s*=\s*run|--config-path|--config-name)',
        re.IGNORECASE,
    )

    def _check_use_flagscale_cli_gate(self, tool_name, arguments):
        """Hard block: forbid python run.py for training. Must use flagscale train CLI."""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._FLAGSCALE_RUNPY_RE.search(cmd):
            return ""
        if self._is_quick_test_command(cmd):
            return ""
        return (
            "[LAUNCHER GATE — COMMAND NOT EXECUTED]\n\n"
            "Direct `python run.py` launch is FORBIDDEN.\n"
            "Use the FlagScale CLI instead:\n\n"
            "  flagscale train <model> --config <path/to/config.yaml>\n\n"
            "Example:\n"
            "  flagscale train qwen3 --config examples/qwen3/conf/train.yaml\n\n"
            "The CLI handles process management, logging, and cleanup.\n"
            f"Blocked command: {cmd[:200]}\n"
        )

    def _check_experiment_gate(self, tool_name, arguments):
        """Hard block: reject training launch without a pending attempt.

        Always checks the experiment manager directly — no reliance on flags.
        Two-level check:
        1. At least one experiment must exist
        2. A pending attempt must exist for the current/latest experiment
        """
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._TRAIN_LAUNCH_RE.search(cmd):
            return ""
        if self._is_quick_test_command(cmd):
            return ""
        # LLM confirmation: is this really a training launch?
        if not self._regex_judge_confirm("is_training_launch", cmd):
            return ""

        # Check 1: any experiment exists?
        all_experiments = self._experiment_manager.list()
        if not all_experiments:
            return (
                "[EXPERIMENT GATE — COMMAND NOT EXECUTED]\n\n"
                "Training launch BLOCKED. No experiment registered.\n\n"
                "You MUST do the following before this command can run:\n"
                "1. workspace_experiment(action='create',\n"
                "     name='<descriptive_name>',\n"
                "     purpose='<why this experiment>',\n"
                "     hypothesis='<expected outcome and why>',\n"
                "     base_config={<initial config: model, TP, DP, batch_size, etc.>},\n"
                "     base_dir='<initial log directory>'\n"
                "   )\n"
                "2. workspace_experiment(action='add_attempt',\n"
                "     name='<same name>',\n"
                "     change='initial run',\n"
                "     config={<this run's full config>},\n"
                "     output_dir='<unique output dir for this run>'\n"
                "   )\n"
                "3. Then re-run the training command.\n\n"
                "ALL fields are REQUIRED. No empty values.\n"
                f"Blocked command: {cmd[:200]}\n"
            )

        # Check 2: pending attempt exists for the latest experiment?
        # Use current running experiment, or fall back to most recent one
        current_exp = self._experiment_manager.get_current_experiment()
        exp_name = current_exp or all_experiments[-1].get("name", "")
        if exp_name:
            exp = self._experiment_manager.read(exp_name)
            if exp:
                attempts = exp.get("attempts", [])
                has_pending = any(a.get("result") == "(pending)" for a in attempts)
                if not has_pending:
                    # Allow retry if last attempt failed recently (config-fix-and-retry pattern)
                    # The agent fixed something and wants to retry — auto-reset last failed attempt
                    if attempts and "fail" in str(attempts[-1].get("result", "")).lower():
                        last = attempts[-1]
                        last["result"] = "(pending)"
                        last["retry_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self._experiment_manager._save(exp_name, exp)
                        return ""
                    return (
                        "[EXPERIMENT GATE — COMMAND NOT EXECUTED]\n\n"
                        "Training launch BLOCKED. No pending attempt for this run.\n\n"
                        "Before EACH training launch, you MUST register an attempt:\n"
                        "  workspace_experiment(action='add_attempt',\n"
                        f"    name='{exp_name}',\n"
                        "    change='<what you changed since last attempt>',\n"
                        "    config={<this run's full config: model, TP, DP, batch_size, key flags>},\n"
                        "    output_dir='<UNIQUE output dir for THIS run — must differ from previous attempts>'\n"
                        "  )\n\n"
                        "ALL fields REQUIRED. output_dir must be unique across all attempts.\n"
                        "Then re-run the training command.\n"
                        f"Blocked command: {cmd[:200]}\n"
                    )

        return ""

    _MONITOR_GATE_MAX_BLOCKS = 5

    def _check_monitor_after_launch_gate(self, tool_name, arguments):
        """Hard block: after a real training launch (not dryrun), the next action MUST be monitor.

        Allows: monitor, plan_update, workspace_experiment, and read-only shell commands
        (pgrep, ps, ls, cat, tail, grep, find, head, wc, stat, file) since these are
        diagnostic and don't interfere with monitoring.

        Auto-clears after _MONITOR_GATE_MAX_BLOCKS consecutive blocks (prevents permanent
        deadlock when training exits before agent can call monitor), or when a diagnostic
        shell command reveals the training process is no longer running.
        """
        if not self._awaiting_monitor:
            return ""
        if tool_name == "monitor":
            self._awaiting_monitor = False
            self._monitor_gate_block_count = 0
            return ""
        if tool_name in ("plan_update", "workspace_experiment", "read_file"):
            return ""
        # Allow read-only shell commands (diagnostics before monitor)
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            is_read_only = bool(re.match(
                r'\s*(grep|find|cat|ls|head|tail|wc|file|stat|which|type|echo|pwd|'
                r'env|printenv|hostname|uname|date|id|whoami|ps|pgrep|nvidia-smi)\b',
                cmd
            ))
            if is_read_only:
                return ""

        if not hasattr(self, '_monitor_gate_block_count'):
            self._monitor_gate_block_count = 0
        self._monitor_gate_block_count += 1

        if self._monitor_gate_block_count >= self._MONITOR_GATE_MAX_BLOCKS:
            self._awaiting_monitor = False
            self._monitor_gate_block_count = 0
            logger.warning("Monitor gate auto-cleared after %d consecutive blocks",
                           self._MONITOR_GATE_MAX_BLOCKS)
            return ""

        return (
            "[MONITOR GATE — COMMAND NOT EXECUTED]\n\n"
            "After launching training, you MUST call monitor() to observe the process.\n"
            "Training was just launched — use monitor(output_dir=...) to watch for "
            "errors or progress before doing anything else.\n\n"
            "Read-only commands (pgrep, ps, cat, ls) are allowed for diagnostics.\n"
        )

    def _check_no_parallel_writes_gate(self, tool_name, arguments):
        """Hard block: non-read shell commands cannot be issued in parallel.

        This is checked at the batch level in _execute_tools, not here.
        This gate is a no-op placeholder — actual enforcement is in agent.py.
        """
        return ""

    def _check_no_delete_experiment_dir_gate(self, tool_name, arguments):
        """Hard block: experiment output directories with actual training data cannot be deleted.

        Only protects dirs where training has started (result != pending).
        Dryrun-only output (scripts) is NOT protected — it can be regenerated.
        """
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not re.search(r'\brm\s+.*-[rRf]', cmd):
            return ""
        # Only protect directories where training has actually run
        protected_dirs = set()
        try:
            for exp_info in self._experiment_manager.list():
                exp = self._experiment_manager.read(exp_info.get("name", ""))
                if exp:
                    for attempt in exp.get("attempts", []):
                        od = attempt.get("output_dir", "")
                        if not od:
                            continue
                        result = str(attempt.get("result", "")).lower()
                        # Only protect if training has actually started/completed
                        if result in ("(pending)", "pending", ""):
                            continue
                        protected_dirs.add(od.rstrip("/"))
        except Exception:
            pass

        if not protected_dirs:
            return ""

        for d in protected_dirs:
            if d and d in cmd:
                return (
                    "[DELETE PROTECTION GATE — COMMAND NOT EXECUTED]\n\n"
                    "This directory is registered as an experiment output directory "
                    "where training has run, and CANNOT be deleted.\n"
                    "It contains training logs, checkpoints, and metrics "
                    "essential for experiment tracking and reproducibility.\n\n"
                    f"Protected directory: {d}\n"
                    f"Blocked command: {cmd[:200]}\n"
                )
        return ""

    def _check_failure_mode_analysis_gate(self, tool_name, arguments):
        """B1: After writing training code, require failure mode analysis before first launch."""
        if self._failure_mode_analyzed or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        recent_writes = [
            rest for name, *rest in list(self._recent_tool_calls)[-20:]
            if name in ("write_file", "edit_file")
        ]
        has_training_code = any(
            re.search(r'train_|pretrain_|forward_step|get_batch', str(args))
            for args in recent_writes
        )
        if not has_training_code:
            return ""

        return (
            "[FAILURE MODE ANALYSIS GATE]\n\n"
            "You wrote training code and are about to launch without analyzing failure modes.\n"
            "Document in workspace_experiment: top 3 likely failures, detection method, and fix for each.\n"
            "Then re-launch.\n"
        )

    # ── Phase 2 Gates: Pre-Launch (consolidated) ──────────────────────────

    def _check_sanity_check_gate(self, tool_name, arguments):
        """B2: Before first training launch, require 4 sanity checks."""
        if self._sanity_checks_passed or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""
        if self._is_quick_test_command(cmd):
            return ""

        recent_launches = sum(
            1 for name, *rest in list(self._recent_tool_calls)[-30:]
            if name == "shell" and self._is_training_launch(str(rest))
            and not self._is_quick_test_command(str(rest))
        )
        if recent_launches > 0:
            return ""

        return (
            "[SANITY CHECK GATE]\n\n"
            "First real training launch. Verify before proceeding:\n"
            "1. Data check: get_batch returns expected shapes\n"
            "2. Model init: model builds without error, param count is reasonable\n"
            "3. Config: TP×PP×DP = world_size, batch_size divisible by DP\n"
            "4. Checkpoint (if loading): keys match model state_dict\n\n"
            "Run checks or explain why each skipped check is safe.\n"
        )

    def _check_config_model_consistency_gate(self, tool_name, arguments):
        """B4: After generating config, verify config keys match model __init__ parameters."""
        if self._config_model_verified or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        recent_writes = [
            str(rest) for name, *rest in list(self._recent_tool_calls)[-15:]
            if name == "write_file" and re.search(r'\.yaml|\.yml|config', str(rest))
        ]
        if not recent_writes:
            return ""

        return (
            "[CONFIG-MODEL CONSISTENCY GATE]\n\n"
            "You recently wrote a config and are launching training.\n"
            "Verify: each config key matches model __init__ parameter names exactly.\n"
            "Common mismatches: hidden_size vs hidden_dim, num_attention_heads vs n_heads, "
            "ffn_hidden_size vs intermediate_size.\n"
        )

    def _check_environment_consistency_gate(self, tool_name, arguments):
        """C2: Before training launch, verify installed packages are from correct paths."""
        if self._env_verified or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        self._env_verified = True

        return (
            "[ENVIRONMENT CONSISTENCY GATE]\n\n"
            "Quick check: run `python -c \"import megatron; print(megatron.__file__)\"` "
            "to verify packages point to your working environment.\n"
            "If FlagScale wrapper fails on imports, check compatible Megatron-LM-FL tag.\n"
        )

    def _check_component_integration_gate(self, tool_name, arguments):
        """C4: Before full training with multi-component model, verify whole-model forward pass works."""
        if self._component_integration_verified or not self._porting_mode:
            return ""
        if not self._component_plan_created:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""
        if self._is_quick_test_command(cmd):
            return ""

        if self._verification_stage in ("forward_aligned", "backward_ok", "distributed_ok", "full_training"):
            self._component_integration_verified = True
            return ""

        return (
            "[WHOLE-MODEL VERIFICATION GATE]\n\n"
            "Launching full training with multi-component model but the complete model "
            "hasn't produced a verified forward pass yet.\n"
            "Run a short validation (--train-iters 20) with real data first: "
            "load_state_dict(strict=True) → forward pass → finite loss → loss decreases.\n"
        )

    # ── Phase 3 Gates: Post-Launch (Informational) ───────────────────────

    def _check_gradient_health(self, cmd, result):
        """D1: After first iteration, check grad_norm is non-zero and finite."""
        if not self._porting_mode or not self._is_training_launch(cmd) or not result:
            return ""

        grad_norm_match = re.search(r'grad.norm[:\s]+([0-9.eE+\-]+|nan|inf)', result, re.I)
        if not grad_norm_match:
            return ""

        grad_val = grad_norm_match.group(1).lower()
        issues = []

        if grad_val in ("nan", "inf"):
            issues.append(f"grad_norm is {grad_val} — exploding gradients or NaN in forward")
        elif grad_val in ("0.0", "0"):
            issues.append("grad_norm is 0 — frozen params or detached tensor?")
        else:
            try:
                val = float(grad_val)
                if val > 1000:
                    issues.append(f"grad_norm={val} very large — add gradient clipping")
                elif val < 1e-10:
                    issues.append(f"grad_norm={val} near-zero — loss not connected to all params?")
            except ValueError:
                pass

        zero_grad_match = re.search(r'zero.grad.*?([0-9.]+)%', result, re.I)
        if zero_grad_match:
            ratio = float(zero_grad_match.group(1))
            if ratio > 50:
                issues.append(f"zero_grad_ratio={ratio}% — over half of params have zero gradients")

        if not issues:
            return ""
        return "[GRADIENT HEALTH]\n" + "\n".join(f"- {i}" for i in issues)

    def _check_loss_sanity(self, cmd, result):
        """D2: After step 0 and step 10, check loss is reasonable."""
        if not self._porting_mode or not result:
            return ""

        loss_matches = re.findall(r'(?:loss|lm.loss)[:\s]+([0-9.eE+\-]+)', result, re.I)
        if not loss_matches:
            return ""

        issues = []
        try:
            losses = [float(l) for l in loss_matches[:10]]
        except ValueError:
            return ""
        if not losses:
            return ""

        first_loss = losses[0]
        if first_loss > 20:
            issues.append(f"Initial loss={first_loss:.2f} unusually high (expected ~10-12 for random init)")
        elif first_loss < 0.1:
            issues.append(f"Initial loss={first_loss:.4f} suspiciously low — data leak?")

        if len(losses) >= 3:
            if all(losses[i] >= losses[i-1] for i in range(1, min(5, len(losses)))):
                issues.append(f"Loss not decreasing: {losses[:5]} — check LR or data pipeline")
            if any(l != l or l == float('inf') for l in losses):
                issues.append("Loss contains NaN/Inf — critical")

        if not issues:
            return ""
        return "[LOSS SANITY]\n" + "\n".join(f"- {i}" for i in issues)

    def _check_component_gradient_flow(self, cmd, result):
        """D3: For multi-component models, check all components receive gradients."""
        if not self._porting_mode or not self._component_plan_created or not result:
            return ""

        no_grad_components = re.findall(
            r'(vision|vit|encoder|decoder|vae|router|expert).*?grad.*?(?:None|0\.0)', result, re.I)
        if not no_grad_components:
            return ""
        components = list(set(c.lower() for c in no_grad_components))
        return (
            f"[COMPONENT GRADIENT FLOW]\n"
            f"Components with zero/missing gradients: {', '.join(components)}\n"
            f"Check for .detach(), torch.no_grad(), or missing connections.\n"
        )

    def _check_checkpoint_numerical_verification(self, cmd, result):
        """D4: After checkpoint loading, verify tensor statistics match HF originals."""
        if not self._porting_mode or not result:
            return ""
        if not re.search(r'(?:loaded|loading|checkpoint|ckpt).*(?:success|done|complete)|successfully loaded', result, re.I):
            return ""
        if re.search(r'mean.*std.*match|tensor.*verification.*pass|numerical.*check.*ok', result, re.I):
            return ""
        return (
            "[CHECKPOINT NUMERICAL VERIFICATION]\n"
            "Checkpoint loaded but numerical correctness NOT verified.\n"
            "Compare key tensor mean/std against HF originals to catch transposition/permutation bugs.\n"
        )

    def _check_gpu_zombie_escalation(self, cmd, result):
        """Detect GPU zombie processes and provide escalation strategy."""
        if not result:
            return ""
        zombie_indicators = [
            re.search(r'CUDA out of memory', result, re.I),
            re.search(r'RuntimeError.*CUDA.*OOM', result, re.I),
            re.search(r'No running processes found.*MiB.*[1-9]', result),
            re.search(r'memory.used.*[1-9]\d{3,}.*MiB.*\|\s*0%', result),
        ]
        if not any(zombie_indicators):
            return ""
        return (
            "[GPU ZOMBIE DETECTED]\n"
            "GPU memory occupied but no active process. Steps:\n"
            "1. nvidia-smi — identify PIDs\n"
            "2. kill -9 <PID>\n"
            "3. fuser -v /dev/nvidia* — find all GPU holders\n"
            "4. fuser -k /dev/nvidia* — force-kill\n"
            "Prevention: pkill -f torchrun before relaunching.\n"
        )

    # ── Phase 4 Gates: Specialized ───────────────────────────────────────

    def _check_import_verification_gate(self, tool_name, arguments):
        """A4: Before writing train_*.py, verify critical imports resolve."""
        if self._imports_verified or not self._porting_mode:
            return ""
        if tool_name != "write_file":
            return ""
        path = arguments.get("path", "")
        if not re.search(r'train_.*\.py|pretrain_', path):
            return ""

        content = arguments.get("content", "")
        imports = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
        critical_imports = [
            imp for imp in imports
            if any(kw in imp for kw in ("megatron", "transformer_engine", "flagscale", "apex"))
        ]
        if not critical_imports:
            return ""

        return (
            "[IMPORT VERIFICATION GATE]\n"
            f"Critical imports detected: {', '.join(critical_imports[:5])}\n"
            "Run `python -c \"import <pkg>\"` for each before writing to avoid wasted launch attempts.\n"
        )

    def _check_tp_compatibility_gate(self, tool_name, arguments):
        """C3: Before TP>1 with custom layers, verify sharded_state_dict() exists."""
        if not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        tp = self._extract_arg_value(cmd, r'tensor.model.parallel.size[=\s]+(\d+)')
        if tp <= 1:
            tp = self._extract_arg_value(cmd, r'--tp[=\s]+(\d+)')
        if tp <= 1:
            return ""

        if not self._code_written:
            return ""

        recent_writes = [
            str(rest) for name, *rest in list(self._recent_tool_calls)[-30:]
            if name in ("write_file", "edit_file")
        ]
        has_custom_layers = any(
            re.search(r'class.*Layer|class.*Attention|class.*MLP|class.*Embed', str(w))
            for w in recent_writes
        )
        if not has_custom_layers:
            return ""

        return (
            "[TP COMPATIBILITY GATE]\n"
            f"TP={tp} with custom layers. Verify:\n"
            "1. Custom layers implement sharded_state_dict() for checkpoint save/load\n"
            "2. Linear layers use ColumnParallelLinear/RowParallelLinear\n"
            "3. Embedding uses VocabParallelEmbedding\n"
        )

    def _check_reference_comparison_gate(self, tool_name, arguments):
        """A5: Before writing model code, require comparison strategy against reference."""
        if self._reference_comparison_planned or not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        path = arguments.get("path", "") or arguments.get("file_path", "")
        if not re.search(r'model|layer|attention|mlp|embed', path, re.I):
            return ""

        if len(self._files_read_this_session) < 5:
            return ""

        return (
            "[REFERENCE COMPARISON GATE]\n"
            "Writing model code without a comparison strategy.\n"
            "Document: what reference implementation are you comparing against? "
            "How will you verify numerical equivalence?\n"
        )

    def _extract_arg_value(self, cmd: str, pattern: str) -> int:
        """Extract integer argument value from command."""
        match = re.search(pattern, cmd)
        return int(match.group(1)) if match else 1

    # ── Consolidated gate dispatch ─────────────────────────────────────

    def _run_pre_execution_gates(self, tool_name, arguments):
        """Run all pre-execution gates. Returns (hard_block, warnings).

        hard_block: str or None — if set, tool execution is blocked and this is returned.
        warnings: str — concatenated soft warnings to append to tool result.
        """
        # Hard-block gates (return immediately if triggered)
        hard_block_gates = [
            self._check_use_flagscale_cli_gate,
            self._check_monitor_after_launch_gate,
            self._check_no_delete_experiment_dir_gate,
            self._check_porting_path_gate,
            self._check_porting_path_deviation_gate,
            self._check_pipeline_comprehension_gate,
            self._check_data_pipeline_gate,
            self._check_data_parallelism_gate,
            self._check_understanding_verification_gate,
            self._check_component_isolation_gate,
            self._check_phase_ordering_gate,
            self._check_structure_completeness_gate,
            self._check_experiment_gate,
        ]
        for gate in hard_block_gates:
            warning = gate(tool_name, arguments)
            if warning:
                return warning, ""

        # Progress gate (special: returns tuple)
        progress_warning, progress_hard_block = self._check_progress_gate(tool_name)
        if progress_hard_block:
            return progress_warning, ""

        # Plan creation gate (hard block if "TOOL NOT EXECUTED" in result)
        plan_gate_warning = self._check_plan_creation_gate(tool_name)
        if plan_gate_warning and "TOOL NOT EXECUTED" in plan_gate_warning:
            return plan_gate_warning, ""

        # Soft-warning gates (collected and appended to result)
        soft_warnings = []
        soft_gates = [
            self._check_reading_depth_gate,
            self._check_reading_quality,
            self._check_import_verification_gate,
            self._check_reference_comparison_gate,
            self._check_checkpoint_verified_gate,
            self._check_distributed_prerequisite_gate,
            self._check_failure_mode_analysis_gate,
            self._check_sanity_check_gate,
            self._check_config_model_consistency_gate,
            self._check_environment_consistency_gate,
            self._check_tp_compatibility_gate,
            self._check_component_integration_gate,
            self._check_error_escalation,
            self._check_source_reading_gate,
            self._check_diagnostic_print_hint,
            self._check_analysis_persistence,
            self._check_verification_ladder,
            self._check_config_understanding,
        ]
        _MAX_SOFT_WARNINGS = 3
        for gate in soft_gates:
            w = gate(tool_name, arguments)
            if w:
                soft_warnings.append(w)
                if len(soft_warnings) >= _MAX_SOFT_WARNINGS:
                    break

        if plan_gate_warning:
            soft_warnings.append(plan_gate_warning)
        if progress_warning:
            soft_warnings.append(progress_warning)

        # Plan maintenance gate (soft warning only)
        plan_maint_warning = self._check_plan_maintenance_gate(tool_name)
        if plan_maint_warning:
            soft_warnings.append(plan_maint_warning)

        # Config validation hint after YAML write
        config_hint = self._check_config_validation_hint(tool_name)
        if config_hint:
            soft_warnings.append(config_hint)

        return None, "\n".join(soft_warnings)

    def _run_post_execution_gates(self, cmd, result):
        """Run post-execution gates on shell results. Returns additional info to append."""
        info_parts = []

        # Training health checks (only for training commands)
        if self._TRAIN_CMD_RE.search(cmd) if hasattr(self, '_TRAIN_CMD_RE') else False:
            for gate in (self._check_gradient_health, self._check_loss_sanity,
                         self._check_component_gradient_flow,
                         self._check_checkpoint_numerical_verification):
                info = gate(cmd, result)
                if info:
                    info_parts.append(info)

        # GPU zombie escalation (all shell commands)
        zombie_info = self._check_gpu_zombie_escalation(cmd, result)
        if zombie_info:
            info_parts.append(zombie_info)

        return "\n".join(info_parts)

