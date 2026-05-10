"""Enforcement gates for ReactAgent — progress, plan, dry-run, training, and phase gates."""

import json
import logging
import re
import time
import threading

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

    # ── LLM call timeout wrapper ─────────────────────────────────────────

    _LLM_JUDGE_TIMEOUT = 30  # seconds

    def _llm_call_with_timeout(self, llm_fn, prompt, timeout=None):
        """Call an LLM function with a timeout. Returns None on timeout."""
        timeout = timeout or self._LLM_JUDGE_TIMEOUT
        result_container = [None]
        error_container = [None]

        def _call():
            try:
                result_container[0] = llm_fn(prompt)
            except Exception as e:
                error_container[0] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            logger.warning("LLM judge call timed out after %ds", timeout)
            return None
        if error_container[0]:
            raise error_container[0]
        return result_container[0]

    # ── LLM-based memory knowledge check (shared by all gates) ──────────

    def _llm_check_memory_has_knowledge(self, entries, knowledge_description: str) -> bool:
        """Use LLM to judge whether memory entries contain the described knowledge.

        Falls back to keyword heuristic if LLM is unavailable.
        """
        if not entries:
            return False

        llm_fn = getattr(self.session_memory, '_llm_fn', None)
        if not llm_fn:
            return False

        # Build concise memory summary for LLM
        summaries = []
        for e in entries[:20]:
            key = e.get("key", "?")
            content = (e.get("content") or "")[:200]
            summaries.append(f"- [{key}]: {content}")

        if not summaries:
            return False

        prompt = (
            "You are checking whether stored memory entries contain specific knowledge.\n\n"
            f"REQUIRED KNOWLEDGE: {knowledge_description}\n\n"
            "MEMORY ENTRIES:\n" + "\n".join(summaries) + "\n\n"
            "Does the memory contain this knowledge (even partially or under a different name)? "
            "Answer ONLY 'yes' or 'no':"
        )

        try:
            response = self._llm_call_with_timeout(llm_fn, prompt)
            if response is None:
                return True  # On timeout, don't block
            return response.strip().lower().startswith("yes")
        except Exception as e:
            logger.warning("LLM memory check failed: %s, defaulting to pass", e)
            return True  # On failure, don't block

    def _llm_judge_is_genuine_megatron_native(self, content: str, path: str) -> bool:
        """Use LLM to judge whether code with both HF and Megatron symbols is genuinely Megatron-native.

        Returns True if the code is genuine Megatron-native (HF is only for weight loading/reference).
        Returns False if the code wraps/delegates to HF models at runtime.
        """
        llm_fn = getattr(self.session_memory, '_llm_fn', None)
        if not llm_fn:
            return False  # Can't judge, block conservatively

        # Truncate content for LLM context
        code_snippet = content[:3000]

        prompt = (
            "You are reviewing code for a Megatron Native model implementation.\n\n"
            "RULE: In Megatron Native (Mode B), ALL components of the model — including frozen "
            "ones — MUST be implemented using Megatron/TransformerEngine primitives. There is ONE "
            "top-level MegatronModule that owns everything. Whether a component is frozen "
            "(requires_grad=False) is a training decision, NOT an architecture decision.\n\n"
            "HuggingFace model classes are ONLY acceptable for:\n"
            "- Weight conversion utilities (loading HF checkpoint → Megatron state_dict)\n"
            "- Reference comparison during testing\n"
            "- Config parsing\n\n"
            "HuggingFace models are NOT acceptable as:\n"
            "- Runtime model components (self.backbone = HFModel(...))\n"
            "- Forward pass delegates (output = self.hf_model(input))\n"
            "- Submodules that process data during training\n"
            "- 'Frozen feature extractors' loaded from HF pretrained checkpoints\n\n"
            "CRITICAL: 'The component is frozen / has no gradient / is just a feature extractor' "
            "is NOT a valid reason to use HF models. Frozen components must STILL be Megatron-native "
            "because: (1) unified checkpoint conversion, (2) future unfreezing, (3) TP memory "
            "distribution even for frozen params, (4) architectural consistency.\n\n"
            f"FILE: {path}\n"
            f"CODE:\n```\n{code_snippet}\n```\n\n"
            "QUESTION: Is this code genuinely Megatron-native (ALL components — including any "
            "frozen backbone/encoder — use Megatron/TE layers), or does it use HuggingFace "
            "models for any runtime component (even frozen ones)?\n\n"
            "Answer ONLY: {\"genuine_native\": true} or {\"genuine_native\": false}"
        )

        try:
            response = self._llm_call_with_timeout(llm_fn, prompt)
            if response is None:
                return False  # On timeout, block conservatively
            response = response.strip()
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return bool(data.get("genuine_native", False))
        except Exception as e:
            logger.warning("LLM judge for Megatron native check failed: %s", e)
        return False  # On failure, block conservatively

    # ── Progress gate ───────────────────────────────────────────────────

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
    _SHELL_WRITE_PATTERN = re.compile(
        r'cat\s*>|tee\s+|>\s*/|echo\s.*>|printf\s.*>|python3?\s.*<<',
        re.IGNORECASE
    )
    _MIN_READS_BEFORE_PORTING_WRITE = 8

    _PORTING_PATH_EARLY_READ_LIMIT = 8

    def _extract_shell_write_target(self, cmd):
        """Extract file path from shell write commands like 'cat > path' or 'tee path'."""
        m = re.search(r'(?:cat|echo|printf)\s*>\s*(\S+)', cmd)
        if m:
            return m.group(1).strip("'\"")
        m = re.search(r'tee\s+(\S+)', cmd)
        if m:
            return m.group(1).strip("'\"")
        m = re.search(r'>\s*(\S+\.(?:py|yaml|yml|json))', cmd)
        if m:
            return m.group(1).strip("'\"")
        return ""

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
        r'layer_spec|ColumnParallelLinear|RowParallelLinear|'
        r'TEDotProductAttention|TransformerLayer|VocabParallelEmbedding|'
        r'tensor.model.parallel|pipeline.model.parallel|'
        r'get_gpt_layer_with_transformer_engine|'
        r'mode\s*b\b|megatron\s*native',
        re.IGNORECASE,
    )
    _MODE_C_SIGNALS = re.compile(
        r'HuggingFace\s*(?:Module|Wrapper)|FSDP2?|hf_module|'
        r'wrap.*existing.*model|from_pretrained.*wrapper|'
        r'MegatronModule.*wrap|wrap.*MegatronModule|'
        r'keep.*(?:HF|huggingface|existing).*model|'
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
            f"2. Forward: REAL data batch (NOT dummy/torch.rand) → finite loss\n"
            f"3. Backward: --train-iters 20, verify loss decreases\n"
            f"4. Distributed: target TP/PP, check no hang\n"
            f"ALL verification uses real data pipeline — dummy data is forbidden.\n"
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
            r'train_gpt|train_qwen|train_llava|train_gr00t|train_.*\.py|forward_step|get_batch'
        ),
        "model_provider_and_builder": re.compile(
            r'model_provider|gpt_builders|_model\.py|_builders\.py'
        ),
        "megatron_layer_spec_system": re.compile(
            r'spec_utils|gpt_layer_specs|layer_specs|TransformerLayerSubmodules'
        ),
        "training_loop": re.compile(
            r'training/training\.py|pretrain|training\.py'
        ),
        "flagscale_custom_models": re.compile(
            r'flagscale/models/|flagscale.*megatron.*model|megatron_native/'
        ),
        "parallelism_system": re.compile(
            r'parallel_state|tensor_parallel|pipeline_parallel|parallel.*layers'
        ),
        "te_attention_system": re.compile(
            r'transformer_engine|extensions.*transformer'
        ),
    }
    _MIN_PIPELINE_DIMENSIONS = 5

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
        if tool_name in ("write_file", "edit_file"):
            target = arguments.get("path", "") or arguments.get("file_path", "")
        elif tool_name == "shell":
            cmd = arguments.get("command", "")
            if not self._SHELL_WRITE_PATTERN.search(cmd):
                return ""
            target = self._extract_shell_write_target(cmd)
            if not target:
                return ""
        else:
            return ""
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""

        # If knowledge already persisted, skip Phase 1 (reading check)
        if self._pipeline_knowledge_persisted:
            # Phase 3 only
            if not self._pipeline_knowledge_confirmed:
                if not hasattr(self, '_pipeline_confirm_block_count'):
                    self._pipeline_confirm_block_count = 0
                self._pipeline_confirm_block_count += 1
                if self._pipeline_confirm_block_count >= 3:
                    self._pipeline_knowledge_confirmed = True
                    return ""
                return (
                    "\n\n[KNOWLEDGE CONFIRMATION GATE] You've read the code and persisted knowledge. "
                    "Now CONFIRM: for THIS specific porting task, is your knowledge sufficient?\n\n"
                    "You MUST include in your response ONE of these markers:\n"
                    "[PIPELINE_KNOWLEDGE_CONFIRMED: YES]\n"
                )
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
            # Pick the first missing dimension as the concrete next action
            first_missing = sorted(missing)[0]
            first_action = dim_descs[first_missing]
            return (
                f"\n\n[PIPELINE COMPREHENSION GATE] You're writing porting code but haven't "
                f"studied the training pipeline deeply enough. Covered {len(covered)}/{self._MIN_PIPELINE_DIMENSIONS} "
                f"required dimensions.\n\n"
                f"Missing knowledge:\n{missing_list}\n\n"
                f"▶ YOUR NEXT ACTION: Use read_file to read the files listed above. "
                f"Start with: {first_action}\n\n"
                f"Even if you believe you already understand the pipeline from prior reading, "
                f"this gate requires you to actually open and read these specific files in this session. "
                f"The gate tracks which files you've read — it will unblock automatically once you've covered enough dimensions.\n\n"
                f"Do NOT attempt to write code again until this gate clears."
            )

        # Phase 2: Check knowledge persistence
        if not self._pipeline_knowledge_persisted:
            return (
                "\n\n[KNOWLEDGE PERSISTENCE GATE] You've read the pipeline code — good. "
                "Now PERSIST your understanding before writing.\n\n"
                "Call memory_write(key='megatron_pipeline_knowledge', content='...') with a "
                "structured summary covering:\n"
                "- How data flows (get_batch → model input) — THIS IS CRITICAL, not optional\n"
                "- How models are constructed (model_provider → layer_spec → TE layers)\n"
                "- The layer_spec system: ModuleSpec, TransformerLayerSubmodules, how FlagScale extends it\n"
                "- Parallelism: initialize_model_parallel, TP (ColumnParallelLinear), PP (schedules), CP/EP/SP\n"
                "- How to add a new model (what files to create, what to register)\n\n"
                "IMPORTANT: Data pipeline integration is EQUALLY important as model adaptation. "
                "Your summary MUST cover how get_batch provides data to the model, including "
                "parallelism-aware data distribution (broadcast_data, pre_process/post_process).\n\n"
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

    # ── Data→Model interface contract gate ─────────────────────────────

    _MODEL_CODE_PATHS = re.compile(
        r'(model|forward|backbone|head|encoder|decoder|transformer|attention|mlp).*\.py$'
    )
    _DATA_MODEL_INTERFACE_MEMORY_KEY = "data_model_interface"

    def _check_data_model_interface_gate(self, tool_name, arguments):
        """HARD BLOCK: Require data→model interface documentation before writing model code.

        The LLM must explicitly document:
        1. What the data pipeline outputs (keys, shapes, dtypes)
        2. What the model's forward() expects as input
        3. How data pipeline output maps to model input

        Without this, the model's forward signature is guesswork, and the entire
        model will need to be rewritten when real data is connected.
        """
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file", "shell"):
            return ""

        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "") or arguments.get("new_string", "")
        else:
            cmd = arguments.get("command", "")
            if not self._SHELL_WRITE_PATTERN.search(cmd):
                return ""
            path = self._extract_shell_write_target(cmd) or ""
            content = cmd

        if not path:
            return ""

        # Only trigger for model implementation files
        is_model_code = (
            self._MODEL_CODE_PATHS.search(path) and
            any(kw in content for kw in (
                "def forward(", "def __init__(", "class ", "MegatronModule",
            ))
        )
        if not is_model_code:
            return ""

        # Check if data→model interface has been documented in memory (LLM-based judgment)
        entries = self.session_memory.list_entries()
        has_interface_doc = self._llm_check_memory_has_knowledge(
            entries,
            "data-to-model interface documentation: what the data pipeline outputs, "
            "what the model's forward() expects as input, and how they map together"
        )

        if has_interface_doc:
            return ""

        # Anti-loop: if this gate has blocked too many times, auto-pass
        if not hasattr(self, '_data_interface_gate_blocks'):
            self._data_interface_gate_blocks = 0
        self._data_interface_gate_blocks += 1
        if self._data_interface_gate_blocks > 5:
            logger.warning("data_model_interface gate: auto-passing after %d blocks", self._data_interface_gate_blocks)
            return ""

        # Quick heuristic: if the code being written already shows data interface awareness, pass
        content_has_interface_awareness = (
            "get_batch" in content and
            any(kw in content for kw in (
                "input_ids", "attention_mask", "labels", "pixel_values",
                "tokenizer", "data_path", "batch[",
            ))
        )
        if content_has_interface_awareness:
            return ""

        return (
            "[DATA→MODEL INTERFACE GATE — BLOCKED]\n\n"
            "You are writing model code WITHOUT documenting the data→model interface contract.\n\n"
            "This is the #1 cause of porting rework: the model's forward() signature is designed "
            "in isolation, then when real data is connected, everything needs to be rewritten "
            "because the model expects different input keys/shapes/dtypes than what the data "
            "pipeline actually produces.\n\n"
            "BEFORE writing model code, you MUST document the data→model interface:\n\n"
            "1. **Data pipeline output**: What does get_batch / the dataloader produce?\n"
            "   - Exact dict keys (e.g., 'input_ids', 'attention_mask', 'pixel_values')\n"
            "   - Tensor shapes for each key (e.g., [B, seq_len], [B, C, H, W])\n"
            "   - Dtypes (int64 for tokens, float32/bfloat16 for images)\n\n"
            "2. **Model forward() input**: What does the model expect?\n"
            "   - Parameter names in forward() signature\n"
            "   - Expected shapes and dtypes for each parameter\n"
            "   - Which parameters are optional vs required\n\n"
            "3. **Mapping**: How does data output → model input?\n"
            "   - Key renaming (e.g., 'pixel_values' → 'images')\n"
            "   - Shape transforms (e.g., flatten, pad, reshape)\n"
            "   - Any preprocessing between get_batch and model.forward\n\n"
            "4. **Parallelism contract** (MANDATORY — without this, Megatron integration FAILS):\n"
            "   - TP: How is input broadcast to all TP ranks? (broadcast_data)\n"
            "   - PP: Which inputs go to first stage vs last stage? (pre_process/post_process)\n"
            "   - DP: How are micro-batches distributed? (sampler)\n"
            "   - CP/EP/SP: Any special handling for context/expert/sequence parallelism?\n"
            "   A data pipeline without parallelism awareness is NOT a valid Megatron data pipeline.\n\n"
            "▶ YOUR NEXT ACTION:\n"
            "1. Read the source model's forward() to see what inputs it expects\n"
            "2. Read the source training script's data loading to see what get_batch produces\n"
            "3. Read an existing Megatron train_*.py to see how parallelism is handled in get_batch\n"
            "4. Save the interface contract (including parallelism plan) to memory:\n"
            "   memory_write(key='data_model_interface', content='...')\n\n"
            "This contract is your SINGLE SOURCE OF TRUTH for both model and data implementation. "
            "Design them together, not separately.\n\n"
            "🚫 Do NOT use dummy/synthetic data (torch.rand/zeros) to 'test' the model. "
            "ALL verification must use real data through the real pipeline."
        )

    # ── Component mapping gate ───────────────────────────────────────────

    def _check_component_mapping_gate(self, tool_name, arguments):
        """HARD BLOCK: Require Megatron component inventory + per-component mapping before writing model code.

        Enforces first-principles thinking: start from what Megatron provides,
        then map each source component to its Megatron equivalent.
        'Too complex' or 'too much work' are NOT valid reasons to skip Megatron primitives.
        Only genuine technical impossibility justifies using vanilla torch.
        """
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""

        path = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")
        if not path or not content:
            return ""
        if not self._MODEL_CODE_PATHS.search(path):
            return ""

        # Only trigger for actual implementation code
        has_impl = any(kw in content for kw in (
            "def forward(", "def __init__(", "class ", "MegatronModule",
        ))
        if not has_impl:
            return ""

        # Check if component mapping already validated this session
        if getattr(self, '_component_mapping_validated', False):
            return ""

        # Check memory for component mapping via LLM judge
        entries = self.session_memory.list_entries()
        if not entries:
            return self._component_mapping_block_message()

        has_mapping = self._llm_judge_component_mapping_quality(entries)
        if has_mapping:
            self._component_mapping_validated = True
            return ""

        # Anti-loop: auto-pass after too many blocks
        if not hasattr(self, '_component_mapping_blocks'):
            self._component_mapping_blocks = 0
        self._component_mapping_blocks += 1
        if self._component_mapping_blocks > 6:
            logger.warning("component_mapping gate: auto-passing after %d blocks", self._component_mapping_blocks)
            self._component_mapping_validated = True
            return ""

        return self._component_mapping_block_message()

    def _component_mapping_block_message(self):
        return (
            "[COMPONENT MAPPING GATE — BLOCKED]\n\n"
            "Before writing model code, you MUST document a Megatron component mapping.\n\n"
            "Step 1: SCAN available Megatron/TE components (read the actual files):\n"
            "  Top-level models:\n"
            "    - GPTModel, LLaVAModel (megatron/core/models/)\n"
            "    - CLIPViTModel (megatron/core/models/vision/)\n"
            "    - TransformerBlock (megatron/core/transformer/transformer_block.py)\n"
            "  Mid-level layers:\n"
            "    - TransformerLayer (megatron/core/transformer/transformer_layer.py)\n"
            "    - SelfAttention, CrossAttention (megatron/core/transformer/attention.py)\n"
            "    - MLP with gated variants (megatron/core/transformer/mlp.py)\n"
            "    - MoE layer (megatron/core/transformer/moe/)\n"
            "  TE (TransformerEngine) layers:\n"
            "    - TEDotProductAttention (flash/fused backends, CP integration)\n"
            "    - TE ColumnParallelLinear / RowParallelLinear (FP8 capable)\n"
            "    - FusedLayerNorm, FusedRMSNorm\n"
            "  Primitives:\n"
            "    - ColumnParallelLinear, RowParallelLinear (megatron/core/tensor_parallel/layers.py)\n"
            "    - VocabParallelEmbedding\n"
            "    - RotaryEmbedding (megatron/core/models/common/embeddings/)\n"
            "  FlagScale custom:\n"
            "    - Check flagscale/models/megatron/ for existing implementations\n\n"
            "Step 2: MAP each source model component → Megatron target:\n"
            "  Format per component:\n"
            "    'SourceClass.layer_name → MegatronEquivalent (reason)'\n"
            "  OR:\n"
            "    'SourceClass.layer_name → vanilla torch (TECHNICAL reason: ...)'\n\n"
            "  INVALID reasons for vanilla torch:\n"
            "    ✗ 'too complex', 'too much work', 'will add later'\n"
            "    ✗ 'other models in the repo don't do this'\n"
            "    ✗ 'keep as-is for initial porting'\n"
            "    ✗ 'frozen / no gradient / feature extractor — no need for native'\n"
            "    ✗ 'no TP benefit for frozen module'\n"
            "    ✗ 'backbone is frozen so use HF directly'\n"
            "  VALID reasons:\n"
            "    ✓ 'per-joint MLP with variable input dims — ColumnParallelLinear requires fixed dims'\n"
            "    ✓ 'adaptive norm injection into cross-attention — no TE equivalent for this pattern'\n"
            "    ✓ 'flow-matching noise schedule — pure math, no neural network layer involved'\n\n"
            "Step 3: Save to memory:\n"
            "  memory_write(key='megatron_component_mapping', content='...')\n\n"
            "▶ YOUR NEXT ACTION:\n"
            "1. Read megatron/core/transformer/ files to understand available components\n"
            "2. Read the source model's __init__() to list all components\n"
            "3. Create the mapping table and save to memory\n\n"
            "The mapping is your CONTRACT — implementation must follow it."
        )

    def _llm_judge_component_mapping_quality(self, entries) -> bool:
        """Use LLM to validate that the component mapping is genuine and complete."""
        llm_fn = getattr(self.session_memory, '_llm_fn', None)
        if not llm_fn:
            # No LLM available — fall back to keyword heuristic
            return self._component_mapping_keyword_check(entries)

        # Build memory summary
        summaries = []
        for e in entries[:25]:
            key = e.get("key", "?")
            content = (e.get("content") or "")[:400]
            if any(kw in key.lower() or kw in content.lower() for kw in (
                "component", "mapping", "inventory", "megatron_component",
                "primitive", "migration", "blueprint",
            )):
                summaries.append(f"- [{key}]: {content}")

        if not summaries:
            return False

        prompt = (
            "You are validating a Megatron component mapping for model porting.\n\n"
            "STRICT RULES:\n"
            "- Every source model component (attention, MLP, norm, embedding, encoder, decoder) "
            "MUST be mapped to a Megatron/TE equivalent OR have a TECHNICAL justification.\n"
            "- ALL components must be Megatron-native — including frozen/non-trainable ones.\n"
            "- Valid technical justifications: 'no Megatron equivalent exists for X because Y', "
            "'TE does not support this attention pattern', "
            "'component has non-standard topology that cannot be expressed as TransformerLayer'\n"
            "- INVALID justifications (MUST reject): 'too complex', 'too much work', "
            "'keep as-is for now', 'will add later', 'not worth the effort', "
            "'other models don't do this', 'pragmatic approach', 'initial porting', "
            "'frozen so no need for native', 'no gradient so keep HF', "
            "'just a feature extractor', 'no TP benefit for frozen module', "
            "'backbone is frozen so use existing HF model'\n"
            "- CRITICAL: 'frozen/no_grad/feature_extractor' is NEVER a valid reason to skip "
            "Megatron-native implementation. Frozen components still need: unified checkpoint "
            "conversion, future unfreezing support, TP memory distribution, architectural consistency.\n"
            "- TP support is per-component (some may not need it), but even without TP, "
            "the component must use Megatron primitives (not HF classes).\n"
            "- The mapping must show awareness of Megatron's component hierarchy:\n"
            "  Top-level: GPTModel, LLaVAModel, TransformerBlock\n"
            "  Mid-level: TransformerLayer, SelfAttention, CrossAttention, MLP\n"
            "  Primitives: ColumnParallelLinear, RowParallelLinear, TEDotProductAttention, "
            "VocabParallelEmbedding, FusedLayerNorm/RMSNorm, RotaryEmbedding\n"
            "- A mapping that says 'wrap HF model in MegatronModule' is NOT valid.\n"
            "- A mapping that says 'keep backbone as-is' without technical justification is NOT valid.\n"
            "- A mapping that says 'backbone is frozen → use HF/vanilla torch' is NOT valid.\n\n"
            "MEMORY ENTRIES:\n" + "\n".join(summaries) + "\n\n"
            "Does this mapping meet ALL the requirements above?\n"
            "Answer ONLY: {\"valid\": true} or {\"valid\": false, \"reason\": \"...\"}"
        )

        try:
            response = self._llm_call_with_timeout(llm_fn, prompt)
            if response is None:
                logger.warning("LLM judge for component mapping timed out, falling back to heuristic")
                return self._component_mapping_keyword_check(entries)
            response = response.strip()
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                result = bool(data.get("valid", False))
                if not result:
                    reason = data.get("reason", "unknown")
                    logger.info("Component mapping rejected by LLM judge: %s", reason)
                return result
        except Exception as e:
            logger.warning("LLM judge for component mapping failed: %s", e)

        # On failure, fall back to keyword heuristic
        return self._component_mapping_keyword_check(entries)

    def _component_mapping_keyword_check(self, entries) -> bool:
        """Fallback heuristic: check if memory has component mapping keywords."""
        all_content = " ".join(
            (e.get("content") or "") + " " + (e.get("key") or "")
            for e in entries
        ).lower()

        # Must mention Megatron component hierarchy
        hierarchy_keywords = [
            "transformerlayer", "transformerblock", "columnparallellinear",
            "rowparallellinear", "tedotproductattention",
        ]
        hierarchy_count = sum(1 for kw in hierarchy_keywords if kw in all_content)
        if hierarchy_count < 3:
            return False

        # Must have mapping language (→ or "maps to" or "equivalent")
        mapping_signals = ["→", "maps to", "equivalent", "replace with", "rebuild using"]
        has_mapping = any(sig in all_content for sig in mapping_signals)
        if not has_mapping:
            return False

        # Must NOT have lazy justifications as the primary approach
        lazy_signals = ["keep as-is", "keep as is", "wrap", "wrapper", "pragmatic",
                        "frozen so", "frozen →", "frozen->", "no gradient",
                        "feature extractor", "no need for native"]
        lazy_count = sum(1 for sig in lazy_signals if sig in all_content)
        if lazy_count > hierarchy_count:
            return False

        return True

    # ── Migration blueprint gate ──────────────────────────────────────────

    def _check_migration_blueprint_gate(self, tool_name, arguments):
        """HARD BLOCK: Mode 2 (Megatron Native) requires a migration blueprint before writing code.

        The LLM must document a complete migration blueprint that maps:
        1. Source forward logic → Megatron Native forward (component-by-component)
        2. Source data pipeline → Megatron get_batch design
        3. Source optimizer/scheduler → Megatron training config

        This prevents the #1 failure mode: LLM reads source code, then immediately
        tries to import/wrap/inherit instead of reimplementing with Megatron primitives.
        By forcing the blueprint first, the LLM commits to a rewrite plan before touching code.
        """
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("write_file", "edit_file", "shell"):
            return ""

        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "") or arguments.get("new_string", "")
        else:
            cmd = arguments.get("command", "")
            if not self._SHELL_WRITE_PATTERN.search(cmd):
                return ""
            path = self._extract_shell_write_target(cmd) or ""
            content = cmd

        if not path:
            return ""

        # Only trigger for model/train implementation files
        is_impl_code = (
            self._MODEL_CODE_PATHS.search(path) or
            re.search(r'(pretrain_|train_).*\.py$', path)
        )
        if not is_impl_code:
            return ""

        # Must contain actual implementation (not just reading/analysis)
        has_impl = any(kw in content for kw in (
            "def forward(", "def __init__(", "class ", "def get_batch",
            "def model_provider", "def forward_step",
        ))
        if not has_impl:
            return ""

        # Check if migration blueprint exists in memory (LLM-based)
        entries = self.session_memory.list_entries()
        has_blueprint = self._llm_check_memory_has_knowledge(
            entries,
            "migration blueprint that references a component-by-component mapping with specific "
            "Megatron primitives for each source component (not just 'use MegatronModule wrapper' "
            "or 'keep as-is' — must have concrete per-component decisions like "
            "'attention → TEDotProductAttention', 'MLP → ColumnParallelLinear + RowParallelLinear')"
        )

        if has_blueprint:
            return ""

        # Check if content itself demonstrates blueprint awareness
        # (mentions specific Megatron primitives for specific source components)
        has_mapping_awareness = (
            bool(self._MEGATRON_NATIVE_INDICATORS.search(content)) and
            any(kw in content for kw in (
                "ColumnParallelLinear", "RowParallelLinear",
                "TEDotProductAttention", "TransformerLayer",
            ))
        )
        if has_mapping_awareness:
            return ""

        detail = (
            "[MIGRATION BLUEPRINT GATE — BLOCKED]\n\n"
            "Mode 2 (Megatron Native) implementation BLOCKED. No migration blueprint found.\n\n"
            "Before writing ANY model or training code, you MUST document a migration blueprint "
            "that maps source implementation → Megatron Native implementation. This prevents "
            "the common failure of reading source code and then trying to import/wrap it "
            "instead of reimplementing with Megatron primitives.\n\n"
            "Your blueprint must cover:\n\n"
            "**0. Megatron-Core Survey** — what components already exist?\n"
            "   Check megatron/core/models/, megatron/core/transformer/, flagscale/models/megatron/\n"
            "   Priority: Megatron high-level model > TE layer > Megatron primitive > compose > torch\n\n"
            "**1. Forward Logic Mapping** (source component → Megatron target):\n"
            "   For EACH component: source class → Megatron equivalent (or 'compose' / 'custom torch')\n"
            "   Frozen components are still part of the model — use Megatron primitives + requires_grad=False\n"
            "   'Frozen' is NEVER a reason to skip native implementation. Reasons:\n"
            "   - Unified checkpoint conversion (one converter for the whole model)\n"
            "   - Future unfreezing (architecture must support it without rewrite)\n"
            "   - TP memory distribution (even frozen params can be sharded)\n"
            "   - Architectural consistency (one top-level MegatronModule owns everything)\n"
            "   TP support is per-component — assess whether each component benefits from TP.\n"
            "   Even without TP, use Megatron primitives (ColumnParallelLinear with gather_output=True).\n"
            "   PREFER TransformerEngine (TE) layers wherever possible:\n"
            "   - Attention → TEDotProductAttention (supports flash/fused backends, CP integration)\n"
            "   - Linear → TE's ColumnParallelLinear/RowParallelLinear (FP8 capable)\n"
            "   - LayerNorm → TE's FusedLayerNorm or RMSNorm\n"
            "   - Full transformer block → TransformerLayer with TE submodules via layer_spec\n"
            "   Only fall back to vanilla torch when TE has no equivalent (e.g., custom gating, per-joint encoders)\n\n"
            "**2. Data Pipeline Mapping** (source preprocessing → Megatron get_batch):\n"
            "   Steps, parallelism distribution, output format (keys, shapes, dtypes)\n\n"
            "**3. Optimizer/Scheduler Mapping** (source training config → Megatron args):\n"
            "   - Source: optimizer type, lr, betas, weight_decay, warmup, scheduler\n"
            "   - Target: corresponding Megatron args (--lr, --adam-beta1/2, --lr-warmup-iters, etc.)\n"
            "   - Grad clipping, loss scaling, precision settings\n\n"
            "Save this blueprint to memory:\n"
            "  memory_write(key='migration_blueprint', content='...')\n\n"
            "▶ YOUR NEXT ACTION (in this order):\n"
            "1. Read the source model's __init__() and forward() to understand the architecture\n"
            "2. SURVEY Megatron-Core's available components — check what already exists:\n"
            "   - megatron/core/models/ (GPTModel, CLIPViTModel, LLaVAModel, etc.)\n"
            "   - megatron/core/transformer/ (TransformerLayer, layer specs)\n"
            "   - megatron/core/extensions/transformer_engine.py (TE integration)\n"
            "   - flagscale/models/megatron/ (existing ported models as reference)\n"
            "   For each source component, ask: does Megatron/TE already have this?\n"
            "3. Read the source training script's data pipeline and optimizer config\n"
            "4. Write the blueprint: source component → TE/Megatron component (or 'compose' / 'custom')\n"
            "5. Save blueprint to memory, THEN start implementing"
        )
        return {
            "name": "migration_blueprint",
            "description": "Mode 2 requires a migration blueprint before writing code",
            "reason": "Writing model code but no blueprint found in memory",
            "detail": detail,
        }

    # ── Megatron Native integrity gate ─────────────────────────────────

    _HF_IMPORT_PATTERNS = re.compile(
        r'from\s+transformers\s+import\s+\w+|'
        r'from\s+transformers\.models\.\w+|'
        r'AutoModel\w*\.from_pretrained|'
        r'(Siglip|Clip|Llama|Qwen|Mistral|Gemma|Phi)\w*(Model|ForCausalLM|VisionModel|Encoder)',
        re.IGNORECASE
    )
    _REUSE_SOURCE_MODEL_PATTERNS = re.compile(
        r'\.from_pretrained\s*\(|'
        r'\.from_config\s*\(|'
        r'AutoModel\w*\s*\.\s*from_|'
        r'# (?:reuse|wrap|import|use)\s+(?:original|source|existing|HF)',
        re.IGNORECASE
    )
    _MEGATRON_NATIVE_INDICATORS = re.compile(
        r'ColumnParallelLinear|RowParallelLinear|TEDotProductAttention|'
        r'TransformerLayer|ModuleSpec|TransformerLayerSubmodules|'
        r'tensor_model_parallel|set_input_tensor|layer_spec|'
        r'VocabParallelEmbedding|FusedLayerNorm|TENorm|'
        r'megatron\.core\.tensor_parallel|megatron\.core\.transformer'
    )

    def _check_megatron_native_integrity_gate(self, tool_name, arguments):
        """HARD BLOCK: Detect Mode 2 (Megatron Native) that actually uses HF models internally.

        If the porting path is mode_b (Megatron Native), ALL model components MUST use
        Megatron's parallelism primitives (ColumnParallelLinear, TEDotProductAttention, etc.).
        This includes frozen/non-trainable components — 'frozen' is a training decision,
        not an architecture decision. Importing HuggingFace models directly defeats the
        entire purpose of Megatron Native porting — it breaks unified checkpoint conversion,
        prevents future unfreezing, and fragments the architecture.
        """
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("write_file", "edit_file", "shell"):
            return ""

        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "") or arguments.get("new_string", "")
        else:
            cmd = arguments.get("command", "")
            if not self._SHELL_WRITE_PATTERN.search(cmd):
                return ""
            path = self._extract_shell_write_target(cmd) or ""
            content = cmd

        if not path:
            return ""

        # Only check model implementation files
        if not self._MODEL_CODE_PATHS.search(path):
            return ""

        # Check if content imports HF models or reuses source VLA models
        has_hf_imports = bool(self._HF_IMPORT_PATTERNS.search(content))
        has_reuse_source = bool(self._REUSE_SOURCE_MODEL_PATTERNS.search(content))
        if not has_hf_imports and not has_reuse_source:
            return ""

        # Check if content ALSO uses Megatron native primitives
        has_megatron_primitives = bool(self._MEGATRON_NATIVE_INDICATORS.search(content))
        if has_megatron_primitives:
            # Mixed code: has both HF imports AND Megatron primitives.
            # Use LLM judge to determine if this is genuine Megatron-native code that
            # references HF for weight conversion/comparison, or a wrapper that sprinkles
            # Megatron symbols to bypass this gate.
            if self._llm_judge_is_genuine_megatron_native(content, path):
                return ""
            # LLM says it's a wrapper disguised with Megatron imports
            return (
                "[MEGATRON NATIVE INTEGRITY GATE — BLOCKED]\n\n"
                "Your code imports HuggingFace models AND has some Megatron symbols, but the "
                "core architecture is still wrapping/delegating to HF models rather than "
                "rebuilding with Megatron primitives.\n\n"
                "⚠️ Adding a few Megatron imports to a wrapper does NOT make it Megatron Native.\n"
                "The model's forward pass must flow through Megatron/TE layers, not through "
                "HF model.forward().\n\n"
                "⚠️ 'The component is frozen / no gradient / feature extractor' is NOT a valid "
                "reason to use HF models. ALL components must be Megatron-native for:\n"
                "- Unified checkpoint conversion (one converter for the whole model)\n"
                "- Future unfreezing (architecture supports it without rewrite)\n"
                "- TP memory distribution (even frozen params can be sharded)\n"
                "- Architectural consistency (one top-level MegatronModule)\n\n"
                "The correct approach:\n"
                "1. READ the source model's forward() to understand the ALGORITHM\n"
                "2. REWRITE using TE/Megatron primitives — the forward logic flows through "
                "TransformerLayer, TEDotProductAttention, ColumnParallelLinear, etc.\n"
                "3. HF imports are ONLY acceptable for weight loading/conversion utilities, "
                "NOT as runtime model components.\n"
                "4. Set requires_grad=False for frozen components AFTER building them natively.\n\n"
                "▶ YOUR NEXT ACTION:\n"
                "Remove the HF model as a runtime component. Rebuild its layers using "
                "Megatron primitives.\n"
                "Reference: `flagscale/models/megatron/qwen2_5_vl/`, `flagscale/models/megatron/qwen3_vl/`"
            )

        violation_type = (
            "uses .from_pretrained() or .from_config() to load existing model classes"
            if has_reuse_source and not has_hf_imports
            else "imports HuggingFace models"
        )

        return (
            "[MEGATRON NATIVE INTEGRITY GATE — BLOCKED]\n\n"
            f"You chose Mode 2 (Megatron Native) but your model code {violation_type} "
            "WITHOUT using any Megatron parallelism primitives.\n\n"
            "⚠️ 'Megatron Native' means REWRITING the model logic using Megatron primitives "
            "and TransformerEngine (TE), NOT importing/wrapping existing model classes.\n\n"
            "⚠️ ALL components must be native — including frozen/non-trainable ones. "
            "'Frozen' is a training config decision (requires_grad=False), not an architecture "
            "decision. A frozen component still needs Megatron primitives for unified checkpoint "
            "conversion, future unfreezing, and TP memory distribution.\n\n"
            "The correct approach:\n"
            "1. READ the source model's forward() to understand the ALGORITHM\n"
            "2. REWRITE using TE/Megatron primitives (prefer TE where available):\n"
            "   - nn.Linear → ColumnParallelLinear / RowParallelLinear (TE, FP8 capable)\n"
            "   - Self-attention → TEDotProductAttention (flash/fused backends)\n"
            "   - Transformer blocks → TransformerLayer with TE submodules via layer_spec\n"
            "   - LayerNorm/RMSNorm → TE FusedLayerNorm / FusedRMSNorm\n"
            "   - Embeddings → VocabParallelEmbedding\n"
            "   Only use vanilla torch when no TE/Megatron equivalent exists.\n"
            "3. The forward() logic (how tensors flow) can be the same\n"
            "4. The IMPLEMENTATION of each component must use TE/Megatron primitives\n"
            "5. For frozen components: build natively, then set requires_grad=False\n\n"
            "▶ YOUR NEXT ACTION:\n"
            "Read the source model's forward() and __init__() to extract the algorithm.\n"
            "Then implement it from scratch using TE/Megatron primitives.\n"
            "Reference: `flagscale/models/megatron/qwen2_5_vl/`, `flagscale/models/megatron/qwen3_vl/`"
        )

    # ── Mode B design integrity gate ──────────────────────────────────

    _WRAPPER_DESIGN_SIGNALS = re.compile(
        r'wrap.*(?:HF|huggingface|existing).*model|'
        r'keep.*(?:HF|huggingface|existing).*(?:model|module)|'
        r'(?:HF|huggingface).*model.*(?:inside|within)|'
        r'(?:DDP|FSDP|DistributedDataParallel).*only|'
        r'data.parallel.only|'
        r'(?:not|don.t|skip).*(?:TP|tensor.parallel|pipeline.parallel)|'
        r'frozen.*(?:skip|keep|use.*HF|use.*existing|no.*need|don.t.*need)|'
        r'(?:no.*gradient|no.*train).*(?:skip|keep|use.*HF|don.t.*need)|'
        r'(?:backbone|encoder|vision).*frozen.*(?:as.is|HF|existing|load.*pretrained)',
        re.IGNORECASE,
    )

    _WRAPPER_CODE_SIGNALS = re.compile(
        r'self\.\w+\s*=\s*\w+(Model|ForCausalLM|VisionModel|Encoder)\s*\(|'
        r'self\.\w+\s*=\s*\w+\.from_pretrained\s*\(|'
        r'self\.\w+\s*=\s*\w+\.from_config\s*\(|'
        r'(?:output|hidden|features)\s*=\s*self\.\w+\s*\(\s*(?:input|x|hidden|pixel)|'
        r'self\.(?:backbone|encoder|vision_model)\s*=\s*None|'
        r'torch\.no_grad.*self\.(?:backbone|encoder|vision)',
        re.IGNORECASE,
    )

    def _check_mode_b_design_integrity_gate(self, tool_name, arguments):
        """HARD BLOCK: detect wrapper code in write_file/edit_file during Mode B porting.

        Blocks model code that instantiates HF model classes as runtime submodules.
        """
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""

        path = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")
        if not path or not content:
            return ""
        if not self._MODEL_CODE_PATHS.search(path):
            return ""
        # Detect code that instantiates HF models as runtime submodules
        has_wrapper_code = bool(self._WRAPPER_CODE_SIGNALS.search(content))
        if not has_wrapper_code:
            return ""
        # Check if it also has real Megatron primitives as the primary architecture
        has_megatron_primitives = bool(self._MEGATRON_NATIVE_INDICATORS.search(content))
        if has_megatron_primitives:
            # Mixed — defer to megatron_native_integrity gate's LLM judge
            return ""
        return (
            "[MODE B DESIGN INTEGRITY GATE — BLOCKED]\n\n"
            "Your model code instantiates HuggingFace model classes as runtime submodules. "
            "This is a wrapper approach (Mode C), NOT Megatron Native (Mode B).\n\n"
            "In Mode B, the model's __init__ must construct ALL layers from Megatron/TE primitives:\n"
            "- TransformerLayer (with layer_spec)\n"
            "- TEDotProductAttention\n"
            "- ColumnParallelLinear / RowParallelLinear\n"
            "- VocabParallelEmbedding\n"
            "- FusedLayerNorm / TENorm\n\n"
            "This applies to ALL components — including frozen/non-trainable ones.\n"
            "'Frozen' is a training decision (requires_grad=False), not an architecture decision.\n"
            "A frozen component must still be Megatron-native for unified checkpoint conversion, "
            "future unfreezing, and architectural consistency.\n\n"
            "HF model classes (SiglipVisionModel, Qwen2ForCausalLM, etc.) must NOT appear "
            "as self.xxx = HFModel(...) in the Megatron-native implementation.\n\n"
            "▶ YOUR NEXT ACTION:\n"
            "Read the source HF model's __init__() and forward() to extract the algorithm, "
            "then rebuild using Megatron/TE primitives. For frozen components, build natively "
            "then set requires_grad=False.\n"
            "Reference: `flagscale/models/megatron/qwen2_5_vl/`, `flagscale/models/megatron/qwen3_vl/`"
        )

    def _check_mode_b_design_integrity_soft_gate(self, tool_name, arguments):
        """Soft warning: detect wrapper/DDP-only design language in plans or memory during Mode B."""
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("plan_create", "plan_update", "memory_write"):
            return ""
        if getattr(self, '_mode_b_design_warnings', 0) >= 2:
            return ""

        content = arguments.get("content", "") or arguments.get("plan", "")
        if not content:
            return ""

        if not self._WRAPPER_DESIGN_SIGNALS.search(content):
            return ""

        # Check if it also mentions real Mode B primitives — if so, might be mixed/valid
        has_real_primitives = bool(re.search(
            r'ColumnParallelLinear|RowParallelLinear|TEDotProductAttention|'
            r'TransformerLayer|layer_spec|VocabParallelEmbedding',
            content
        ))
        if has_real_primitives:
            return ""

        if not hasattr(self, '_mode_b_design_warnings'):
            self._mode_b_design_warnings = 0
        self._mode_b_design_warnings += 1

        detail = (
            "\n\n[MODE B DESIGN WARNING] Your plan/design describes a wrapper or DDP-only approach, "
            "but the user confirmed Mode B (Megatron Native).\n\n"
            "Mode B means ALL components — including frozen ones — use Megatron primitives:\n"
            "- Rebuild model layers using Megatron primitives and TransformerEngine (TE)\n"
            "- Prefer TE layers: TEDotProductAttention, ColumnParallelLinear, RowParallelLinear, "
            "TransformerLayer with TE submodules\n"
            "- TP support is per-component (assess each component), but even without TP, "
            "use Megatron primitives (not HF classes)\n"
            "- Use layer_spec system for model construction\n"
            "- Use broadcast_data in get_batch for TP data distribution\n\n"
            "CRITICAL: 'Component is frozen / no gradient / feature extractor' is NOT a valid "
            "reason to use HF models. Frozen components must still be Megatron-native for: "
            "unified checkpoint conversion, future unfreezing, TP memory distribution.\n\n"
            "There is ONE top-level MegatronModule. Every submodule lives inside it. "
            "Freeze/unfreeze is set via requires_grad — it is never an architecture decision.\n\n"
            "Wrapping an existing HF model inside MegatronModule is Mode C, not Mode B.\n"
            "Refer to existing Megatron-native models: flagscale/models/megatron/qwen2_5_vl/, "
            "flagscale/models/megatron/qwen3_vl/"
        )
        return {
            "name": "mode_b_design_integrity",
            "description": "Detects wrapper/DDP-only design in Mode B porting plans",
            "reason": "Plan describes wrapping existing model instead of rebuilding with Megatron primitives",
            "detail": detail,
        }

    # ── Megatron primitives usage gate ─────────────────────────────────

    _VANILLA_TORCH_PATTERNS = re.compile(
        r'nn\.Linear\(|nn\.Embedding\(|nn\.LayerNorm\(|'
        r'F\.scaled_dot_product_attention|'
        r'torch\.nn\.functional\.scaled_dot_product_attention'
    )

    def _check_megatron_primitives_usage_gate(self, tool_name, arguments):
        """Soft warning: Mode 2 code uses ONLY vanilla torch with zero Megatron primitives.

        Priority for Megatron Native implementation:
        1. Megatron has a ready primitive → use it directly
        2. Can be assembled from Megatron primitives → compose them
        3. No Megatron equivalent exists → then use torch

        This gate only fires when the code is ENTIRELY vanilla torch (suggesting
        the LLM didn't even try to use Megatron primitives). Mixed usage is fine —
        some custom ops legitimately need torch when Megatron has no equivalent.

        Fires at most 3 times per session.
        """
        if not self._porting_mode:
            return ""
        if not hasattr(self, '_confirmed_porting_path'):
            return ""
        if self._confirmed_porting_path != "mode_b":
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        if getattr(self, '_primitives_usage_warnings', 0) >= 3:
            return ""

        path = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")

        if not self._MODEL_CODE_PATHS.search(path):
            return ""

        # Count vanilla torch usage — need significant amount to trigger
        vanilla_matches = self._VANILLA_TORCH_PATTERNS.findall(content)
        if len(vanilla_matches) < 5:
            return ""  # Small amount is fine (custom ops, small projections)

        # If ANY Megatron primitives are present, the LLM is trying — don't warn
        has_megatron = bool(self._MEGATRON_NATIVE_INDICATORS.search(content))
        if has_megatron:
            return ""

        if not hasattr(self, '_primitives_usage_warnings'):
            self._primitives_usage_warnings = 0
        self._primitives_usage_warnings += 1

        detail = (
            "\n\n[MEGATRON PRIMITIVES WARNING] Your Mode 2 model code uses "
            f"{len(vanilla_matches)} vanilla torch modules (nn.Linear, nn.Embedding, etc.) "
            "with zero Megatron primitives.\n\n"
            "Priority for Megatron Native (prefer TE wherever possible):\n"
            "  1. Megatron has a high-level model → USE IT (GPTModel, CLIPViTModel, LLaVAModel)\n"
            "  2. TransformerEngine (TE) layer available → USE IT:\n"
            "     - Attention: TEDotProductAttention (flash/fused backends, CP-aware)\n"
            "     - Linear: ColumnParallelLinear / RowParallelLinear (FP8 capable)\n"
            "     - Norm: TE FusedLayerNorm / RMSNorm\n"
            "     - Full block: TransformerLayer with TE submodules via layer_spec\n"
            "  3. Megatron primitive without TE → USE IT (VocabParallelEmbedding, etc.)\n"
            "  4. Can be composed from Megatron/TE primitives → compose them\n"
            "  5. No Megatron/TE equivalent at all → THEN use torch\n\n"
            "TE is NOT mandatory, but strongly preferred — it enables FP8, fused kernels, "
            "and seamless CP/TP integration. Only skip TE when the component has no TE equivalent "
            "(e.g., custom gating, per-joint encoders, specialized loss functions).\n\n"
            "A file with 5+ nn.Linear and zero Megatron/TE imports suggests you're writing "
            "a pure-torch reimplementation instead of using Megatron's high-performance modules.\n"
            "Survey megatron/core/models/ and flagscale/models/megatron/ first."
        )
        return {
            "name": "megatron_primitives_usage",
            "description": "Checks model code for Megatron primitive usage",
            "reason": f"{len(vanilla_matches)} vanilla torch modules with zero Megatron imports",
            "detail": detail,
        }

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

        # Use LLM to check if memory contains data pipeline understanding
        entries = self.session_memory.list_entries()
        has_pipeline_knowledge = self._llm_check_memory_has_knowledge(
            entries,
            "data pipeline understanding: source data format, processing/tokenization steps, "
            "and how data enters the model (get_batch, dataloader, dataset class)"
        )
        if has_pipeline_knowledge:
            self._data_pipeline_understood = True
            return ""

        # Fallback: check file read coverage
        data_reads = 0
        covered = set()
        for path in self._files_read_this_session:
            for cat, pattern in self._DATA_READ_CATEGORIES.items():
                if pattern.search(path):
                    covered.add(cat)
                    data_reads += 1

        if data_reads >= self._MIN_DATA_READS and len(covered) >= self._MIN_DATA_CATEGORIES:
            return ""

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

        # Check if parallelism strategy has been documented (LLM-based)
        entries = self.session_memory.list_entries()
        has_parallelism_doc = self._llm_check_memory_has_knowledge(
            entries,
            "parallelism strategy documentation: tensor parallel (TP), pipeline parallel (PP), "
            "data parallel (DP), expert parallel (EP), context parallel (CP), or sequence parallel (SP) "
            "configuration and how data/model is distributed across ranks"
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
            "⚠️ CRITICAL PRINCIPLE: A data pipeline without parallelism awareness is a FAILED "
            "Megatron integration. There is NO valid Megatron data pipeline that ignores parallelism. "
            "This is not optional — it is the fundamental contract of distributed training.\n\n"
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
            "Data pipeline and parallelism are NOT separable — they are ONE design.\n"
            "If you write get_batch without broadcast_data/pre_process/post_process, "
            "it WILL deadlock or produce wrong results at runtime."
        )

    # ── Train script data pipeline completeness gate ────────────────────

    _TRAIN_SCRIPT_PATH = re.compile(r'(pretrain_|train_).*\.py$')

    def _check_train_script_data_pipeline_gate(self, tool_name, arguments):
        """HARD BLOCK: Ensure train script properly integrates data pipeline.

        When writing train_xxx.py, the LLM often treats it as purely a model adapter
        (model_provider + forward_step) and either stubs get_batch or omits data
        pipeline considerations entirely. This gate enforces that data pipeline
        integration is treated as equally important as model adaptation.
        """
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file", "shell"):
            return ""

        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "") or arguments.get("new_string", "")
        else:
            cmd = arguments.get("command", "")
            if not self._SHELL_WRITE_PATTERN.search(cmd):
                return ""
            path = self._extract_shell_write_target(cmd) or ""
            content = cmd

        if not self._TRAIN_SCRIPT_PATH.search(path):
            return ""

        has_model_provider = "model_provider" in content or "def forward_step" in content
        if not has_model_provider:
            return ""

        has_get_batch = "def get_batch" in content or "get_batch" in content
        has_real_data_logic = any(kw in content for kw in (
            "tokenizer", "input_ids", "attention_mask", "labels",
            "data_path", "dataset", "DataLoader", "data_config",
            "image", "pixel_values", "multimodal",
        ))
        has_parallelism_in_data = any(kw in content for kw in (
            "broadcast_data", "get_data_parallel",
            "pre_process", "post_process",
            "tensor_model_parallel", "pipeline_model_parallel",
        ))
        uses_dummy_data = any(kw in content for kw in (
            "torch.rand(", "torch.randn(", "torch.zeros(",
            "torch.ones(", "dummy", "fake_data", "random_tensor",
            "placeholder", "mock_batch", "synthetic",
        ))

        issues = []
        if uses_dummy_data:
            issues.append(
                "DUMMY/FAKE DATA DETECTED — using torch.rand/zeros/ones or synthetic data "
                "is STRICTLY FORBIDDEN during porting. You MUST use real data pipeline integration"
            )
        if not has_get_batch:
            issues.append("Missing get_batch() — data pipeline entry point is absent")
        elif not has_real_data_logic:
            issues.append(
                "get_batch() appears to be a stub or placeholder — "
                "no real data loading logic (tokenizer, input_ids, data_path, etc.)"
            )
        if not has_parallelism_in_data:
            issues.append(
                "No parallelism handling in data flow — "
                "missing broadcast_data/pre_process/post_process/parallel group references"
            )

        if not issues:
            return ""

        return (
            "[TRAIN SCRIPT DATA PIPELINE GATE — BLOCKED]\n\n"
            "Your train script is incomplete: data pipeline integration is EQUALLY important "
            "as model adaptation. A train script is NOT just model_provider + forward_step.\n\n"
            "Issues found:\n" +
            "\n".join(f"  - {i}" for i in issues) + "\n\n"
            "🚫 DUMMY DATA IS STRICTLY FORBIDDEN:\n"
            "   Do NOT use torch.rand/randn/zeros/ones as input data for verification.\n"
            "   Do NOT create synthetic/fake/placeholder batches.\n"
            "   ALL verification must use REAL data pipeline with actual data loading.\n\n"
            "A complete train script MUST include:\n"
            "1. get_batch() — REAL data loading, tokenization, and formatting\n"
            "   - Connect to actual dataset (data_path from config)\n"
            "   - Real tokenization and preprocessing\n"
            "   - Correct keys/shapes/dtypes for the model's forward()\n"
            "2. Parallelism-aware data distribution:\n"
            "   - broadcast_data() for TP rank consistency\n"
            "   - pre_process/post_process guards for PP stages\n"
            "   - Correct micro-batch handling for DP\n"
            "3. Data format alignment with model expectations:\n"
            "   - Input tensor names matching model forward() signature\n"
            "   - Proper padding, masking, and label construction\n\n"
            "▶ YOUR NEXT ACTION: Before writing this train script, read an existing "
            "train_*.py (e.g., train_gpt.py or a VL train script) to understand how "
            "get_batch integrates with the training loop. Then implement get_batch with "
            "REAL data logic and parallelism support — not a stub, not dummy data.\n\n"
            "Data pipeline is NOT something to 'add later after checkpoint works'. "
            "A train script without proper data integration WILL fail at runtime."
        )

    # ── No dummy data gate ─────────────────────────────────────────────

    _DUMMY_DATA_PATTERNS = re.compile(
        r'torch\.(rand|randn|zeros|ones|empty)\s*\(|'
        r'dummy[_\s]*(data|batch|input|tensor)|'
        r'fake[_\s]*(data|batch|input|tensor)|'
        r'random[_\s]*(input|batch|tensor)|'
        r'synthetic[_\s]*(data|batch)|'
        r'placeholder[_\s]*(data|batch)',
        re.IGNORECASE
    )

    def _check_no_dummy_data_gate(self, tool_name, arguments):
        """HARD BLOCK: Forbid dummy/synthetic data for model verification during porting.

        During model porting, ALL verification must use real data pipeline.
        Using torch.rand/zeros/ones as model input hides data integration bugs
        that only surface later, wasting time.
        """
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file", "shell"):
            return ""

        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "") or arguments.get("new_string", "")
        else:
            cmd = arguments.get("command", "")
            content = cmd
            path = ""

        if not content:
            return ""

        # Only check files/commands that involve model invocation for verification
        # (not model definition files that happen to define forward())
        is_model_invocation = any(kw in content for kw in (
            "model(", "model.forward(", "output = model",
            "verify", "test_forward", "sanity_check", "inference",
        ))
        # Model definition files (class with def forward) are NOT verification
        is_model_definition = "class " in content and "def forward(" in content
        if not is_model_invocation or is_model_definition:
            return ""

        if not self._DUMMY_DATA_PATTERNS.search(content):
            return ""

        return (
            "[NO DUMMY DATA GATE — BLOCKED]\n\n"
            "🚫 DUMMY/SYNTHETIC DATA IS STRICTLY FORBIDDEN for model verification during porting.\n\n"
            "You are attempting to verify the model using torch.rand/zeros/ones or synthetic data. "
            "This is NOT allowed because:\n"
            "1. Dummy data hides data pipeline integration bugs\n"
            "2. Shape/dtype mismatches between real data and model only surface with real data\n"
            "3. You will have to redo this verification anyway once real data is connected\n"
            "4. It creates a false sense of progress — 'model works' with dummy data means nothing\n\n"
            "▶ YOUR NEXT ACTION: Implement the REAL data pipeline first:\n"
            "1. Read the existing train_*.py to understand get_batch interface\n"
            "2. Implement get_batch with real data loading (tokenizer, data_path, etc.)\n"
            "3. Use the real get_batch output to verify the model\n\n"
            "ALL verification must flow through the real data pipeline. "
            "There are NO exceptions — not for 'quick checks', not for 'just testing shapes', "
            "not for 'I'll add real data later'. Real data integration comes FIRST."
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

    # ── Parallelism assessment gate ──────────────────────────────────────

    _TRAINING_CONFIG_PATTERNS = re.compile(
        r'conf/.*\.(yaml|yml)$|train.*config.*\.(yaml|yml)$'
    )

    def _check_parallelism_assessment_gate(self, tool_name, arguments):
        """Soft warning: training config written without parallelism feasibility assessment.

        When writing training YAML configs, the agent should have already assessed
        which parallelism strategies are feasible for this model based on its actual
        dimensions (hidden_size, num_heads, num_layers, seq_len). Without this,
        the config may specify strategies that don't work (e.g., TP=8 when num_kv_heads=4)
        or miss strategies that would be beneficial.

        Fires at most twice per session.
        """
        if not self._porting_mode and not self._data_prep_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        if getattr(self, '_parallelism_assessment_warnings', 0) >= 2:
            return ""

        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._TRAINING_CONFIG_PATTERNS.search(target):
            return ""

        content = arguments.get("content", "") or arguments.get("new_string", "")
        # Only trigger if config mentions parallelism settings
        mentions_parallelism = any(
            kw in content
            for kw in ("tensor_model_parallel", "pipeline_model_parallel",
                       "context_parallel", "expert_model_parallel",
                       "sequence_parallel", "data_parallel",
                       "tp_size", "pp_size", "cp_size", "ep_size")
        )
        if not mentions_parallelism:
            return ""

        # Check if parallelism assessment has been documented in memory (LLM-based)
        entries = self.session_memory.list_entries()
        has_assessment = self._llm_check_memory_has_knowledge(
            entries,
            "parallelism feasibility assessment: whether model dimensions (num_heads, hidden_size) "
            "are divisible by TP/PP/EP degrees, and recommended parallelism strategy"
        )

        if has_assessment:
            return ""

        # Check if content itself shows assessment awareness
        content_lower = content.lower()
        has_inline_assessment = (
            "num_heads" in content_lower and
            ("divisible" in content_lower or "feasib" in content_lower)
        )
        if has_inline_assessment:
            return ""

        if not hasattr(self, '_parallelism_assessment_warnings'):
            self._parallelism_assessment_warnings = 0
        self._parallelism_assessment_warnings += 1

        return (
            "\n\n[PARALLELISM ASSESSMENT WARNING] You are writing a training config with "
            "parallelism settings but have not documented a parallelism feasibility assessment.\n\n"
            "Before choosing TP/PP/CP/SP/EP values, assess feasibility based on actual model dimensions:\n"
            "- TP: num_heads must be divisible by TP degree (also num_kv_heads for GQA)\n"
            "- PP: stages must be roughly balanced in params (max/min < 2x)\n"
            "- SP: only useful with TP, and only if seq_len × hidden is large\n"
            "- CP: only useful if seq_len > 4096\n"
            "- EP: only if model has MoE layers\n"
            "- If model fits on single GPU (params × 18 < GPU_mem × 0.8), TP/PP add overhead without benefit\n\n"
            "Save assessment to memory (memory_write key='parallelism_assessment') with:\n"
            "1. Model dimensions (hidden, heads, kv_heads, layers, seq_len, total params)\n"
            "2. Per-strategy feasibility verdict\n"
            "3. Recommended combination for target GPU count\n"
            "4. Strategies explicitly NOT recommended and why"
        )

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
                    next_phase = phase_order[current_idx + 1] if current_idx + 1 < len(phase_order) else "unknown"
                    return (
                        f"[PHASE ORDERING GATE — BLOCKED]\n\n"
                        f"Current phase: {current_phase}\n"
                        f"Checkpoint conversion is NOT allowed yet.\n\n"
                        f"Required order:\n"
                        f"1. Complete model structure implementation\n"
                        f"2. Verify structure completeness (all components present)\n"
                        f"3. Implement data pipeline with parallelism support ← EQUALLY IMPORTANT as model\n"
                        f"4. THEN convert checkpoint\n\n"
                        f"⚠ Data pipeline integration (get_batch, data loading, parallelism-aware distribution) "
                        f"is NOT optional and NOT something to 'add later'. It must be implemented BEFORE "
                        f"checkpoint conversion because runtime verification requires real data flow.\n\n"
                        f"▶ YOUR NEXT ACTION: Complete the '{next_phase}' phase first. "
                        f"Even if you believe the model structure is ready, you must explicitly "
                        f"advance through each phase in order. "
                        f"Use plan_update to mark the current phase done when its requirements are met.\n\n"
                        f"Do NOT attempt checkpoint conversion again until you reach that phase."
                    )

        # Detect data pipeline implementation (write_file or shell write)
        if tool_name in ("write_file", "shell"):
            if tool_name == "write_file":
                path = arguments.get("path", "")
                content = arguments.get("content", "")
            else:
                cmd = arguments.get("command", "")
                if not self._SHELL_WRITE_PATTERN.search(cmd):
                    path, content = "", ""
                else:
                    path = self._extract_shell_write_target(cmd) or ""
                    content = cmd
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
                    f"▶ YOUR NEXT ACTION: Finish the current phase ('{current_phase}') first. "
                    f"If you're in 'structure_implementation', verify all model components are present. "
                    f"If you're in 'structure_verification', run import tests and shape checks.\n\n"
                    f"Even if you believe the model is complete, the gate requires explicit verification "
                    f"before moving to data pipeline. Do NOT attempt data pipeline code again until the phase advances."
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
                        f"- Checkpoint conversion\n\n"
                        f"▶ YOUR NEXT ACTION: Complete the '{current_phase}' phase. "
                        f"Do NOT launch training until all prior phases are done."
                    )

        return ""

    # ── Model completeness check gate ────────────────────────────────────

    def _check_model_completeness_gate(self, tool_name, arguments):
        """Soft warning: model class written without all enumerated components.

        When writing a top-level model class (__init__ with self.xxx = ...),
        checks whether the documented structure enumeration in memory contains
        components that are NOT present in the code being written. This catches
        the common mistake of implementing a model shell that's missing submodules.

        Fires at most 3 times per session.
        """
        if not self._porting_mode:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        if getattr(self, '_model_completeness_warnings', 0) >= 3:
            return ""

        path = arguments.get("path", "") or arguments.get("file_path", "")
        content = arguments.get("content", "") or arguments.get("new_string", "")

        # Only trigger for model implementation files with class definitions
        if not self._MODEL_CODE_PATHS.search(path):
            return ""
        if "class " not in content or "def __init__" not in content:
            return ""

        # Extract self.xxx assignments from the written code
        written_attrs = set(re.findall(r'self\.(\w+)\s*=', content))
        if len(written_attrs) < 2:
            return ""  # Too small to be a top-level model

        # Check memory for structure enumeration (use semantic search)
        relevant = self.session_memory.query_relevant(
            ["component", "checklist", "structure", "enumeration", "module", "submodules", "porting"],
            max_tokens=2000
        )
        enumeration_content = ""
        for e in relevant:
            enumeration_content += (e.get("content") or "") + "\n"

        if not enumeration_content.strip():
            return ""  # No enumeration to check against

        # Extract component names from enumeration (look for patterns like
        # "vision_encoder", "self.xxx", "- xxx", "[ ] xxx")
        enum_components = set()
        for m in re.finditer(
            r'(?:self\.|(?:^|\n)\s*[-\[\]✓✗ ]*\s*)(\w+(?:_\w+)+)',
            enumeration_content
        ):
            name = m.group(1)
            if len(name) > 3 and name not in ('__init__', 'forward', 'def_forward'):
                enum_components.add(name)

        if not enum_components:
            return ""

        # Find components in enumeration but NOT in written code
        missing = enum_components - written_attrs
        # Filter out common false positives (method names, non-module attrs)
        missing = {m for m in missing if not m.startswith(('num_', 'has_', 'is_', 'use_'))}

        if len(missing) < 2:
            return ""  # Minor gap, don't warn

        if not hasattr(self, '_model_completeness_warnings'):
            self._model_completeness_warnings = 0
        self._model_completeness_warnings += 1

        missing_list = ", ".join(sorted(missing)[:8])
        return (
            f"\n\n[MODEL COMPLETENESS WARNING] Your model class is missing components "
            f"that appear in your structure enumeration:\n"
            f"  Missing: {missing_list}\n"
            f"  Written: {len(written_attrs)} attrs, Enumerated: {len(enum_components)} components\n\n"
            f"A ported model must own ALL submodules from the source — including frozen ones. "
            f"Excluding components (even 'frozen' ones) breaks unified checkpoint conversion "
            f"and prevents future unfreezing. Whether a component is trained is a config "
            f"decision (requires_grad), not an architecture decision.\n"
            f"Check your enumeration and include all components in __init__.\n"
            f"If these components are genuinely not needed (e.g., inference-only heads), "
            f"document why in a comment."
        )

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

        # Check memory for structure enumeration evidence (LLM-based)
        entries = self.session_memory.list_entries()
        has_enumeration = self._llm_check_memory_has_knowledge(
            entries,
            "model structure enumeration or porting checklist: a list of all source model components, "
            "module tree, parameter counts, or component-by-component porting status"
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
        """Run all pre-execution gates. Returns (hard_block, soft_warnings).

        hard_block: dict or None — if set, tool execution is blocked.
            dict keys: name, description, reason, detail
        soft_warnings: list of dicts — soft warnings to append to tool result.
            Each dict has: name, description, reason, detail
        """
        # Hard-block gates (return immediately if triggered)
        hard_block_gates = [
            self._check_use_flagscale_cli_gate,
            self._check_monitor_after_launch_gate,
            self._check_no_delete_experiment_dir_gate,
            self._check_porting_path_gate,
            self._check_porting_path_deviation_gate,
            self._check_pipeline_comprehension_gate,
            self._check_data_model_interface_gate,
            self._check_component_mapping_gate,
            self._check_migration_blueprint_gate,
            self._check_mode_b_design_integrity_gate,
            self._check_megatron_native_integrity_gate,
            self._check_data_pipeline_gate,
            self._check_data_parallelism_gate,
            self._check_train_script_data_pipeline_gate,
            self._check_no_dummy_data_gate,
            self._check_understanding_verification_gate,
            self._check_component_isolation_gate,
            self._check_phase_ordering_gate,
            self._check_structure_completeness_gate,
            self._check_experiment_gate,
        ]
        if not hasattr(self, '_gate_block_counts'):
            self._gate_block_counts = {}
        if not hasattr(self, '_gate_overrides_pending'):
            self._gate_overrides_pending = {}

        for gate in hard_block_gates:
            # Check permanent exemption BEFORE running gate (avoids expensive LLM calls)
            fn_name = gate.__name__
            pre_override_key = fn_name.replace("_check_", "").replace("_gate", "").upper()
            if hasattr(self, '_gate_permanently_passed') and pre_override_key in self._gate_permanently_passed:
                continue

            result = gate(tool_name, arguments)
            if result:
                # Normalize to dict
                if isinstance(result, dict):
                    gate_info = result
                    warning = result["detail"]
                    override_key = result["name"].upper().replace(" ", "_")
                else:
                    override_key = pre_override_key
                    gate_info = {"name": override_key.lower(), "description": "", "reason": "", "detail": result}
                    warning = result

                # Check if LLM has declared an override for this gate
                if override_key in self._gate_overrides_pending:
                    reason = self._gate_overrides_pending.pop(override_key)
                    logger.info(
                        "Gate %s OVERRIDDEN by LLM declaration. Reason: %s",
                        gate_info["name"], reason
                    )
                    self._gate_block_counts.pop(gate_info["name"], None)
                    # Permanent exemption: once overridden, don't block this gate again
                    if not hasattr(self, '_gate_permanently_passed'):
                        self._gate_permanently_passed = set()
                    self._gate_permanently_passed.add(override_key)
                    continue  # Let it through

                self._gate_block_counts[gate_info["name"]] = self._gate_block_counts.get(gate_info["name"], 0) + 1
                count = self._gate_block_counts[gate_info["name"]]
                if count >= 3:
                    warning += (
                        f"\n\n🚨 CRITICAL: This gate has blocked you {count} times. "
                        f"You are stuck in a loop. Your current approach WILL NOT WORK.\n\n"
                        f"MANDATORY: Read the '▶ YOUR NEXT ACTION' section above and do EXACTLY that. "
                        f"Even if you believe you have already satisfied the requirements, "
                        f"the gate disagrees — follow its instructions literally. "
                        f"Do NOT attempt any write/create operation until the gate clears. "
                        f"Do NOT try to bypass via shell (cat >, tee, heredoc). "
                        f"The gate tracks your actual actions, not your beliefs about them.\n\n"
                        f"However, if you are ABSOLUTELY CERTAIN your approach is correct and "
                        f"the gate is wrong, you may declare an override:\n"
                        f"[GATE_OVERRIDE: {override_key}] Reason: <detailed justification why this gate does not apply>\n"
                        f"The override is one-shot: it passes this gate ONCE on your next tool call."
                    )
                elif count >= 2:
                    warning += (
                        f"\n\n⚠ This gate has blocked you {count} times consecutively. "
                        f"You are repeating the same blocked action without addressing the requirements above. "
                        f"STOP attempting this action and complete the prerequisites FIRST. "
                        f"Do NOT try to bypass this gate using shell commands (cat >, tee, python heredoc). "
                        f"Follow the '▶ YOUR NEXT ACTION' instruction above.\n\n"
                        f"If you are CERTAIN this gate does not apply to your situation, "
                        f"you may declare an override in your response:\n"
                        f"[GATE_OVERRIDE: {override_key}] Reason: <detailed justification>\n"
                        f"The override is one-shot and requires a clear, specific reason."
                    )
                else:
                    # First block — mention override exists but encourage following the gate
                    warning += (
                        f"\n\n💡 If after reading the above you are CERTAIN this gate does not apply "
                        f"(e.g., your model genuinely doesn't need this step), you may override:\n"
                        f"[GATE_OVERRIDE: {override_key}] Reason: <why this gate is inapplicable>\n"
                        f"But first, seriously consider whether the gate's requirements are valid for your case."
                    )
                gate_info["detail"] = warning
                return gate_info, []

        # Progress gate (special: returns tuple)
        progress_warning, progress_hard_block = self._check_progress_gate(tool_name)
        if progress_hard_block:
            return {"name": "progress", "description": "Progress stall detection", "reason": "No substantial progress for extended period", "detail": progress_warning}, []

        # Plan creation gate (hard block if "TOOL NOT EXECUTED" in result)
        plan_gate_warning = self._check_plan_creation_gate(tool_name)
        if plan_gate_warning and "TOOL NOT EXECUTED" in plan_gate_warning:
            return {"name": "plan_creation", "description": "Plan creation prerequisite", "reason": "Started implementation without creating a plan", "detail": plan_gate_warning}, []

        # Soft-warning gates (collected and appended to result)
        soft_results = []
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
            self._check_parallelism_assessment_gate,
            self._check_model_completeness_gate,
            self._check_megatron_primitives_usage_gate,
            self._check_mode_b_design_integrity_soft_gate,
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
                if isinstance(w, dict):
                    soft_results.append(w)
                else:
                    name = gate.__name__.replace("_check_", "").replace("_gate", "").replace("_", " ")
                    soft_results.append({"name": name, "description": "", "reason": "", "detail": w})
                if len(soft_results) >= _MAX_SOFT_WARNINGS:
                    break

        if plan_gate_warning:
            soft_results.append({"name": "plan_creation", "description": "", "reason": "", "detail": plan_gate_warning})
        if progress_warning:
            soft_results.append({"name": "progress", "description": "", "reason": "", "detail": progress_warning})

        # Plan maintenance gate (soft warning only)
        plan_maint_warning = self._check_plan_maintenance_gate(tool_name)
        if plan_maint_warning:
            soft_results.append({"name": "plan_maintenance", "description": "", "reason": "", "detail": plan_maint_warning})

        # Config validation hint after YAML write
        config_hint = self._check_config_validation_hint(tool_name)
        if config_hint:
            soft_results.append({"name": "config_validation", "description": "", "reason": "", "detail": config_hint})

        return None, soft_results

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

