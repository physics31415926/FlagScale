"""Unified Judge + JudgeBudget — Phase 2 tiered architecture.

Three-tier classification:
1. Fast path: Heuristic classifiers (zero LLM cost, instant)
2. Cache path: MD5-keyed per-category cache (zero LLM cost)
3. Deep path: LLM calls with multi-round support

Additional features:
- classify_batch(): merge multiple classify calls into one LLM request
- health(), result(), skill(), complexity(): domain-specific judges
- Per-turn call budget (max 64/turn)
- Source tracking: SOURCE_FAST / SOURCE_CACHE / SOURCE_LLM / SOURCE_DEFAULT / SOURCE_UNAVAILABLE
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from flagscale.agent.react.judge_fast import FastClassifier

logger = logging.getLogger(__name__)

# ── Classify prompts (replaces ALL regex/keyword matching) ─────────────────

_CLASSIFY_PROMPTS: Dict[str, str] = {
    "is_error": """\
Determine if this is a REAL execution error that needs attention.

Context: {context}

Answer NO for: warnings, deprecation notices, informational logs,
HTTP error pages returned by web_fetch (these are the content, not an error in the tool itself),
expected cleanup messages, compile messages, pkg_resources deprecation notices.

Answer YES for: crashes, exceptions, OOM, NCCL failures, missing modules,
assertion errors, CUDA errors, process termination, non-zero exit code.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_success": """\
Did this shell command complete successfully?

Context: {context}

Answer YES when the output shows the command completed normally.
Answer NO when there are errors, failures, or unclear outcome.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_dangerous": """\
Is this shell command DANGEROUS and should be BLOCKED?

Context: {context}

Answer YES for: rm -rf on system paths (/ or ~), chmod 777 on system dirs,
fork bombs, mkfs, dd without clear target, redirects to /dev/sd*.
Answer NO for: normal file operations, package management, regular shell commands.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_read_only_shell": """\
Is this a read-only diagnostic command (safe to run anytime)?

Context: {context}

Answer YES for: grep, find, cat, ls, head, tail, wc, file, stat, which, type,
echo, pwd, env, printenv, hostname, uname, date, id, whoami, ps, pgrep, nvidia-smi, rocminfo.
Answer NO for: anything that modifies files, installs packages, launches processes, or writes data.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_training_command": """\
Is this ACTUALLY launching a distributed training job?

Context: {context}

Answer YES for: torchrun, deepspeed, flagscale train (not --dryrun), python pretrain_*.py,
mpirun with training script, horovodrun.
Answer NO for: importing modules, config reading, grep/log analysis,
dryrun, --help, --version, training utilities.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_kill_command": """\
Is this a process kill command?

Context: {context}

Answer YES for: kill, pkill, killall, or equivalent.
Answer NO for anything else.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_training_failure": """\
Did this training output indicate a REAL failure that stopped training?

Context: {context}

Answer YES for: crashes, OOM, NCCL errors, unrecoverable exceptions,
process exit with error, missing dependencies.
Answer NO for: warnings that don't stop training, expected init messages,
non-fatal deprecation warnings, wandb/logging issues.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_zombie_gpu": """\
Does this output indicate zombie GPU processes?

Context: {context}

Answer YES if GPU memory is held by dead/stale processes, or memory allocation conflicts.
Answer NO if GPU state is clean or processes are legitimately running.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_user_porting_confirm": """\
Is the user choosing between Mode B and Mode C?

Context: {context}

Answer "mode_b" if Mode B / Megatron native / 模式B / 原生.
Answer "mode_c" if Mode C / wrapper / 模式C / 包装.
Answer "" (empty) if neither.
Reply ONLY: {"decision": "mode_b"/"mode_c"/""}""",

    "checklist_rule_batch": """\
You are a constraint checker. Below is a tool call and a list of constraints to evaluate.

Tool call context:
{context}

Constraints (each with id, description, prompt):
{items}

For each constraint, follow these two steps:

Step 1 — SCOPE GATE: Each constraint's prompt begins with "SCOPE: <condition>". This condition defines when the constraint is applicable. Read it carefully. If the tool call (tool name + arguments + result) does NOT satisfy the SCOPE condition, this constraint is not applicable — SKIP it entirely.

Step 2 — VIOLATION CHECK: Only if SCOPE matches, check the "CHECK:" part. Does the tool call actually exhibit the violation pattern?

FINAL CROSS-CHECK: Before outputting, review every violation you flagged. For each one, ask: "Does the SCOPE condition genuinely describe this tool call?" If you flagged anything where the answer is no, REMOVE it.

Reply ONLY: {{"violations": [{{"id": "constraint_id", "reason": "one-line explanation"}}]}}
If no constraint is both in-scope AND violated, reply: {{"violations": []}}""",

    "checklist_rule": """\
Evaluate whether a tool action violates a checklist constraint. You are given:

- **Description**: what the constraint requires
- **Prompt**: the specific condition to check for
- **Context**: the tool call details (name, args, result) plus any auto-detected runtime facts

Auto-detected facts (if present) are authoritative — trust them over any inference from the tool call itself. Examples:
- _facts.shared_storage: paths like ["/share/project"] mean shared storage IS available
- _facts.driver_version: the actual NVIDIA driver version from nvidia-smi
- _facts.gpu_count: the actual GPU count detected at startup

Context: {context}

Constraint (id={item_id}): {description}

Check for: {prompt}

Reply ONLY: {{"match": true/false, "reason": "one-line explanation of why this does or does not violate the constraint"}}""",

    "is_frozen_excuse": """\
Determine if this gate override reason is an INVALID excuse based on "frozen parameters."

A gate override is INVALID if the reason claims a component can be skipped because:
- Parameters are frozen / no_grad / requires_grad=False
- It's a feature extractor with no trainable params
- It's inference-only or not trained
- Frozen layers don't need native implementation
- No TP benefit because parameters are frozen

These are NOT valid reasons to skip design integrity gates — frozen parameters still need correct architecture mapping for inference, checkpointing, and future fine-tuning.

A gate override IS VALID if the reason is about genuine architectural differences, missing features in the target framework, or legitimate design decisions unrelated to parameter freezing.

Reason text: {reason}
Gate: {gate_name}

Reply ONLY: {{"real": true/false, "need_more": null}}
(true = this IS an invalid frozen excuse, false = this is a legitimate reason)""",

    "extract_constraints": """\
Read the skill content below and extract ONLY constraints that can be checked by looking at a single tool call result (shell command + output, file write, file read).

Principle: a constraint is valid if, given the tool call context, an LLM can answer "does this tool call violate the rule?" with confidence. If the answer requires knowing what the agent did NOT do, or requires multi-step reasoning about the agent's plan, SKIP it.

SEVERITY (critical — choose carefully):
- "error" (HARD BLOCK): violating this constraint causes irreversible harm — package conflicts destroying environments, training on wrong data, incorrect results that waste hours. The tool call will be BLOCKED before execution when the SCOPE matches. Only use for truly destructive mistakes.
- "warning" (soft reminder): violating this is suboptimal but not immediately destructive — suboptimal config, missing optimization, mild performance impact. The tool call still executes.

For each constraint you extract, output:
- id: snake_case prefixed with the skill name
- description: 1 line
- trigger_on: {{"tool": "shell"|"write_file"|"read_file"|"edit_file"}}
- prompt: "SCOPE: <concrete condition>. CHECK: <violation signal>."
- reminder: 1-sentence warning to show the agent
- severity: "error" or "warning"
- max_reminders: 3-5

Skill content:
{skill_content}

Reply ONLY with a JSON array. If no checkable constraints, return []. Do NOT invent constraints the skill doesn't describe.""",

}


# ── Health judge prompt ──────────────────────────────────────────────────

_HEALTH_JUDGE_PROMPT = """\
You are monitoring a running shell command. Analyze its status and decide
whether it should continue or be terminated.

Command: {command}
Total elapsed: {elapsed}
Output changed since last check: {output_changed}
Consecutive checks with no output change: {stall_count}
Recent output:
{output}

Phase-aware monitoring - adapt check frequency to the command's lifecycle stage:
- STARTUP (no output yet, imports loading, initializing): check frequently (10-30s). Early failures are common.
- LOADING (model weights loading, data downloading, progress bars advancing): moderate (30-60s).
- STABLE (training iterations running, loss printing regularly): relaxed (120-300s).
- ANOMALY (errors in output, repeated failures, output stalled unexpectedly): check soon (10-15s) or kill.

Key judgment rules:
- If output is actively progressing (new lines, advancing percentages): healthy, adjust interval to phase.
- If output has stalled but the operation is known to be slow (large compile, decompression): allow more time.
- If the command contains embedded sleep/wait and produces no output: you CANNOT verify whether the process being waited for is still alive. After a reasonable initial wait, KILL the command so the agent can check external state (process liveness, GPU status, logs) with its full tool set, then decide whether to retry.
- If you see repeated errors, network failures, or crash signatures: kill immediately.
- Do NOT let a silent command run indefinitely just because 'it might be working.' When in doubt, kill early.

Reply with ONLY a JSON object:
{{"kill": true/false, "reason": "...", "next_check_seconds": <int 10-300>}}

If everything looks normal and healthy, set reason to empty string.
Only provide a reason when there is something noteworthy - an issue, a phase transition, or a kill decision."""


_RESULT_JUDGE_PROMPT = """\
You are analyzing the output of a shell command run by an AI infrastructure agent.

Command: {command}
Duration: {elapsed:.0f}s
Output (last 3000 chars):
{output}

Identify issues and provide SHORT, actionable annotations. Consider:
- Non-zero exit code or error messages: identify root cause
- CUDA/cuDNN/driver version conflicts: give specific diagnosis commands
- Network errors (connection refused, timeout, DNS): suggest proxy or retry
- Download failures: suggest resume with wget -c or curl -C -
- PyTorch/CUDA incompatibility: suggest version check commands
- Inefficient patterns (sleep+tail for monitoring): suggest find_latest_log or timeout+tail -f
- Log searching with find/ls -R/ls -lt: suggest find_latest_log tool or workspace_experiment list
- Training launch (flagscale/torchrun/deepspeed): remind to verify GPU utilization and logs
- Package install success (pip/conda): remind to verify runtime compatibility
- pip upgraded/downgraded a critical package (torch, numpy, etc.): WARN that this may break CUDA compatibility
- Long duration (>2min) for simple commands: flag as unexpected
- OOM (out of memory): suggest reducing batch size, enabling gradient checkpointing, or adjusting parallelism
- NCCL errors: suggest checking network config, NCCL env vars, and multi-node connectivity
- Training output showing ce_loss or lm_loss near ln(vocab_size) (10.4, 10.8, 11.1, 11.8): WARN: loss indicates random output, check weight loading
- Config file edit containing path values: remind to verify paths exist before launching
- Reading/grepping code from a different workspace than the current one: WARN: source code provenance mismatch
- cp -r from another environment's site-packages: WARN: never copy packages between environments, use pip install
- Checkpoint conversion output showing 'missed' or 'skipped' or 'unexpected' keys: WARN: audit the FULL list
- Checkpoint saved to disk without a reload verification: WARN: verify saved checkpoint
- Training log/output showing crash, error, or exitcode!=0: WARN: update the experiment via workspace_experiment
- Training log showing successful completion: remind to update experiment entry with final metrics

Reply with ONLY a JSON object:
  {{"annotations": ["annotation1", "annotation2"], "severity": "info|warning|error"}}
If no issues: {{"annotations": [], "severity": "info"}}"""


_SKILL_JUDGE_PROMPT = """\
You are deciding which skill (if any) to load for an AI infrastructure agent.

User request: {user_input}
Conversation context: {conversation_context}
Available skills:
{skills_list}
Already loaded: {loaded}

IMPORTANT - dependency chains:
{dependency_chains}

Rules:
- Only suggest a skill if it's clearly relevant to the user's request
- If the user explicitly names a skill or task that maps to one, suggest it
- If the request is ambiguous or general, suggest nothing
- Never suggest a skill that's already loaded
- Match user intent to skill keywords and descriptions listed above
- The system will automatically load all requires/suggests of each skill you select

Reply with ONLY a JSON object:
  {{"skills": ["skill-name"]}} or {{"skills": []}}"""


_COMPLEXITY_JUDGE_PROMPT = """\
You are evaluating whether a user request requires a structured task plan.

User request: {user_input}
Active plan exists: {has_plan}
Session memory context: {memory_context}

A task needs planning when:
- It involves 3+ distinct sequential steps (install -> configure -> run -> verify)
- Steps have dependencies (can't train before data is ready)
- It will take multiple tool calls across different domains (download, config, shell)
- Failure at one step requires knowing what was already done

A task does NOT need planning when:
- Simple question or lookup
- Single command execution
- Continuing an existing plan (plan already exists)
- Quick fix or small edit
- User is asking a follow-up question in an ongoing conversation

Reply with ONLY a JSON object:
  {{"needs_plan": true/false, "reason": "one-line explanation"}}"""


_ROUTE_INTENT_PROMPT = """\
You are routing a user request to the right execution mode.

User request: {user_input}

Determine the execution mode:

- "single": Simple task that one worker can handle directly.
  Examples: "read this file", "run this command", "explain this code",
  "fix this bug", "configure this parameter", "check GPU status",
  "what version is installed", "how do I..."

- "subtask": Multi-stage task that needs a serial pipeline with DAG stages.
  Examples: "set up environment AND reproduce training", "migrate model from HF to Megatron",
  "download source code AND configure AND train", "build env, download data, run training"

- "batch": Comparing multiple independent variants.
  Examples: "compare training with tp=2 vs tp=4", "run experiment A and experiment B",
  "which config is better: X or Y", "try both approaches and compare"

Available profiles: {profiles}

Choose the profile that best matches the task domain. Use "general" for simple shell operations,
file inspection, cleanup, or Q&A that don't need domain-specific skills.

Reference subtask templates (you may reuse or ignore these): {templates}

Reply with ONLY a JSON object:
{{"mode": "single"|"subtask"|"batch", "profile": "<profile_name>"}}

For "subtask" mode — choose one of two approaches:

  A) Reuse an existing template:
     {{
       "mode": "subtask",
       "profile": "<profile_name>",
       "template": "<template_name>"
     }}

  B) Generate custom stages when no template fits:
     {{
       "mode": "subtask",
       "profile": "<default_profile_name>",
       "template": "",
       "dynamic_stages": [
         {{"id": "stage_1", "description": "what to do", "profile": "env-setup", "depends_on": []}},
         {{"id": "stage_2", "description": "what to do next", "profile": "training-reproduce", "depends_on": ["stage_1"]}},
         {{"id": "stage_3", "description": "what to do last", "profile": "training-reproduce", "depends_on": ["stage_2"]}}
       ]
     }}
     Each stage needs: id (unique), description (1 sentence), profile (from available profiles),
     depends_on (list of stage ids that must complete first, empty list for first stage).

For "batch" mode, also include:
  "batch_tasks": ["<task1 description>", "<task2 description>"]

For "single" mode, omit template, dynamic_stages, and batch_tasks."""

# Register route_intent in the classify prompts dict (must be after the prompt definition)
_CLASSIFY_PROMPTS["route_intent"] = _ROUTE_INTENT_PROMPT

# ── Skill suggestion (semantic, replaces keyword matching) ────────────────────

_CLASSIFY_PROMPTS["skill_suggest"] = """\
Given the user request, decide which skills (if any) should be loaded.

User request: {user_input}

Available skills (not yet loaded):
{available_skills}

Rules:
- Only suggest skills that are clearly relevant to the user's request.
- For simple operations (delete files, check status, list processes, read files), return empty list.
- Return ONLY a JSON array of skill names, e.g. ["train-config", "train-run"] or [].
"""

# ── Constraint violation judgment (Phase 3) ──────────────────────────────────

_CLASSIFY_PROMPTS["is_constraint_violated"] = """\
Determine if this tool call violates the given constraint.

Constraint: {constraint}
Judgment prompt: {prompt}

Tool call:
- Tool: {tool_name}
- Args: {tool_args}
- Result: {tool_result}

Reply ONLY: {{"real": true/false, "need_more": null}}
(true = constraint IS violated, false = constraint is NOT violated)"""

# ── JudgeBudget ──────────────────────────────────────────────────────────

@dataclass
class JudgeBudget:
    """Per-turn call budget for LLM judges.

    Strategy:
    - Max 16 judge calls per turn (classify may use multi-round, plus health/result/skill/complexity)
    - Each judge type caches independently
    - On exhaustion: health -> heuristic, skill/complexity -> default, classify -> cached/default
    """

    max_calls_per_turn: int = 64
    calls_this_turn: int = 0
    total_calls: int = 0
    total_saved_by_cache: int = 0
    _skipped_summary: str = ""  # summary of skipped categories for user-visible warning
    _exhausted_warned: bool = False  # only warn once per turn

    @property
    def exhausted(self) -> bool:
        return self.calls_this_turn >= self.max_calls_per_turn

    def consume(self) -> bool:
        """Return True if budget allows another call, incrementing if so."""
        if self.calls_this_turn >= self.max_calls_per_turn:
            return False
        self.calls_this_turn += 1
        self.total_calls += 1
        return True

    def note_skipped(self, source: str, category: str):
        """Record a skipped classify call for later reporting."""
        if self._skipped_summary:
            self._skipped_summary += f", {source}/{category}"
        else:
            self._skipped_summary = f"{source}/{category}"

    def reset_turn(self):
        self.calls_this_turn = 0
        self._skipped_summary = ""

    @property
    def skipped_detail(self) -> str:
        return self._skipped_summary


# ── Judge ────────────────────────────────────────────────────────────────

# ── Classify source tracking ────────────────────────────────────────────

#: Fast-path heuristic returned a confident answer (no LLM call).
SOURCE_FAST = "fast"
#: LLM returned a valid classification.
SOURCE_LLM = "llm"
#: Result was served from local MD5 cache.
SOURCE_CACHE = "cache"
#: Budget exhausted or provider unavailable — default value returned.
SOURCE_DEFAULT = "default"
#: Provider is None (never initialized) — no LLM available at all.
SOURCE_UNAVAILABLE = "unavailable"


class ClassifyTrace:
    """Per-turn trace of classify() calls: category → source.

    Attached to Judge._last_trace after each classify() call.
    Callers (especially safety-critical Guards) can inspect
    trace to decide whether to trust the result or take conservative action.
    """

    def __init__(self):
        self._entries: dict[str, str] = {}  # category → source

    def record(self, category: str, source: str):
        self._entries.setdefault(category, source)

    def source_of(self, category: str) -> str:
        """Return the source for a category, or 'unavailable' if never called."""
        return self._entries.get(category, SOURCE_UNAVAILABLE)

    def any_from(self, *sources: str) -> bool:
        """True if any recorded call has one of the given sources."""
        return any(s in sources for s in self._entries.values())

    def clear(self):
        self._entries.clear()


class Judge:
    """Unified LLM judge with budget control, caching, and multi-round classify.

    classify() returns a (value, source) tuple so safety-critical guards can
    distinguish "LLM said safe" from "Judge unavailable, assuming safe by default."
    """

    _MAX_CLASSIFY_ROUNDS = 3

    def __init__(self, provider, budget: JudgeBudget | None = None):
        self.provider = provider
        self.budget = budget or JudgeBudget()
        self._trace = ClassifyTrace()

        # Caches
        self._health_cache: dict[str, dict] = {}
        self._result_cache: dict[str, list] = {}
        self._skill_cache: dict[str, list] = {}
        self._classify_cache: dict[str, dict] = {}

    def reset_turn(self):
        """Reset per-turn budget and trace. Caches stay warm across turns."""
        self.budget.reset_turn()
        self._trace.clear()

    # ── classify: replaces ALL regex/keyword matching ─────────────────────

    def classify(self, category: str, context: dict, default: Any = None) -> Any:
        """Lightweight LLM classification. Replaces all regex/keyword matching.

        category: one of "is_error", "is_success", "is_dangerous", "is_read_only_shell",
                  "is_training_command", "is_kill_command", "is_training_failure",
                  "is_zombie_gpu", "is_user_porting_confirm", "checklist_rule",
                  "checklist_rule_batch", "route_intent"

        context: dict with relevant fields. LLM can request more in multi-round mode.

        Returns SAME TYPE as before (bool, str, dict, list) — the return value
        contract is unchanged. Callers that need source information should instead
        use classify_traced() or inspect self._trace.source_of(category).
        """
        value, _source = self.classify_traced(category, context, default)
        return value

    def classify_traced(self, category: str, context: dict, default: Any = None) -> tuple[Any, str]:
        """Same as classify() but returns (value, source) tuple.

        Three-tier resolution:
        1. Fast path: heuristic classifiers (instant, no LLM)
        2. Cache path: MD5-keyed cache hit
        3. Deep path: LLM call with multi-round support

        source is one of: SOURCE_FAST, SOURCE_LLM, SOURCE_CACHE, SOURCE_DEFAULT, SOURCE_UNAVAILABLE.

        Safety-critical callers (SafetyGuard) should use this method and
        treat SOURCE_DEFAULT / SOURCE_UNAVAILABLE as "unknown → be conservative."
        """
        # Provider never initialized
        if self.provider is None:
            self._trace.record(category, SOURCE_UNAVAILABLE)
            return (default, SOURCE_UNAVAILABLE)

        # ── Tier 1: Fast path (heuristic) ────────────────────────────────
        fast_result = self._try_fast_path(category, context)
        if fast_result is not None:
            self._trace.record(category, SOURCE_FAST)
            return (fast_result, SOURCE_FAST)

        # ── Tier 2: Cache path ───────────────────────────────────────────
        cache_key = self._classify_cache_key(category, context)
        if cache_key in self._classify_cache:
            self.budget.total_saved_by_cache += 1
            self._trace.record(category, SOURCE_CACHE)
            return (self._classify_cache[cache_key], SOURCE_CACHE)

        # ── Tier 3: Deep path (LLM) ─────────────────────────────────────
        prompt_template = _CLASSIFY_PROMPTS.get(category)
        if not prompt_template:
            logger.warning("Unknown classify category: %s", category)
            self._trace.record(category, SOURCE_DEFAULT)
            return (default, SOURCE_DEFAULT)

        truncated = self._truncate_context(context, max_chars=800)

        for round_num in range(self._MAX_CLASSIFY_ROUNDS):
            if self.budget.exhausted:
                break
            if not self.budget.consume():
                break

            prompt = prompt_template
            if "{context}" in prompt:
                prompt = prompt.replace("{context}", self._format_context(truncated))
            if "{rule}" in prompt:
                prompt = prompt.replace("{rule}", json.dumps(context.get("rule", ""), ensure_ascii=False))
            if "{item_id}" in prompt:
                prompt = prompt.replace("{item_id}", str(context.get("item_id", "")))
            if "{description}" in prompt:
                prompt = prompt.replace("{description}", str(context.get("description", "")))
            if "{prompt}" in prompt:
                # The constraint-specific prompt from ChecklistItem
                prompt = prompt.replace("{prompt}", str(context.get("prompt", "")))
            if "{items}" in prompt:
                # JSON array of {id, description, prompt} for batch checklist evaluation
                prompt = prompt.replace("{items}", json.dumps(context.get("items", []), ensure_ascii=False))
            if "{skill_content}" in prompt:
                prompt = prompt.replace("{skill_content}", str(context.get("skill_content", "")))

            data = self._call_and_parse(prompt, default={})
            need_more = data.get("need_more") if isinstance(data, dict) else None
            if need_more and isinstance(need_more, list) and round_num < self._MAX_CLASSIFY_ROUNDS - 1:
                for field in need_more:
                    if field in context and field not in truncated:
                        truncated[field] = self._truncate_one(str(context[field]), max_chars=2000)[:2000]
                continue

            result = self._parse_classify_result(category, data, default)
            self._classify_cache[cache_key] = result
            self._trace.record(category, SOURCE_LLM)
            return (result, SOURCE_LLM)

        self.budget.note_skipped("classify", category)
        logger.warning(
            "Judge.classify [%s] budget exhausted (%d/%d) — returning default=%r, "
            "safety/interrupt checks may be degraded. "
            "Consider reducing batch size or checklist items.",
            category, self.budget.calls_this_turn, self.budget.max_calls_per_turn, default,
        )
        self._classify_cache[cache_key] = default
        self._trace.record(category, SOURCE_DEFAULT)
        return (default, SOURCE_DEFAULT)

    # ── Fast path dispatch ──────────────────────────────────────────────────

    _FAST_DISPATCH = {
        "is_read_only_shell": lambda ctx: FastClassifier.is_read_only_shell(
            str(ctx.get("command", ""))
        ),
        "is_dangerous": lambda ctx: FastClassifier.is_dangerous(
            str(ctx.get("command", ""))
        ),
        "is_training_command": lambda ctx: FastClassifier.is_training_command(
            str(ctx.get("command", ""))
        ),
        "is_kill_command": lambda ctx: FastClassifier.is_kill_command(
            str(ctx.get("command", ""))
        ),
    }

    def _try_fast_path(self, category: str, context: dict) -> Any:
        """Try fast-path heuristic for a category.

        Returns the classification result if confident, None to escalate to LLM.
        """
        dispatch = self._FAST_DISPATCH.get(category)
        if dispatch is None:
            return None
        return dispatch(context)

    # ── Batch classify ───────────────────────────────────────────────────

    def classify_batch(
        self, items: list[tuple[str, dict, Any]],
    ) -> list[tuple[Any, str]]:
        """Classify multiple items, batching LLM calls where possible.

        items: list of (category, context, default) tuples.

        Returns list of (value, source) tuples in same order.

        Strategy:
        1. Resolve fast-path and cache hits immediately
        2. Batch remaining items into a single LLM call if they share the same
           category (e.g., multiple is_error checks)
        3. Fall back to individual calls for mixed categories
        """
        results: list[tuple[Any, str] | None] = [None] * len(items)
        pending: list[tuple[int, str, dict, Any]] = []  # (index, category, context, default)

        # Phase 1: resolve fast-path and cache
        for i, (category, context, default) in enumerate(items):
            if self.provider is None:
                self._trace.record(category, SOURCE_UNAVAILABLE)
                results[i] = (default, SOURCE_UNAVAILABLE)
                continue

            # Fast path
            fast_result = self._try_fast_path(category, context)
            if fast_result is not None:
                self._trace.record(category, SOURCE_FAST)
                results[i] = (fast_result, SOURCE_FAST)
                continue

            # Cache path
            cache_key = self._classify_cache_key(category, context)
            if cache_key in self._classify_cache:
                self.budget.total_saved_by_cache += 1
                self._trace.record(category, SOURCE_CACHE)
                results[i] = (self._classify_cache[cache_key], SOURCE_CACHE)
                continue

            pending.append((i, category, context, default))

        # Phase 2: batch LLM calls for same-category items
        if pending and not self.budget.exhausted:
            # Group by category
            by_category: dict[str, list[tuple[int, dict, Any]]] = {}
            for idx, cat, ctx, dflt in pending:
                by_category.setdefault(cat, []).append((idx, ctx, dflt))

            for cat, group in by_category.items():
                if self.budget.exhausted:
                    for idx, ctx, dflt in group:
                        self._trace.record(cat, SOURCE_DEFAULT)
                        results[idx] = (dflt, SOURCE_DEFAULT)
                    continue

                # Single item — just do normal classify
                if len(group) == 1:
                    idx, ctx, dflt = group[0]
                    value, source = self.classify_traced(cat, ctx, dflt)
                    results[idx] = (value, source)
                    continue

                # Multiple items of same category — individual calls
                # (True batching into one prompt is only for checklist_rule_batch)
                for idx, ctx, dflt in group:
                    if self.budget.exhausted:
                        self._trace.record(cat, SOURCE_DEFAULT)
                        results[idx] = (dflt, SOURCE_DEFAULT)
                    else:
                        value, source = self.classify_traced(cat, ctx, dflt)
                        results[idx] = (value, source)

        # Fill any remaining None slots with defaults
        for i, item in enumerate(results):
            if item is None:
                cat, ctx, dflt = items[i]
                self._trace.record(cat, SOURCE_DEFAULT)
                results[i] = (dflt, SOURCE_DEFAULT)

        return results  # type: ignore[return-value]

    # ── Health judge ──────────────────────────────────────────────────────

    def health(
        self, command: str, recent_output: str, elapsed: str,
        output_changed: bool = True, stall_count: int = 0,
    ) -> dict:
        """Evaluate whether a long-running command is healthy."""
        if self.budget.exhausted:
            self.budget.note_skipped("health", "health")
            if stall_count >= 3:
                return {"kill": True, "reason": "Output stalled and health check unavailable (judge budget exhausted)"}
            return {"kill": False, "reason": "Judge budget exhausted, health check skipped — command still running"}

        cache_key = hashlib.md5(
            f"{command[:100]}:{elapsed}:{stall_count}".encode()
        ).hexdigest()[:12]
        if cache_key in self._health_cache:
            self.budget.total_saved_by_cache += 1
            return self._health_cache[cache_key]

        if not self.budget.consume():
            return {"kill": False}

        prompt = _HEALTH_JUDGE_PROMPT.format(
            command=command, elapsed=elapsed,
            output=recent_output[-2000:],
            output_changed="yes" if output_changed else "no",
            stall_count=stall_count,
        )
        result = self._call_and_parse(prompt, default={"kill": False})
        self._health_cache[cache_key] = result
        return result

    # ── Result judge ──────────────────────────────────────────────────────

    def result(self, command: str, output: str, elapsed: float) -> list[str]:
        """Analyze shell output and return annotations."""
        if self.budget.exhausted:
            return []

        cache_key = hashlib.md5(
            f"{command[:100]}:{output[-500:]}".encode()
        ).hexdigest()[:12]
        if cache_key in self._result_cache:
            self.budget.total_saved_by_cache += 1
            return self._result_cache[cache_key]

        if not self.budget.consume():
            return []

        prompt = _RESULT_JUDGE_PROMPT.format(
            command=command, elapsed=elapsed, output=output[-3000:],
        )
        data = self._call_and_parse(prompt, default={})
        annotations = data.get("annotations", [])
        self._result_cache[cache_key] = annotations
        return annotations

    # ── Skill judge ───────────────────────────────────────────────────────

    def skill(
        self, user_input: str, skills_list: str, loaded: str,
        dependency_chains: str, conversation_context: str, valid_names: set[str],
    ) -> list[str]:
        """Decide which skill to auto-load."""
        if self.budget.exhausted:
            return []

        cache_key = hashlib.md5(user_input[:200].encode()).hexdigest()[:12]
        if cache_key in self._skill_cache:
            self.budget.total_saved_by_cache += 1
            return self._skill_cache[cache_key]

        if not self.budget.consume():
            return []

        prompt = _SKILL_JUDGE_PROMPT.format(
            user_input=user_input[:500],
            conversation_context=conversation_context,
            skills_list=skills_list, loaded=loaded,
            dependency_chains=dependency_chains,
        )
        data = self._call_and_parse(prompt, default={})
        skills = data.get("skills", [])
        result = [s for s in skills if s and s in valid_names]
        self._skill_cache[cache_key] = result
        return result

    # ── Complexity judge ──────────────────────────────────────────────────

    def complexity(
        self, user_input: str, has_plan: bool = False,
        memory_context: str = "",
    ) -> dict:
        """Evaluate whether a user request needs a task plan."""
        if self.budget.exhausted:
            return {"needs_plan": False}

        if not self.budget.consume():
            return {"needs_plan": False}

        prompt = _COMPLEXITY_JUDGE_PROMPT.format(
            user_input=user_input,
            has_plan="yes" if has_plan else "no",
            memory_context=memory_context[:500] if memory_context else "(none)",
        )
        return self._call_and_parse(prompt, default={"needs_plan": False})

    # ── Route intent (replaces Orchestrator regex routing) ─────────────────

    def route(self, user_input: str, profiles: str, templates: str) -> tuple[dict, str]:
        """Route a user request to the right execution mode via LLM.

        Returns ((mode_dict, source)), where source is SOURCE_LLM / SOURCE_UNAVAILABLE.
        mode_dict always contains at least {"mode": "single"}.

        Callers should check source: if SOURCE_UNAVAILABLE, fall back to regex routing.
        This does NOT consume budget — routing happens once per user request,
        before the agent loop starts.
        """
        if self.provider is None:
            return ({"mode": "single"}, SOURCE_UNAVAILABLE)

        context = {
            "user_input": user_input,
            "profiles": profiles,
            "templates": templates,
        }
        value, source = self.classify_traced("route_intent", context,
            default={"mode": "single"})
        if not isinstance(value, dict):
            value = {"mode": "single"}
        return (value, source)

    def suggest_skills(self, user_input: str, available_skills: list[dict]) -> list[str]:
        """Suggest which skills to load based on semantic understanding.

        Args:
            user_input: The user's request text.
            available_skills: List of {"name": ..., "description": ...} for unloaded skills.

        Returns:
            List of skill names to load (may be empty).
        """
        if not available_skills:
            return []

        skills_str = "\n".join(
            f"- {s['name']}: {s.get('description', '')}" for s in available_skills
        )
        context = {
            "user_input": user_input,
            "available_skills": skills_str,
        }
        value, source = self.classify_traced("skill_suggest", context, default=[])
        if not isinstance(value, list):
            return []
        # Validate: only return names that exist in available_skills
        valid_names = {s["name"] for s in available_skills}
        return [n for n in value if isinstance(n, str) and n in valid_names]

    @property
    def trace(self) -> ClassifyTrace:
        """Expose per-turn classify trace for safety-critical callers."""
        return self._trace

    def extract_constraints(self, skill_content: str) -> list[dict]:
        """Extract checklist constraints from a skill's content via LLM.

        Called once per skill load. Returns a list of constraint dicts
        suitable for ChecklistItem construction.
        """
        # Don't count against the per-turn classify budget — this is
        # initialization, not per-tool-call overhead.
        import hashlib
        cache_key = hashlib.md5(skill_content[:500].encode()).hexdigest()[:12]
        if cache_key in self._classify_cache:
            return self._classify_cache[cache_key]

        result = self.classify("extract_constraints",
            {"skill_content": skill_content}, default=[])
        logger.info("extract_constraints returned type=%s len=%s",
                     type(result).__name__, len(result) if isinstance(result, (list, dict)) else "?")
        self._classify_cache[cache_key] = result
        return result

    # ── Classify helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_classify_result(category: str, data: dict, default: Any) -> Any:
        """Extract classification decision from LLM response."""
        if category == "is_user_porting_confirm":
            text = str(data.get("decision", "") or data.get("mode", "") or "").lower()
            if "mode_b" in text or "mode b" in text or "b" == text:
                return "mode_b"
            if "mode_c" in text or "mode c" in text or "c" == text:
                return "mode_c"
            return ""
        if category == "checklist_rule_batch":
            # LLM may return a list directly or {"violations": [...]}
            if isinstance(data, list):
                return data
            violations = data.get("violations", []) if isinstance(data, dict) else []
            if isinstance(violations, list):
                return violations
            return []
        if category == "checklist_rule":
            match = data.get("match")
            if isinstance(match, bool):
                return {"match": match, "reason": data.get("reason", "")}
            return {"match": False, "reason": ""}
        if category == "extract_constraints":
            # _parse_json may return a list directly
            if isinstance(data, list):
                return data
            # Sometimes LLM wraps in {"constraints": [...]}
            if isinstance(data, dict):
                constraints = data.get("constraints", [])
                if isinstance(constraints, list):
                    return constraints
            return []
        if category == "route_intent":
            # Return the full dict: {mode, profile, template, batch_tasks, dynamic_stages}
            if isinstance(data, dict):
                return data
            return default if default is not None else {"mode": "single"}
        if category == "skill_suggest":
            # LLM should return a JSON array of skill names
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                skills = data.get("skills", [])
                if isinstance(skills, list):
                    return skills
            return []
        # Boolean categories
        real = data.get("real")
        if isinstance(real, bool):
            return real
        decision = data.get("decision")
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, str):
            return decision.lower() in ("yes", "true", "y")
        return default if default is not None else False

    @staticmethod
    def _classify_cache_key(category: str, context: dict) -> str:
        raw = category + json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _truncate_context(context: dict, max_chars: int = 800) -> dict:
        result = {}
        for k, v in context.items():
            result[k] = Judge._truncate_one(str(v), max_chars)
        return result

    @staticmethod
    def _truncate_one(text: str, max_chars: int = 800) -> str:
        """Truncate preserving both head and tail (errors usually at end)."""
        if len(text) <= max_chars:
            return text
        head = text[:max_chars // 4]
        tail = text[-(max_chars - max_chars // 4):]
        return f"{head}\n... [{len(text) - max_chars} chars omitted] ...\n{tail}"

    @staticmethod
    def _format_context(context: dict) -> str:
        lines = []
        for k, v in context.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in v.items():
                    lines.append(f"  {sub_k}: {sub_v}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    # ── LLM helpers ───────────────────────────────────────────────────────

    def _call_and_parse(self, prompt: str, default: dict | list) -> dict | list:
        """Make a single LLM call and parse JSON from response."""
        text = self._call(prompt)
        if not text:
            return default
        result = self._parse_json(text)
        return result if result else default

    def _call(self, prompt: str) -> str:
        """Dispatch LLM call through provider."""
        if self.provider is None:
            return ""
        try:
            response = self.provider.chat(
                [{"role": "user", "content": prompt}], tools=[]
            )
            return (response.get("content") or "").strip()
        except Exception as e:
            logger.warning("Judge LLM call failed: %s", e)
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """Extract JSON object or array from LLM response text.

        Handles trailing content after the JSON and tries both
        {...} and [...] top-level formats.
        """
        text = text.strip()
        # Try the whole text first
        for candidate in (text,):
            if not candidate:
                continue
            # Trim trailing characters beyond the last ] or }
            if candidate.startswith("["):
                end = candidate.rfind("]")
                if end > 0:
                    candidate = candidate[:end + 1]
            elif candidate.startswith("{"):
                end = candidate.rfind("}")
                if end > 0:
                    candidate = candidate[:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Fallback: find the outermost JSON bounds
        for first_char, last_char in [("[", "]"), ("{", "}")]:
            start = text.find(first_char)
            end = text.rfind(last_char)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
