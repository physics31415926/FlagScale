"""Unified Judge + JudgeBudget for v3.

Consolidates v1's 11+ LLM judge calls into a single entry point with:
- Per-category caching (MD5 for regex, text for health/result)
- Per-turn call budget (max 3/turn)
- Graceful fallback on budget exhaustion
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Judge prompts (from v1, with non-ASCII chars replaced) ─────────────────

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
- For environment setup/installing: suggest both 'env-setup' AND 'workspace-layout' (global rule: workspace-layout is REQUIRED for ANY env creation, code download, conda install, or pip install task)
- For creating conda envs, downloading code/models/data: always include 'workspace-layout' first
- For training tasks: suggest 'train-run' (which requires workspace-layout - include it too)
- For model porting/migration: suggest 'model-porter'
- For monitoring: suggest 'train-monitor'
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

_REGEX_JUDGE_PROMPTS = {
    "is_error": (
        "Is this a REAL training/runtime error that needs immediate attention?\n"
        "Answer NO for: warnings, deprecation notices, wandb messages, "
        "informational logs, expected cleanup messages, torch compile messages, "
        "pkg_resources deprecation, TensorFloat-32 notices.\n"
        "Answer YES for: crashes, exceptions, OOM, NCCL failures, missing modules, "
        "assertion errors, CUDA errors, process termination.\n"
    ),
    "is_training_launch": (
        "Is this shell command ACTUALLY launching a distributed training job?\n"
        "Answer NO for: importing train modules, reading train configs, "
        "grepping train logs, running train-related utilities (not training itself), "
        "dryrun/script-generation commands, --help, --version.\n"
        "Answer YES for: commands that start real GPU training processes "
        "(torchrun, deepspeed, flagscale train without --dryrun, python pretrain_*.py).\n"
    ),
    "is_training_failure": (
        "Did this training output indicate a REAL failure that stopped training?\n"
        "Answer NO for: warnings that don't stop training, expected messages during init, "
        "wandb/logging issues, non-fatal deprecation warnings, informational messages.\n"
        "Answer YES for: crashes, OOM, NCCL errors, unrecoverable exceptions, "
        "process exit with error, missing dependencies that prevent training.\n"
    ),
    "is_checkpoint_load": (
        "Is this command loading pretrained model weights/checkpoint?\n"
        "Answer NO for: saving checkpoints, listing checkpoint files, "
        "checking checkpoint existence, training from scratch.\n"
        "Answer YES for: commands with flags that load pretrained weights "
        "(--load, --resume, --finetune-from, --pretrained-model, --init-checkpoint).\n"
    ),
    "is_flagscale_train": (
        "Is this a FlagScale training command (not dryrun, not stop, not help)?\n"
        "Answer NO for: flagscale --dryrun, flagscale --stop, flagscale --help, "
        "reading flagscale configs, importing flagscale modules.\n"
        "Answer YES for: flagscale train commands that will actually launch training.\n"
    ),
}


# ── JudgeBudget ────────────────────────────────────────────────────────────

@dataclass
class JudgeBudget:
    """Per-turn call budget for LLM judges.

    Strategy:
    - Max 3 judge calls per turn (health + result + regex may fire simultaneously)
    - Each judge type caches independently
    - On exhaustion: health -> heuristic, skill/complexity -> default, regex -> trust regex
    """

    max_calls_per_turn: int = 3
    calls_this_turn: int = 0
    total_calls: int = 0
    total_saved_by_cache: int = 0

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

    def reset_turn(self):
        self.calls_this_turn = 0


# ── Judge ──────────────────────────────────────────────────────────────────

class Judge:
    """Unified LLM judge with budget control and caching.

    Usage:
        judge = Judge(provider, budget=JudgeBudget(max_calls_per_turn=3))
        result = judge.health(cmd, output, elapsed, changed, stall)
    """

    def __init__(self, provider, budget: JudgeBudget | None = None):
        self.provider = provider
        self.budget = budget or JudgeBudget()

        # Caches
        self._regex_cache: dict[str, bool] = {}
        self._health_cache: dict[str, dict] = {}
        self._result_cache: dict[str, list] = {}
        self._skill_cache: dict[str, list] = {}

    def reset_turn(self):
        """Reset per-turn budget. Caches stay warm across turns."""
        self.budget.reset_turn()

    # ── Health judge ──────────────────────────────────────────────────────

    def health(
        self, command: str, recent_output: str, elapsed: str,
        output_changed: bool = True, stall_count: int = 0,
    ) -> dict:
        """Evaluate whether a long-running command is healthy."""
        if self.budget.exhausted:
            if stall_count >= 3:
                return {"kill": True, "reason": "Output stalled and health check unavailable"}
            return {"kill": False}

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

    # ── Regex judge ───────────────────────────────────────────────────────

    def regex_confirm(
        self, category: str, matched_text: str, context: str = "",
    ) -> bool:
        """Confirm a regex match using LLM. Returns True if match is real."""
        cache_key = self._make_cache_key(category, matched_text)
        if cache_key in self._regex_cache:
            self.budget.total_saved_by_cache += 1
            return self._regex_cache[cache_key]

        if self.budget.exhausted:
            return True  # Trust the regex on budget exhaustion (conservative)

        if not self.budget.consume():
            return True

        result = self._regex_judge_ask(category, matched_text, context)
        self._regex_cache[cache_key] = result
        if not result:
            logger.info("RegexJudge rejected [%s]: %s", category, matched_text[:80])
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_cache_key(category: str, matched_text: str) -> str:
        content_hash = hashlib.md5(matched_text.strip().encode()).hexdigest()[:12]
        return f"{category}:{content_hash}"

    def _regex_judge_ask(self, category: str, matched_text: str, context: str) -> bool:
        """Single LLM call to confirm/reject a regex match."""
        prompt = _REGEX_JUDGE_PROMPTS.get(category, "Is this regex match significant?\n")
        user_content = f"{prompt}\n--- Matched content ---\n{matched_text[:500]}\n"
        if context:
            user_content += f"\n--- Surrounding context ---\n{context[:800]}\n"
        user_content += '\nReply ONLY: {"real": true} or {"real": false}'

        data = self._call_and_parse(user_content, default={"real": True})
        return bool(data.get("real", True))

    def _call_and_parse(self, prompt: str, default: dict | list) -> dict | list:
        """Make a single LLM call and parse JSON from response."""
        text = self._call(prompt)
        if not text:
            return default
        result = self._parse_json(text)
        return result if result else default

    def _call(self, prompt: str) -> str:
        """Dispatch LLM call through provider. Must be set before use."""
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
    def _parse_json(text: str) -> dict:
        """Extract JSON from LLM response text."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {}
