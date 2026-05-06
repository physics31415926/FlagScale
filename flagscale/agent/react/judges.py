"""LLM-based judge logic for ReactAgent — health, result, skill, complexity, and regex judges."""

import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class JudgesMixin:
    """Mixin providing LLM judge capabilities."""

    # ── Result Judge annotation dedup ───────────────────────────────────

    @staticmethod
    def _annotations_match(old, new):
        """Return True if two annotation lists are semantically identical."""
        if not old and not new:
            return True
        if len(old) != len(new):
            return False
        return set(a.strip() for a in old) == set(a.strip() for a in new)

    def _dedup_annotations(self, annotations):
        """Return annotations only if they differ from the last seen set."""
        if not annotations:
            return []
        if self._annotations_match(self._last_result_annotations, annotations):
            return []
        self._last_result_annotations = list(annotations)
        return annotations

    # ── Unified command health judge ─────────────────────────────────────

    _HEALTH_JUDGE_PROMPT = (
        "You are monitoring a running shell command. Analyze its status and decide "
        "whether it should continue or be terminated.\n\n"
        "Command: {command}\n"
        "Total elapsed: {elapsed}\n"
        "Output changed since last check: {output_changed}\n"
        "Consecutive checks with no output change: {stall_count}\n"
        "Recent output:\n{output}\n\n"
        "Phase-aware monitoring — adapt check frequency to the command's lifecycle stage:\n"
        "- STARTUP (no output yet, imports loading, initializing): check frequently (10-30s). "
        "Early failures are common.\n"
        "- LOADING (model weights loading, data downloading, progress bars advancing): "
        "moderate (30-60s).\n"
        "- STABLE (training iterations running, loss printing regularly): "
        "relaxed (120-300s).\n"
        "- ANOMALY (errors in output, repeated failures, output stalled unexpectedly): "
        "check soon (10-15s) or kill.\n\n"
        "Key judgment rules:\n"
        "- If output is actively progressing (new lines, advancing percentages): healthy, "
        "adjust interval to phase.\n"
        "- If output has stalled but the operation is known to be slow (large compile, "
        "decompression): allow more time.\n"
        "- If the command contains embedded sleep/wait and produces no output: you CANNOT "
        "verify whether the process being waited for is still alive. After a reasonable "
        "initial wait, KILL the command so the agent can check external state (process "
        "liveness, GPU status, logs) with its full tool set, then decide whether to retry.\n"
        "- If you see repeated errors, network failures, or crash signatures: kill immediately.\n"
        "- Do NOT let a silent command run indefinitely just because 'it might be working.' "
        "When in doubt, kill early — the agent can always re-check and retry.\n\n"
        "Reply with ONLY a JSON object: "
        "{{\"kill\": true/false, \"reason\": \"...\" or \"\", "
        "\"next_check_seconds\": <integer 10-300>}}\n\n"
        "If everything looks normal and healthy, set reason to empty string. "
        "Only provide a reason when there is something noteworthy — an issue, "
        "a phase transition, or a kill decision."
    )

    def _health_judge(self, command: str, recent_output: str, elapsed: str,
                      output_changed: bool = True, stall_count: int = 0) -> dict:
        """Ask LLM to evaluate whether a long-running command is healthy."""
        prompt = self._HEALTH_JUDGE_PROMPT.format(
            command=command, elapsed=elapsed,
            output=recent_output[-2000:],
            output_changed="yes" if output_changed else "no",
            stall_count=stall_count,
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        if stall_count >= 3:
            return {"kill": True, "reason": "Output stalled and health check unavailable"}
        return {"kill": False}

    # ── Shell result judge (LLM-based output analysis) ─────────────────

    _RESULT_JUDGE_PROMPT = (
        "You are analyzing the output of a shell command run by an AI infrastructure agent.\n\n"
        "Command: {command}\n"
        "Duration: {elapsed:.0f}s\n"
        "Output (last 3000 chars):\n{output}\n\n"
        "Identify issues and provide SHORT, actionable annotations. Consider:\n"
        "- Non-zero exit code or error messages → identify root cause\n"
        "- CUDA/cuDNN/driver version conflicts → give specific diagnosis commands\n"
        "- Network errors (connection refused, timeout, DNS) → suggest proxy or retry\n"
        "- Download failures → suggest resume with wget -c or curl -C -\n"
        "- PyTorch/CUDA incompatibility → suggest version check commands\n"
        "- Inefficient patterns (sleep+tail for monitoring) → suggest find_latest_log or timeout+tail -f\n"
        "- Log searching with find/ls -R/ls -lt → suggest find_latest_log tool or workspace_experiment list\n"
        "- Training launch (flagscale/torchrun/deepspeed) → remind to verify GPU utilization and logs\n"
        "- Package install success (pip/conda) → remind to verify runtime compatibility\n"
        "- pip upgraded/downgraded a critical package (torch, numpy, etc.) → WARN that this may break CUDA compatibility\n"
        "- Long duration (>2min) for simple commands → flag as unexpected\n"
        "- OOM (out of memory) → suggest reducing batch size, enabling gradient checkpointing, or adjusting parallelism\n"
        "- NCCL errors → suggest checking network config, NCCL env vars, and multi-node connectivity\n"
        "- Training output showing ce_loss or lm_loss near ln(vocab_size) (10.4, 10.8, 11.1, 11.8) → WARN: loss indicates random output, check weight loading\n"
        "- Config file edit containing path values → remind to verify paths exist before launching\n"
        "- Reading/grepping code from a different workspace than the current one (e.g., command references /workspace/X/ but agent works in /workspace/Y/) → WARN: source code provenance mismatch, verify you're reading the actually installed code\n"
        "- cp -r from another environment's site-packages → WARN: never copy packages between environments, use pip install\n"
        "- Checkpoint conversion output showing 'missed' or 'skipped' or 'unexpected' keys → WARN: audit the FULL list of missed/skipped keys by grouping them by top-level prefix. Do not assume they are all harmless based on a partial sample\n"
        "- Checkpoint saved to disk (torch.save, save_checkpoint) without a reload verification → WARN: verify saved checkpoint by reloading and checking key count, parameter shapes, and total parameter count match expectations\n"
        "- Training log/output showing crash, error, or exitcode!=0 (e.g., tail/cat/grep of a train.log showing Traceback, RuntimeError, exitcode=1) → WARN: update the experiment via workspace_experiment with the failure reason before debugging or relaunching\n"
        "- Training log showing successful completion (all steps done, checkpoint saved) → remind to update experiment entry with final metrics and reflection\n\n"
        "Reply with ONLY a JSON object:\n"
        '  {{"annotations": ["annotation1", "annotation2"], "severity": "info|warning|error"}}\n'
        "If no issues: {{\"annotations\": [], \"severity\": \"info\"}}"
    )

    def _result_judge(self, command: str, result: str, elapsed: float) -> list:
        """Use LLM to analyze shell output and return annotations."""
        prompt = self._RESULT_JUDGE_PROMPT.format(
            command=command,
            elapsed=elapsed,
            output=result[-3000:],
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return data.get("annotations", [])
        except Exception:
            pass
        return []

    # ── Skill judge (LLM-based skill matching) ─────────────────────────

    _SKILL_JUDGE_PROMPT = (
        "You are deciding which skill (if any) to load for an AI infrastructure agent.\n\n"
        "User request: {user_input}\n"
        "Conversation context: {conversation_context}\n"
        "Available skills:\n{skills_list}\n"
        "Already loaded: {loaded}\n\n"
        "Rules:\n"
        "- Only suggest a skill if it's clearly relevant to the user's request\n"
        "- If the user explicitly names a skill or task that maps to one, suggest it\n"
        "- If the request is ambiguous or general, suggest nothing\n"
        "- Never suggest a skill that's already loaded\n"
        "- For training tasks: suggest 'train-run'\n"
        "- For model porting/migration: suggest 'model-porter'\n"
        "- For environment setup: suggest 'env-setup'\n"
        "- For monitoring: suggest 'train-monitor'\n\n"
        "Reply with ONLY a JSON object:\n"
        '  {{"skills": ["skill-name"]}} or {{"skills": []}}'
    )

    def _get_recent_conversation_context(self) -> str:
        """Extract recent user/assistant messages for context in judge prompts."""
        recent = []
        for msg in self.history.messages[-6:]:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                recent.append(f"{role}: {content[:150]}")
        return "\n".join(recent) if recent else "(none)"

    def _skill_judge(self, user_input: str) -> list:
        """Ask LLM which skill to auto-load for the user's request."""
        skills = self.skill_manager.list_skills()
        if not skills:
            return []
        valid_names = {s['name'] for s in skills}
        skills_list = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
        loaded = ", ".join(self._loaded_skills) if self._loaded_skills else "(none)"

        prompt = self._SKILL_JUDGE_PROMPT.format(
            user_input=user_input[:500],
            conversation_context=self._get_recent_conversation_context(),
            skills_list=skills_list,
            loaded=loaded,
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                # Support both {"skill": "name"} and {"skills": ["name", ...]}
                skills_result = data.get("skills")
                if isinstance(skills_result, list):
                    return [s for s in skills_result
                            if s and s in valid_names and s not in self._loaded_skills]
                skill = data.get("skill")
                if skill and skill in valid_names and skill not in self._loaded_skills:
                    return [skill]
        except Exception:
            pass
        return []

    # ── Complexity judge ────────────────────────────────────────────────

    _COMPLEXITY_JUDGE_PROMPT = (
        "You are evaluating whether a user request requires a structured task plan.\n\n"
        "User request: {user_input}\n"
        "Active plan exists: {has_plan}\n"
        "Session memory context: {memory_context}\n\n"
        "A task needs planning when:\n"
        "- It involves 3+ distinct sequential steps (install -> configure -> run -> verify)\n"
        "- Steps have dependencies (can't train before data is ready)\n"
        "- It will take multiple tool calls across different domains (download, config, shell)\n"
        "- Failure at one step requires knowing what was already done\n\n"
        "A task does NOT need planning when:\n"
        "- Simple question or lookup\n"
        "- Single command execution\n"
        "- Continuing an existing plan (plan already exists)\n"
        "- Quick fix or small edit\n"
        "- User is asking a follow-up question in an ongoing conversation\n\n"
        "Reply with ONLY a JSON object:\n"
        '  {{"needs_plan": true/false, "reason": "one-line explanation"}}'
    )

    def _complexity_judge(self, user_input: str) -> dict:
        """Ask LLM whether the user request warrants a task plan."""
        if not self.config.auto_plan:
            return {"needs_plan": False}
        if self.task_plan.get_active():
            return {"needs_plan": False, "reason": "active plan exists"}

        memory_context = self._build_memory_context()
        prompt = self._COMPLEXITY_JUDGE_PROMPT.format(
            user_input=user_input,
            has_plan="no",
            memory_context=memory_context[:500] if memory_context else "(none)",
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return {"needs_plan": False}

    # ── Regex Judge — LLM confirmation for regex matches ──────────────────

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

    def _init_regex_judge(self):
        """Initialize regex judge state. Call from __init__."""
        self._regex_judge_cache = {}
        self._regex_judge_stats = {"hits": 0, "misses": 0, "confirms": 0, "rejects": 0}

    def _regex_judge_confirm(self, category: str, matched_text: str, context: str = "") -> bool:
        """Confirm a regex match using LLM. Returns True if match is real.

        Args:
            category: One of the keys in _REGEX_JUDGE_PROMPTS
            matched_text: The text that matched the regex
            context: Surrounding lines (5 lines before/after) for disambiguation
        """
        cache_key = self._regex_judge_cache_key(category, matched_text)
        if cache_key in self._regex_judge_cache:
            self._regex_judge_stats["hits"] += 1
            return self._regex_judge_cache[cache_key]

        self._regex_judge_stats["misses"] += 1
        result = self._regex_judge_ask(category, matched_text, context)
        self._regex_judge_cache[cache_key] = result

        if result:
            self._regex_judge_stats["confirms"] += 1
        else:
            self._regex_judge_stats["rejects"] += 1
            logger.info("RegexJudge rejected [%s]: %s", category, matched_text[:80])

        return result

    def _regex_judge_cache_key(self, category: str, matched_text: str) -> str:
        content_hash = hashlib.md5(matched_text.strip().encode()).hexdigest()[:12]
        return f"{category}:{content_hash}"

    def _regex_judge_ask(self, category: str, matched_text: str, context: str) -> bool:
        """Single LLM call to confirm/reject a regex match."""
        prompt_intro = self._REGEX_JUDGE_PROMPTS.get(category, "Is this regex match significant?\n")
        user_content = f"{prompt_intro}\n--- Matched content ---\n{matched_text[:500]}\n"
        if context:
            user_content += f"\n--- Surrounding context ---\n{context[:800]}\n"
        user_content += '\nReply ONLY: {"real": true} or {"real": false}'

        try:
            messages = [{"role": "user", "content": user_content}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return bool(data.get("real", True))
        except Exception as e:
            logger.warning("RegexJudge LLM call failed: %s", e)
        # On failure, trust the regex (conservative)
        return True

    @property
    def regex_judge_stats(self):
        return dict(self._regex_judge_stats)
