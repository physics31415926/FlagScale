"""ReAct agent — the core loop."""

import atexit
import json
import logging
import os
import re
import signal
import shlex
import sys
import time
import uuid

import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle

from flagscale.agent.react import display
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.history import HistoryManager, COMPACTION_NOTICE
from flagscale.agent.react.logger import setup_logging
from flagscale.agent.react.providers import get_provider
from flagscale.agent.react.retry import retry_with_backoff
from flagscale.agent.react.session import save_session, load_session, list_sessions
from flagscale.agent.react.skills import SkillManager
from flagscale.agent.react.tools import ToolRegistry
from flagscale.agent.react.tools.edit_file import EditFileTool
from flagscale.agent.react.tools.load_skill import LoadSkillTool
from flagscale.agent.react.tools.read_file import ReadFileTool
from flagscale.agent.react.tools.shell import ShellTool
from flagscale.agent.react.tools.write_file import WriteFileTool
from flagscale.agent.react.tools.web_fetch import WebFetchTool
from flagscale.agent.react.tools.find_log import FindLatestLogTool
from flagscale.agent.react.tools.parse_metrics import ParseTrainingMetricsTool
from flagscale.agent.react.tools.workspace_manager import WorkspaceManager
from flagscale.agent.react.tools.workspace_current import WorkspaceCurrentTool
from flagscale.agent.react.tools.workspace_experiment import WorkspaceExperimentTool
from flagscale.agent.react.tools.workspace_hardware import WorkspaceHardwareTool
from flagscale.agent.react.memory import SessionMemory
from flagscale.agent.react.tools.memory_write import MemoryWriteTool
from flagscale.agent.react.tools.memory_read import MemoryReadTool
from flagscale.agent.react.tools.memory_list import MemoryListTool
from flagscale.agent.react.plan import TaskPlan
from flagscale.agent.react.tools.monitor import MonitorTool
from flagscale.agent.react.tools.plan_create import PlanCreateTool
from flagscale.agent.react.tools.plan_update import PlanUpdateTool
from flagscale.agent.react.tools.plan_status import PlanStatusTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CORE = """You are FlagScale Agent, an AI infrastructure expert specialized in large model training with FlagScale. You execute, not just explain.

Tools: read_file, write_file, edit_file, shell, web_fetch, load_skill, memory_write, memory_read, memory_list, find_latest_log, parse_training_metrics, monitor, workspace_current, workspace_experiment, workspace_hardware, plan_create, plan_update, plan_status

Skills available:
{skills}

To activate a skill, call load_skill with the skill name. When a user asks what you can do, list ALL available skills above.

Working directory: {cwd}

## Core Principles

1. Context First: review memories/workspace/plan before acting. Don't re-discover what you already have.
2. Understand then Act: for complex tasks, read source code deeply and build understanding before implementing. Simple tasks with clear intent can be executed immediately. Quality of understanding determines quality of execution.
3. Transparent Execution: show findings, explain approach (1-2 sentences), report outcomes. Transparent ≠ verbose.
4. Parallel Execution: run independent commands simultaneously.
5. Use `monitor` for Waiting: use the `monitor` tool instead of shell+sleep loops. It polls locally without LLM calls. Examples: `monitor(file="log.txt", target_step=100)`, `monitor(command="nvidia-smi ...", duration=300, success_pattern="[4-7][0-9]{{4}}")`.
6. Know When to Ask vs Act: ASK when ambiguous/destructive/unclear. ACT when task is clear from context.
7. Follow explicit instructions exactly. If you disagree, state concern and ask — don't silently override.
8. Proactive Problem Detection: flag issues immediately. Fail-fast with pre-checks. After 2nd consecutive failure of same category, STOP and do systematic audit.
9. Plan Complex Work: multi-step tasks need plans. Update progress. Replan when things go wrong.

**Workspace & storage**: Load `workspace-layout` skill before downloading models/data, creating envs, or launching training. Workspace root = FlagScale's parent. Conda envs use `--prefix <root>/envs/<name>`.

**Reproduction vs Verification**: "Reproduce" = STRICT (immutable params: model arch, tokenizer, optimizer, loss, data pipeline). "Verify" = QUICK (confirm pipeline runs). If ambiguous, ASK.

## Shell Command Essentials

- Use `conda run --prefix <env_path> <command>`, never `conda activate`. Never install into base env.
- Never `find /` — scope to working directory. Exclude `*/envs/*`, `*site-packages*`, `*__pycache__*`.
- Use `read_file` to read source code, not `sed -n` or `cat`.
- Process lifecycle: kill → verify dead (`pgrep`) → clean → relaunch.
- FlagScale Launcher: `flagscale train <model> --config <config>`. Stop with `--stop`. Dryrun with `--dryrun`.
- Network errors: STOP and tell user to configure proxy.

## Auto Mode

End responses with `[TASK_COMPLETE]` (done) or `[NEED_USER_INPUT]` (blocked). If neither, system uses LLM judge.

## Language & Identity

Match user's language. You are FlagScale Agent — never call yourself Claude, GPT, or any other AI name.

{plan_context}
{memory_context}
{workspace_context}
{situational_context}
{optional_sections}
{skill_context}"""

SYSTEM_PROMPT_OPTIONAL = {
    "planning": """## Planning Discipline

- Separate analysis from action: read and understand the relevant code deeply before installing/implementing
- Deep reading is productive work: reading 20+ source files to understand architecture is expected and encouraged for complex tasks
- **Parallelism is a binding decision**: once determined, it constrains ALL subsequent steps. Fix failures to match decided parallelism, don't change parallelism to work around failures.""",

    "memory_rules": """## Memory vs Workspace

Two persistence mechanisms:
- **workspace**: current task state (workspace_current), experiment registry (workspace_experiment), hardware info (workspace_hardware)
- **memory**: persistent knowledge across sessions — env quirks, version constraints, user preferences, findings that took effort to derive

Rules:
- Experiment records → workspace_experiment
- Current task state → workspace_current
- Discovered version constraints, user preferences, env locations → memory
- **Memory is a claim, not a fact**: before acting on stored conclusions, re-verify the underlying evidence

Proactive memory:
- After unexpected failures requiring workarounds → memorize if a future session would hit the same issue
- After discovering env-specific facts through trial-and-error → memorize them
- Before writing a new memory, check if related memories exist. If contradicts, use 'supersedes' to delete old key""",

    "experiment": """## Experiment Lifecycle (MANDATORY)

**HARD GATE: Do NOT launch any training run without first creating the experiment entry via workspace_experiment.**

Before launching:
1. Create experiment: workspace_experiment(action='create', name='...', purpose='...', hypothesis='...', config={{...}}, dir='...')
2. Set as current: workspace_current(action='update', current_experiment='...', status='running')

After each attempt:
- workspace_experiment(action='add_attempt', name='...', change='...', result='...')

When done:
- workspace_experiment(action='finalize', name='...', status='failed|completed', root_cause='...', learnings=[...])
- workspace_current(action='update', status='blocked|completed', blockers=[...], next_steps=[...])""",

    "knowledge": """## Knowledge Caching

Check <context-summary> tags before re-reading files — they contain conclusions from compacted context.""",

    "decision": """## Decision Discipline

List ALL constraints before choosing an approach. Never flip between approaches more than twice (A→B→A = stop and ask user). When debugging, isolate ONE variable at a time.

## Diagnose Root Causes

Maximum 2 fix attempts for the same error category. After 2 failures, stop and do a systematic audit or try a fundamentally different approach. Before applying any fix, state the root cause hypothesis in one sentence.""",

    "porting": """## Model Porting Tasks

Porting means implementing the model IN Megatron-LM-FL / TransformerEngine-FL to leverage Megatron's parallelism, optimized kernels, and distributed training infrastructure. Wrapping the original model with a launcher is not porting.

Load the `model-porter` skill BEFORE writing any code. It has mandatory gates that must be passed sequentially.

**ENVIRONMENT COMPATIBILITY PRE-CHECK**: FlagScale wraps Megatron-LM-FL, but they can drift out of sync. If you hit import errors or missing APIs, check FlagScale's install docs for the pinned Megatron-LM-FL tag and roll back to it. Decide early: FlagScale wrapper OR direct torchrun.

**CRITICAL EXECUTION ORDER**: Data pipeline MUST be implemented and verified BEFORE model code. Order: get_batch → dataset → model code → training script. This is the #1 source of migration failures.

For parallelism selection/debugging, data pipelines under parallelism, attention under TP, or OOM/NCCL/hang issues, load the `parallel-strategy` skill.""",

    "user_commands": """## User commands

Users can type these slash commands directly:
- `/mode confirm` — risky shell commands require user confirmation (default)
- `/mode auto` — fully autonomous: no confirmations, auto-continues until task complete
- `/memory list` — show all memory entries
- `/memory clear [type]` — clear memory entries
- `/memory delete <key>` — delete a specific memory entry
- `/skill <name>` — load a skill manually
- `/file <path>` — add a file to context
- `/save [path]` — save conversation to file
- `/plan` — show current task plan status
- `/plan list` — list all plans
- `/plan abandon` — abandon the current plan
- `/reload` — reload skills and config
- `/quit` — exit the agent""",
}

# Keep old SYSTEM_PROMPT as alias for backward compatibility in tests
SYSTEM_PROMPT = SYSTEM_PROMPT_CORE


def _is_tool_result_msg(msg):
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


class ReactAgent:
    """A ReAct agent with streaming, history management, and parallel tool execution."""

    def __init__(self, config: AgentConfig):
        setup_logging()
        self.config = config
        self.skill_manager = SkillManager(config.skill_dirs)
        self.tool_registry = ToolRegistry()

        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(EditFileTool())
        self.tool_registry.register(
            ShellTool(
                remind_interval=config.shell_remind_interval,
                check_dangerous=config.dangerous_commands_check,
                require_confirm=config.confirm_commands,
                env=config.shell_env,
                health_judge_fn=self._health_judge,
            )
        )
        self.tool_registry.register(LoadSkillTool(self.skill_manager))
        self.tool_registry.register(WebFetchTool(proxies=self._build_proxies()))
        self.tool_registry.register(FindLatestLogTool())
        self.tool_registry.register(ParseTrainingMetricsTool())
        self.tool_registry.register(MonitorTool())

        # Workspace manager — split into current.yaml + per-experiment files + hardware.yaml
        workspace_dir = os.path.join(Path.home(), ".flagscale", "workspace_state")
        self._workspace_manager = WorkspaceManager(workspace_dir)
        self.tool_registry.register(WorkspaceCurrentTool(self._workspace_manager))
        self.tool_registry.register(WorkspaceExperimentTool(self._workspace_manager))
        self.tool_registry.register(WorkspaceHardwareTool(self._workspace_manager))
        self._load_plugin_tools()

        memory_dir = os.path.join(Path.home(), ".flagscale", "agent_memory")
        self._session_id = uuid.uuid4().hex[:8]
        self.session_memory = SessionMemory(memory_dir, config.memory_ttl_days)
        self.tool_registry.register(MemoryWriteTool(self.session_memory, self._session_id, workspace_manager=self._workspace_manager))
        self.tool_registry.register(MemoryReadTool(self.session_memory))
        self.tool_registry.register(MemoryListTool(self.session_memory))

        plan_dir = os.path.join(Path.home(), ".flagscale", "plans")
        self.task_plan = TaskPlan(plan_dir)
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        self.tool_registry.register(PlanUpdateTool(self.task_plan))
        self.tool_registry.register(PlanStatusTool(self.task_plan))

        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = get_provider(config.provider, config.model, config.api_key, config.base_url, config.max_output_tokens)

        self.history = HistoryManager(max_context_tokens=config.max_context_tokens)
        self.history.set_summarizer(self._summarize_for_compaction)

        self._turn_count = 0
        self._session_start = time.time()
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._auto_turn_count = 0
        self._last_write_turn = 0
        self._loaded_skills = set()
        self._interrupted = False
        self._streaming_in_code_block = False
        self._recent_iters = []
        self._last_result_annotations = []
        self._consecutive_train_failures = 0
        self._last_train_failure_reasons = []
        self._error_pattern_history = []  # Track error patterns for smart escalation
        self._kill_retry_timestamps = []  # Track kill+relaunch cycles
        self._training_launch_timestamps = []  # Track all training launches for hang detection
        self._context_pressure_soft_warned = False
        self._context_pressure_hard_warned = False
        self._last_checkpoint_tokens = 0  # For progress checkpoint
        self._experiment_registered = False  # True after workspace_state Experiments section is written
        self._dry_run_passed = False  # True after a quick-test / dry-run training command succeeds
        self._last_tool_call = None  # (tool_name, cmd_or_key, was_error) for workaround detection
        self._seen_errors = set()  # Track unique error signatures for checkpoint_new_error
        # Enforcement mechanism state
        self._consecutive_reads = 0  # Progress gate: track read-only tool calls
        self._progress_gate_triggers = 0  # Progress gate: escalation counter
        self._reads_since_last_new_file = 0  # Progress gate: staleness detector
        self._last_unique_file_count = 0  # Progress gate: track discovery rate
        self._recent_shell_errors = []  # Progress gate: track repeated errors
        self._complex_task_no_plan = False  # Plan gate: True when complex task detected but no plan created
        self._pre_plan_tool_calls = 0  # Plan gate: tool calls before plan creation
        self._rereads_without_save = 0  # Re-read gate: consecutive re-reads without memory_write
        self._context_pressure_warned = False  # Context pressure: track if 75% warning shown
        self._last_tool_had_error = False  # Error-escalation gate: track if last tool errored
        self._root_cause_recorded_since_error = False  # Error-escalation gate: track if root cause recorded
        self._last_gate_warning = ""  # Gate dedup: don't inject same warning twice
        self._files_read_this_session = set()  # Reading depth gate: track files read
        self._files_written_this_session = set()  # Track files written for snapshot
        self._porting_mode = False  # Reading depth gate: True after model-porter skill loaded
        self._porting_path_confirmed = False  # Porting path gate: True after user confirms Mode B/C
        self._data_prep_mode = False  # Data pipeline gate: True after data-prep skill loaded
        self._data_pipeline_understood = False  # Data pipeline gate: True after pipeline comprehension persisted
        self._analysis_persisted = False  # Analysis persistence gate: True after analysis written
        self._verification_stage = "none"  # Verification ladder: none -> analysis -> init_ok -> forward_aligned -> backward_ok -> distributed_ok -> full_training
        self._last_compaction_count = 0  # Session resume gate: detect compaction events
        # Phase transition gate
        self._current_phase = "idle"  # idle -> analysis -> implementation -> verification -> done
        self._phase_tool_counts = {}  # {phase: count} — tools called in each phase
        # Reading quality gate — track coverage across 4 categories
        self._reading_categories = set()  # {"source_model", "megatron_base", "existing_impl", "checkpoint"}
        # Infinite loop detection — track recent tool calls
        self._recent_tool_calls = []  # [(tool_name, key_args), ...] last 10 calls
        self._tool_call_cache = {}  # {(tool_name, key_args): result} — cache within turn
        # New gate state (Phase 1-4)
        self._understanding_verified = False  # A1: True after verification questions answered
        self._component_plan_created = False  # A3: True after component isolation plan created
        self._failure_mode_analyzed = False  # B1: True after failure mode analysis done
        self._sanity_checks_passed = False  # B2: True after 4 sanity checks passed
        self._config_model_verified = False  # B4: True after config-model consistency verified
        self._env_verified = False  # C2: True after environment consistency verified
        self._component_integration_verified = False  # C4: True after per-component verification
        self._imports_verified = False  # A4: True after critical imports verified
        self._reference_comparison_planned = False  # A2: True after comparison strategy created

        # Token optimization: phase-based schema filtering
        from collections import deque
        self._last_tool_calls_deque = deque(maxlen=5)
        self._extra_tools_next_iter = set()
        self._turn_iteration_count = 0

        # Token optimization: skill lifecycle management
        self._active_skill_content = {}  # {skill_name: content_text}
        self._skill_load_iterations = {}  # {skill_name: iteration_when_loaded}
        self._total_iterations = 0
        self._training_started = False
        self._recently_referenced_skills = set()

        # Now refresh system prompt after all state is initialized
        self._refresh_system_prompt()
        atexit.register(self._atexit_hook)

    # ── Atexit safety net ───────────────────────────────────────────────

    def _atexit_hook(self):
        """Update workspace state on any exit path (safety net for abnormal exits)."""
        try:
            if self._session_output_tokens:
                self._auto_update_workspace_state()
                self._archive_session()
        except Exception:
            pass

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

    # ── Context compaction summarizer ────────────────────────────────────

    def _summarize_for_compaction(self, text: str) -> str:
        """Call LLM to summarize conversation segment being dropped during compaction.

        Preserves critical state that must survive compaction.
        """
        # Build structured state snapshot
        state_snapshot = []

        # Mode flags (critical for gate enforcement)
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

        # Error patterns
        if self._error_pattern_history:
            recent_errors = self._error_pattern_history[-3:]
            state_snapshot.append(f"Error patterns: {', '.join(recent_errors)}")

        # Verification stage
        if self._verification_stage != "none":
            state_snapshot.append(f"Verification stage: {self._verification_stage}")

        # Phase
        if self._current_phase != "idle":
            state_snapshot.append(f"Current phase: {self._current_phase}")

        # Reading coverage
        if self._reading_categories:
            state_snapshot.append(f"Reading categories covered: {', '.join(sorted(self._reading_categories))}")

        # Files read (top 10 most recent)
        if self._files_read_this_session:
            recent_files = list(self._files_read_this_session)[-10:]
            state_snapshot.append(f"Recent files read: {', '.join(recent_files)}")

        # Current experiment
        current_exp = self._workspace_manager.get_current_experiment()
        if current_exp:
            state_snapshot.append(f"Current experiment: {current_exp}")

        state_block = "\n".join(state_snapshot) if state_snapshot else "(no critical state)"

        # Call LLM with enhanced prompt
        messages = [
            {"role": "system", "content": "You are a concise summarizer. Output only the summary, no preamble."},
            {"role": "user", "content": f"{text}\n\n--- CRITICAL STATE (preserve in summary) ---\n{state_block}"},
        ]
        response = self.provider.chat(messages, tools=[])
        summary = response.get("content", "").strip()

        # Append state snapshot to ensure it's preserved
        return f"{summary}\n\n[State at compaction: {state_block}]"

    def _restore_state_from_compaction(self):
        """Restore enforcement state from compaction summary after context compaction."""
        # Find the <context-summary> message in history
        summary_content = None
        for msg in self.history.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and "<context-summary>" in content:
                    summary_content = content
                    break

        if not summary_content or "[State at compaction:" not in summary_content:
            logger.debug("No state snapshot found in compaction summary")
            return

        # Extract state block — use rfind for the closing bracket since content may contain ]
        try:
            start_marker = "[State at compaction: "
            start_idx = summary_content.index(start_marker) + len(start_marker)
            # Find the last ] after the start marker (state block ends with ])
            end_idx = summary_content.rfind("]", start_idx)
            if end_idx == -1:
                end_idx = len(summary_content)
            state_block = summary_content[start_idx:end_idx].strip()
        except (IndexError, ValueError):
            logger.warning("Failed to parse state block from compaction summary")
            return

        # Parse and restore state
        for line in state_block.split("\n"):
            line = line.strip()
            if not line or line == "(no critical state)":
                continue

            if line.startswith("Mode flags:"):
                flags = [f.strip() for f in line.split(":", 1)[1].strip().split(",")]
                if "porting" in flags:
                    self._porting_mode = True
                if "data_prep" in flags:
                    self._data_prep_mode = True
                if "analysis_persisted" in flags:
                    self._analysis_persisted = True
                if "path_confirmed" in flags:
                    self._porting_path_confirmed = True
                if "training_started" in flags:
                    self._training_started = True
                if "understanding_verified" in flags:
                    self._understanding_verified = True
                if "data_pipeline_understood" in flags:
                    self._data_pipeline_understood = True
                if "component_plan_created" in flags:
                    self._component_plan_created = True

            elif line.startswith("Error patterns:"):
                patterns = line.split(":", 1)[1].strip()
                if patterns:
                    self._error_pattern_history = [p.strip() for p in patterns.split(",")]

            elif line.startswith("Verification stage:"):
                stage = line.split(":", 1)[1].strip()
                if stage in self._VERIFICATION_STAGES:
                    self._verification_stage = stage

            elif line.startswith("Current phase:"):
                phase = line.split(":", 1)[1].strip()
                if phase in ("idle", "analysis", "implementation", "verification", "done"):
                    self._current_phase = phase

            elif line.startswith("Reading categories covered:"):
                cats = line.split(":", 1)[1].strip()
                if cats:
                    self._reading_categories = set(c.strip() for c in cats.split(","))

            elif line.startswith("Recent files read:"):
                files = line.split(":", 1)[1].strip()
                if files:
                    # Restore files (note: this is partial, only recent 10)
                    self._files_read_this_session.update(f.strip() for f in files.split(","))

        logger.info("Restored state from compaction: phase=%s, stage=%s, files=%d, categories=%d",
                    self._current_phase, self._verification_stage,
                    len(self._files_read_this_session), len(self._reading_categories))

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
        # If LLM unavailable and output stalled for 3+ checks, kill as safety net
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
        "Given the user's request and recent conversation context, decide which "
        "skills (if any) should be loaded to help the agent.\n\n"
        "User request: {user_input}\n\n"
        "Recent conversation context:\n{conversation_context}\n\n"
        "Available skills:\n{skills_list}\n\n"
        "Already loaded: {loaded_skills}\n\n"
        "Guidelines:\n"
        "- Only select skills that are clearly relevant to the request\n"
        "- Consider the conversation context — a follow-up question about training "
        "may need a training skill even if the current message is short\n"
        "- Do NOT load skills speculatively or for simple questions\n"
        "- If the user is asking about hardware/GPU topology, select topo-detect\n"
        "- If the user mentions environment setup, dependencies, or installation, select env-setup\n"
        "- If the user mentions model porting, architecture analysis, or checkpoint conversion, select model-porter\n"
        "- If the user mentions data preprocessing or tokenization, select data-prep\n"
        "- If the user mentions training configuration or YAML config, select train-config\n"
        "- If the user mentions parallelism strategy (TP/PP/DP/EP/CP), OOM debugging, data pipeline under parallelism, or attention under TP, select parallel-strategy\n"
        "- If the user mentions starting, stopping, or launching training, select train-run\n"
        "- If the user mentions monitoring training, checking loss, or training status, select train-monitor\n"
        "- If the user mentions reproducing results or baseline validation, select reproduce\n"
        "- If the user mentions precision alignment or numerical comparison, select precision-alignment\n\n"
        "Reply with ONLY a JSON object:\n"
        '  {{"skills": ["skill-name"]}}\n'
        "If no skill is relevant: {{\"skills\": []}}"
    )

    def _skill_judge(self, user_input: str) -> list:
        """Use LLM to decide which skills to load for a user request."""
        skills = self.skill_manager.list_skills()
        if not skills:
            return []
        available = [s for s in skills if s["name"] not in self._loaded_skills]
        if not available:
            return []

        skills_list = "\n".join(
            f'- {s["name"]}: {s["description"]}' for s in available
        )
        loaded = ", ".join(self._loaded_skills) if self._loaded_skills else "(none)"

        recent = []
        for m in self.history.messages[-6:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                line = content.strip().replace("\n", " ")
                if len(line) > 150:
                    line = line[:147] + "..."
                recent.append(f"{role}: {line}")
        conversation_context = "\n".join(recent) if recent else "(new conversation)"

        prompt = self._SKILL_JUDGE_PROMPT.format(
            user_input=user_input,
            skills_list=skills_list,
            loaded_skills=loaded,
            conversation_context=conversation_context,
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                names = data.get("skills", [])
                valid_names = {s["name"] for s in available}
                return [n for n in names if n in valid_names]
        except Exception:
            pass
        return []

    # ── Complexity judge ────────────────────────────────────────────────

    _COMPLEXITY_JUDGE_PROMPT = (
        "You are evaluating whether a user request needs a structured task plan.\n\n"
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

    # ── Plugin tools (P2-8) ──────────────────────────────────────────────

    def _load_plugin_tools(self):
        dirs = self.config.plugin_tool_dirs
        if not dirs:
            return
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(d, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        spec = json.load(f)
                    tool = _PluginShellTool(spec)
                    self.tool_registry.register(tool)
                    logger.info("Loaded plugin tool: %s from %s", tool.name, path)
                except Exception as e:
                    logger.warning("Failed to load plugin tool %s: %s", path, e)

    # ── System prompt ────────────────────────────────────────────────────

    # Phase-based tool schema filtering: only send relevant tool schemas to reduce tokens.
    _PHASE_TOOL_SETS = {
        "monitoring": {"monitor", "shell", "read_file", "parse_training_metrics"},
        "planning": {"plan_create", "plan_update", "plan_status", "read_file", "shell",
                     "memory_read", "memory_write", "load_skill", "workspace_current"},
        "training": {"shell", "monitor", "read_file", "find_latest_log",
                     "parse_training_metrics", "workspace_experiment", "workspace_current",
                     "memory_write"},
        "default": None,  # None = all tools
    }
    _CORE_TOOLS = {"shell", "read_file"}
    _SENTINEL = object()  # Default sentinel for _refresh_system_prompt kwargs

    # Situational prompt sections: only included when relevant context is active.
    # Maps group name -> set of ## section headers to include when that group is active.
    _SITUATIONAL_SECTIONS = {
        "env_setup": """
## Dependency Installation (env-setup context)

Critical rules for environment management:
- For packages that pull PyTorch/CUDA deps (flash-attn, deepspeed, apex): use pip install --no-deps
- After ANY large pip install, verify: python -c "import torch; print(torch.__version__, torch.version.cuda)"
- FL-customized dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention) are ALL mandatory
- NEVER copy packages between conda envs using cp -r from site-packages (bypasses pip metadata)
- Always install via pip install (from wheel, PyPI, or source)

Load env-setup skill for full dependency resolution protocol.
""",
        "model_porting": """
## Source Code Provenance (model-porting context)

When reading source code to understand an installed package:
- Use conda run -n <env> python -c "import <pkg>; print(<pkg>.__file__)" to find actual installed location
- If package is editable install (pip install -e), verify editable path matches your current workspace
- NEVER read code from different directory than what's installed
- Always run pip show and python -c "import ..." inside TARGET conda environment

**Workspace isolation**: NEVER do editable installs from another workspace's code tree. Clone into your workspace first, then editable-install from local clone.

Load model-porter skill for full porting workflow.
""",
        "training": """
## Experiment Registry (training context)

Every experiment MUST be recorded via workspace_experiment. This is the knowledge base that prevents repeating mistakes.

**HARD GATE: Do NOT launch any training run without first creating the experiment entry.**

One experiment, one directory. Never reuse experiment directories for different purposes.

### What counts as a new experiment

- Produced ≥1 step of training metrics → real experiment, next change = new version
- Changed meaningful parameter (LR, TP/PP, batch size, data, model code) → new experiment
- Launch failed before metrics (import error, path error, config typo) → failed launch attempt, NOT new experiment. Record in current entry's Result field, fix, retry same version.
- Training crashed after producing metrics, restarting with same config → still same experiment

Required fields: Purpose, Hypothesis, Config, Dir, Launch notes, Result, Reflection, Next

Lifecycle:
1. BEFORE launching: write Purpose, Hypothesis, Config, Dir (status=running)
2. If launch fails before metrics: add to Launch notes, fix, retry (don't create new entry)
3. AFTER completion/failure: fill Result, Reflection, Next (update status) IMMEDIATELY
4. When starting next experiment: reference previous Reflection

Load train-run skill for full training launch protocol.
""",
        "general": """
## General Operational Discipline

- List ALL constraints before choosing approach
- Never flip between approaches >2 times (A→B→A = stop and ask)
- When debugging, isolate ONE variable at a time
- Before applying fix, state root cause hypothesis in one sentence
- Check plan_status at start of each turn
- Mark plan steps done as you go

Load ops-discipline skill for full diagnosis protocol.
"""
    }

    _SITUATIONAL_GROUPS = {
        "env_setup": {
            "Dependency Installation (env-setup context)",
        },
        "model_porting": {
            "Source Code Provenance (model-porting context)",
        },
        "training": {
            "Experiment Registry (training context)",
        },
    }

    # Sections always included regardless of context
    _ALWAYS_SECTIONS = {
        "Core Principles", "Planning Discipline", "Memory vs Workspace",
        "Experiment Lifecycle (MANDATORY)", "Knowledge Caching", "Task Planning",
        "Decision Discipline", "Diagnose Root Causes",
        "Model Porting Tasks", "Fast Validation Principle",
        "Design Before Writing", "Investigation Discipline",
        "Verification Before Investment",
        "Context Budget Awareness", "Graceful Degradation",
        "Training Health Quick-Checks", "Efficient Monitoring",
        "Monitoring Strategy During Training",
        "Language", "Identity", "User Commands", "Shell Command Essentials",
    }

    def _get_situational_context(self):
        """Determine which situational sections to include based on context."""
        sections_to_load = {"general"}  # Always include general

        # Based on loaded skills
        if "model-porter" in self._loaded_skills or "precision-alignment" in self._loaded_skills or self._porting_mode:
            sections_to_load.add("model_porting")
        if "train-run" in self._loaded_skills or "train-monitor" in self._loaded_skills:
            sections_to_load.add("training")
        if "env-setup" in self._loaded_skills or "ops-discipline" in self._loaded_skills:
            sections_to_load.add("env_setup")

        # Based on workspace task
        try:
            current = self._workspace_manager.read_current()
            task = current.get("task", "").lower()
            if "env" in task or "install" in task or "setup" in task:
                sections_to_load.add("env_setup")
            if "port" in task or "migrat" in task or "implement" in task:
                sections_to_load.add("model_porting")
            if "train" in task or "experiment" in task:
                sections_to_load.add("training")
        except Exception:
            pass

        # Based on recent tool calls (last 10)
        if hasattr(self, '_recent_tool_calls'):
            for tool_name, *_ in self._recent_tool_calls[-10:]:
                if tool_name == "shell":
                    sections_to_load.add("env_setup")
                if tool_name in ("workspace_experiment", "find_latest_log", "parse_training_metrics"):
                    sections_to_load.add("training")

        return sections_to_load

    def _detect_tool_phase(self) -> str:
        """Detect current activity phase from recent tool calls for schema filtering."""
        if not hasattr(self, '_last_tool_calls_deque') or not self._last_tool_calls_deque:
            return "default"
        recent = list(self._last_tool_calls_deque)
        if recent[-1] == "monitor":
            return "monitoring"
        if sum(1 for t in recent if t.startswith("plan_")) >= 2:
            return "planning"
        training_tools = {"workspace_experiment", "parse_training_metrics", "find_latest_log"}
        if sum(1 for t in recent if t in training_tools) >= 2:
            return "training"
        return "default"

    def _get_filtered_schemas(self, phase: str) -> list:
        """Get tool schemas filtered by current phase."""
        tool_names = self._PHASE_TOOL_SETS.get(phase)
        if tool_names is None:
            return self.tool_registry.to_schemas(self.provider.schema_format)
        # Include core tools + any extras requested by fallback
        active = tool_names | self._CORE_TOOLS | self._extra_tools_next_iter
        return self.tool_registry.to_schemas_filtered(self.provider.schema_format, active)

    def _get_optional_sections(self) -> list:
        """Determine which optional prompt sections to include based on current state."""
        sections = set()

        # First few iterations of a turn: include guidance sections
        if self._turn_iteration_count <= 3:
            sections.update(["planning", "memory_rules", "experiment"])

        # Based on active plan
        if hasattr(self, 'task_plan') and self.task_plan.get_active():
            sections.add("planning")

        # Based on recent phase
        phase = self._detect_tool_phase() if hasattr(self, '_last_tool_calls_deque') else "default"
        if phase == "planning":
            sections.update(["planning", "memory_rules"])
        elif phase == "training":
            sections.add("experiment")

        # Porting mode
        if getattr(self, '_porting_mode', False):
            sections.add("porting")

        # Decision discipline on errors
        if getattr(self, '_last_tool_had_error', False) or getattr(self, '_consecutive_train_failures', 0) >= 2:
            sections.add("decision")

        # Knowledge caching hint after compaction
        if getattr(self, '_last_compaction_count', 0) > 0:
            sections.add("knowledge")

        # User commands only on first iteration of a turn
        if self._turn_iteration_count <= 1:
            sections.add("user_commands")

        return sorted(sections)

    def _refresh_system_prompt(self, memory_context=_SENTINEL, plan_context=_SENTINEL, workspace_context=_SENTINEL):
        # When called with explicit contexts (from _inject_context), cache them.
        # When called without args (from skill unload, /reload), reuse cached values.
        if memory_context is not self._SENTINEL:
            self._last_memory_context = memory_context
        else:
            memory_context = getattr(self, '_last_memory_context', "")
        if plan_context is not self._SENTINEL:
            self._last_plan_context = plan_context
        else:
            plan_context = getattr(self, '_last_plan_context', "")
        if workspace_context is not self._SENTINEL:
            self._last_workspace_context = workspace_context
        else:
            workspace_context = getattr(self, '_last_workspace_context', "")
        skills = self.skill_manager.list_skills()
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
            if skills else "(no skills available)"
        )

        # Determine which situational sections to include
        sections_to_load = self._get_situational_context()

        # Build situational context
        situational_parts = []
        for section_name in sorted(sections_to_load):
            if section_name in self._SITUATIONAL_SECTIONS:
                situational_parts.append(self._SITUATIONAL_SECTIONS[section_name])

        situational_context = "\n".join(situational_parts)

        # Build optional sections (Layer 4: prompt tiering)
        optional_section_names = self._get_optional_sections()
        optional_parts = []
        for name in optional_section_names:
            if name in SYSTEM_PROMPT_OPTIONAL:
                optional_parts.append(SYSTEM_PROMPT_OPTIONAL[name])
        optional_sections = "\n\n".join(optional_parts)

        # Build skill context (Layer 5: skill lifecycle)
        skill_parts = []
        for skill_name, content in self._active_skill_content.items():
            skill_parts.append(f"## Active Skill: {skill_name}\n{content}")
        skill_context = "\n\n".join(skill_parts)

        # Format the prompt
        prompt = SYSTEM_PROMPT_CORE.format(
            skills=skills_text,
            cwd=os.getcwd(),
            memory_context=memory_context,
            plan_context=plan_context,
            workspace_context=workspace_context,
            situational_context=situational_context,
            optional_sections=optional_sections,
            skill_context=skill_context,
        )

        msgs = self.history.messages
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": prompt}
        else:
            self.history._messages.insert(0, {"role": "system", "content": prompt})
        self._system_prompt = prompt

    def _build_proxies(self):
        env = self.config.shell_env
        # Fall back to OS environment if not set in agent.yaml
        http = env.get("HTTP_PROXY") or env.get("http_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https = env.get("HTTPS_PROXY") or env.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        proxy = http or https
        if not proxy:
            return None
        return {"http": http or proxy, "https": https or proxy}

    def _reload_config(self):
        if self.config.reload():
            shell_tool = self.tool_registry.get("shell")
            shell_tool._env = self.config.shell_env
            web_fetch_tool = self.tool_registry.get("web_fetch")
            web_fetch_tool._proxies = self._build_proxies()

    def _build_memory_context(self):
        """Build memory context string from recent memories, with dynamic budget based on context pressure."""
        task = self._workspace_manager.get_current_task()
        # Dynamic budget: reduce memory injection when context is tight
        pressure = self.history.get_context_pressure()
        if pressure > 0.7:
            budget = 1000
        elif pressure > 0.5:
            budget = 2000
        else:
            budget = 4000
        notes = self.session_memory.recent(
            max_tokens=budget, task_filter=task,
            current_session_id=getattr(self, '_session_id', ''),
        )
        if not notes:
            return ""
        lines = []
        stale_keys = []
        stale_threshold = 14 * 86400
        now = time.time()
        for n in notes:
            task_tag = f" @{n.get('task', '')}" if n.get("task") else ""
            lines.append(f'[{n.get("type", "?")}:{n.get("key", "?")}]{task_tag} {n.get("content", "")}')
            age = now - n.get("created", 0)
            if age > stale_threshold:
                stale_keys.append(n.get("key", "?"))
        result = "<session-memory>\n" + "\n".join(lines) + "\n</session-memory>"
        if stale_keys:
            result += self._STALE_MEMORY_WARNING_TEMPLATE.format(
                count=len(stale_keys), days=14,
                keys=", ".join(stale_keys[:5]) + ("..." if len(stale_keys) > 5 else ""),
            )
        if self._turn_count >= 5:
            result += self._SESSION_MEMORY_REVIEW
        return result

    def _build_workspace_context(self):
        """Load current.yaml, hardware.yaml, and recent session history for system prompt injection."""
        parts = []

        # Current state — always inject
        current = self._workspace_manager.read_current()
        if current:
            import yaml as _yaml
            parts.append("## Current State\n" + _yaml.dump(current, allow_unicode=True, default_flow_style=False, sort_keys=False).strip())

        # Hardware — always inject
        hardware = self._workspace_manager.read_hardware()
        if hardware:
            import yaml as _yaml
            parts.append("## Hardware\n" + _yaml.dump(hardware, allow_unicode=True, default_flow_style=False).strip())

        # Recent session history — inject last 3 sessions for continuity
        recent_sessions = self._workspace_manager.get_recent_sessions(n=3)
        if recent_sessions:
            history_lines = []
            for s in recent_sessions:
                ts = s.get("timestamp", "?")
                task = s.get("task", "?")
                summary = s.get("summary", "")
                # Truncate each entry to keep context budget reasonable
                if len(summary) > 150:
                    summary = summary[:147] + "..."
                history_lines.append(f"- [{ts}] {task}: {summary}")
            parts.append("## Recent Sessions\n" + "\n".join(history_lines))

        if not parts:
            return ""

        return "<workspace-state>\n" + "\n\n".join(parts) + "\n</workspace-state>"

    def _inject_context(self, user_input):
        """Auto-inject session memory, plan, and workspace context into the system prompt."""
        memory_context = self._build_memory_context()

        plan_context = self.task_plan.context_for_prompt()

        workspace_context = self._build_workspace_context()

        complexity_hint = ""
        if not plan_context and self.config.auto_plan:
            judge_result = self._complexity_judge(user_input)
            if judge_result.get("needs_plan"):
                self._complex_task_no_plan = True
                self._pre_plan_tool_calls = 0
                complexity_hint = (
                    "\n<system-hint>This task is complex and REQUIRES a plan. "
                    "Take time to deeply read and understand the relevant source code, docs, and configs. "
                    "When you have enough understanding, call plan_create to organize your approach. "
                    "Do NOT start implementation without a plan.</system-hint>\n"
                )
                display.complexity_hint()

        plan_context = plan_context + complexity_hint if complexity_hint else plan_context

        # Session resume gate: use snapshot (preferred) or current.yaml for recovery
        resume_hint = ""
        compaction_count = getattr(self.history, 'compaction_count', 0)
        if self._turn_count <= 1 or compaction_count > self._last_compaction_count:
            self._last_compaction_count = compaction_count
            resume_hint = self._format_snapshot_as_resume()
        if resume_hint:
            plan_context = plan_context + resume_hint if plan_context else resume_hint

        self._refresh_system_prompt(memory_context=memory_context, plan_context=plan_context, workspace_context=workspace_context)

    def _handle_memory_command(self, user_input):
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /memory list | /memory clear [type] | /memory delete <key>")
            return
        sub = parts[1]
        if sub == "list":
            entries = self.session_memory.list_entries()
            if not entries:
                print("No memory entries.")
                return
            for e in entries:
                key = e.get("key", "?")
                mem_type = e.get("type", "?")
                content = e.get("content", "")
                if len(content) > 120:
                    content = content[:117] + "..."
                print(f"  [{mem_type}] \033[1m{key}\033[0m")
                print(f"          {content}")
        elif sub == "clear":
            if len(parts) >= 3:
                mem_type = parts[2]
                count = self.session_memory.clear_by_type(mem_type)
                print(f"Cleared {count} '{mem_type}' memory entries.")
            else:
                count = self.session_memory.clear()
                print(f"Cleared {count} memory entries.")
        elif sub == "delete":
            if len(parts) < 3:
                print("Usage: /memory delete <key>")
                return
            key = parts[2]
            if self.session_memory.delete(key):
                print(f"Deleted memory '{key}'.")
            else:
                print(f"No memory '{key}' found.")
        else:
            print("Usage: /memory list | /memory clear | /memory delete <key>")

    def _handle_mode_command(self, user_input):
        parts = user_input.split()
        if len(parts) < 2:
            print(f"Current mode: {self.config.mode}")
            print("Usage: /mode confirm | /mode auto")
            print("  confirm — risky commands require user confirmation (default)")
            print("  auto    — fully autonomous: no confirmations, auto-continues between turns")
            return
        new_mode = parts[1].lower()
        if new_mode not in ("confirm", "auto"):
            print(f"Unknown mode '{new_mode}'. Available: confirm, auto")
            return
        self.config.mode = new_mode
        if new_mode == "auto":
            self.config.confirm_commands = False
            shell_tool = self.tool_registry.get("shell")
            if shell_tool:
                shell_tool._require_confirm = False
            print(f"Mode: auto — fully autonomous (max {self.config.max_auto_turns} auto turns, Ctrl+C to stop).")
        else:
            self.config.confirm_commands = True
            shell_tool = self.tool_registry.get("shell")
            if shell_tool:
                shell_tool._require_confirm = True
            print("Mode: confirm — risky commands will require confirmation.")

    def _handle_plan_command(self, user_input):
        parts = user_input.split()
        sub = parts[1] if len(parts) >= 2 else "status"
        if sub == "status" or sub == "show":
            text = self.task_plan.summary()
            display.plan_summary(text)
        elif sub == "list":
            plans = self.task_plan.list_plans()
            if not plans:
                print("No plans.")
                return
            for p in plans:
                status_str = p["status"]
                done = p["done"]
                total = p["total"]
                print(f"  {p['id']}  {p['title']}  [{status_str}]  {done}/{total} steps")
        elif sub == "abandon":
            try:
                plan = self.task_plan.abandon(reason="user requested via /plan abandon")
                display.plan_abandoned(plan["title"])
            except ValueError as e:
                print(f"  {e}")
        elif sub == "clear":
            count = self.task_plan.clear_completed()
            print(f"Cleared {count} completed/abandoned plans.")
        else:
            print("Usage: /plan [status|list|abandon|clear]")

    def _check_proxy(self):
        """Warn if no proxy is configured and offer to set one."""
        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
        has_proxy = any(
            self.config.shell_env.get(k) or os.environ.get(k)
            for k in proxy_keys
        )
        if has_proxy:
            return
        print("\033[33m⚠  No HTTP proxy detected.\033[0m")
        print("  If your network requires a proxy, shell commands may fail to reach the internet.")
        try:
            answer = input("  Configure a proxy now? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer not in ("y", "yes"):
            return
        try:
            proxy_url = input("  Proxy URL (e.g. http://host:port): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not proxy_url:
            return
        # Update running config
        self.config.shell_env["HTTP_PROXY"] = proxy_url
        self.config.shell_env["HTTPS_PROXY"] = proxy_url
        # Update shell tool env
        shell_tool = self.tool_registry.get("shell")
        if shell_tool:
            shell_tool._env["HTTP_PROXY"] = proxy_url
            shell_tool._env["HTTPS_PROXY"] = proxy_url
        # Update web_fetch tool proxy
        web_fetch_tool = self.tool_registry.get("web_fetch")
        if web_fetch_tool:
            web_fetch_tool._proxies = self._build_proxies()
        # Persist to ~/.flagscale/agent.yaml
        config_path = os.path.join(Path.home(), ".flagscale", "agent.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        existing = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        shell_env = existing.get("shell_env", {})
        shell_env["HTTP_PROXY"] = proxy_url
        shell_env["HTTPS_PROXY"] = proxy_url
        existing["shell_env"] = shell_env
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False)
        print(f"  \033[32m✓\033[0m Proxy saved to {config_path}")
        # Test proxy connectivity
        self._test_proxy(proxy_url)

    def _test_proxy(self, proxy_url: str):
        """Test proxy connectivity by making a quick HTTP request."""
        import subprocess
        test_url = "https://www.google.com"
        print(f"  Testing proxy connectivity ({test_url})...", end=" ", flush=True)
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--proxy", proxy_url, "--connect-timeout", "10",
                 "--max-time", "15", test_url],
                capture_output=True, text=True, timeout=20,
            )
            code = result.stdout.strip()
            if code and code[0] in ("2", "3"):
                print(f"\033[32m✓\033[0m (HTTP {code})")
            elif code:
                print(f"\033[33m⚠\033[0m (HTTP {code} — proxy responded but may have issues)")
            else:
                stderr = result.stderr.strip()
                short_err = stderr.split("\n")[0][:80] if stderr else "no response"
                print(f"\033[31m✗\033[0m ({short_err})")
                print("  Proxy may be unreachable or misconfigured. You can reconfigure with /reload.")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print(f"\033[31m✗\033[0m (connection timed out)" if "Timeout" in type(e).__name__
                  else f"\033[31m✗\033[0m ({e})")
            print("  Proxy may be unreachable. You can reconfigure with /reload.")

    def _startup_hints(self):
        """Build extra banner lines showing available memory summaries."""
        hints = []
        mem_entries = self.session_memory.list_entries()
        if mem_entries:
            hints.append(f"Memory: {len(mem_entries)} entries (/memory list)")
            for e in mem_entries[:3]:
                content = e.get("content", "")
                if len(content) > 60:
                    content = content[:57] + "..."
                hints.append(f"  [{e.get('type', '?')}] {content}")
            if len(mem_entries) > 3:
                hints.append(f"  ... and {len(mem_entries) - 3} more")
        active_plan = self.task_plan.get_active()
        if active_plan:
            steps = active_plan.get("steps", [])
            done = sum(1 for s in steps if s.get("status") in ("done", "skipped"))
            hints.append(f"Plan: {active_plan['title']} ({done}/{len(steps)} done) (/plan)")
        return hints or None

    # ── Main entry ───────────────────────────────────────────────────────

    def run(self, single_shot_query=None):
        if single_shot_query:
            self._run_single_shot(single_shot_query)
            return

        extra = self._startup_hints()
        display.banner(self.config.provider, self.config.model, mode=self.config.mode, extra_lines=extra)
        self._check_proxy()
        self._check_autosave()

        history_file = os.path.join(os.path.expanduser("~"), ".flagscale", "input_history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        completer = WordCompleter(
            ["/quit", "/reload", "/skill", "/file", "/save", "/load", "/export", "/memory", "/mode", "/plan"],
            sentence=True,
        )
        session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
            style=PromptStyle.from_dict({
                "prompt": "#87d787 bold",
                "": "#e4e4e4",
            }),
        )

        while True:
            try:
                user_input = session.prompt([("class:prompt", "> ")]).strip()
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

            if not user_input:
                continue

            cmd = user_input.split()[0] if user_input.startswith("/") else None
            if cmd == "/quit":
                self._exit()
                break
            elif cmd == "/reload":
                self._reload_config()
                self._refresh_system_prompt()
                print("Config and skills reloaded.")
                continue
            elif cmd == "/skill":
                self._handle_skill_command(user_input)
                continue
            elif cmd == "/file":
                self._handle_file_command(user_input)
                continue
            elif cmd == "/save":
                self._handle_save_command(user_input)
                continue
            elif cmd == "/load":
                self._handle_load_command(user_input)
                continue
            elif cmd == "/export":
                self._handle_export_command(user_input)
                continue
            elif cmd == "/memory":
                self._handle_memory_command(user_input)
                continue
            elif cmd == "/mode":
                self._handle_mode_command(user_input)
                continue
            elif cmd == "/plan":
                self._handle_plan_command(user_input)
                continue

            if self.config.auto_skill:
                self._auto_load_skills(user_input)

            self._auto_turn_count = 0
            self._inject_context(user_input)
            self.history.append({"role": "user", "content": user_input})
            self._react_loop()

            # Auto-continue loop: keep generating turns until task is done
            while self.config.mode == "auto" and self._should_auto_continue():
                self._auto_turn_count += 1
                continuation = self._generate_continuation_prompt()
                print(display.yellow(
                    f"\n[Auto turn {self._auto_turn_count}/{self.config.max_auto_turns}] Continuing...\n"
                ))
                self.history.append({"role": "user", "content": continuation})
                try:
                    self._react_loop()
                except KeyboardInterrupt:
                    display.interrupted()
                    print(display.yellow("\n[Auto mode] Interrupted by user.\n"))
                    break

            if self._auto_turn_count > 0:
                print(display.yellow(
                    f"\n[Auto mode] Stopped after {self._auto_turn_count} auto turns.\n"
                ))
                self._auto_turn_count = 0

    def _run_single_shot(self, query):
        if self.config.auto_skill:
            self._auto_load_skills(query)
        self._inject_context(query)
        self.history.append({"role": "user", "content": query})
        self._react_loop()

    # ── Auto-continue logic ────────────────────────────────────────────

    _AUTO_CONTINUE_JUDGE_PROMPT = (
        "You are judging whether an AI agent should continue working or stop.\n\n"
        "The agent's last response:\n---\n{last_response}\n---\n\n"
        "Active plan status: {plan_status}\n"
        "Auto turns so far: {auto_turns}/{max_turns}\n"
        "Turns since last file write: {turns_since_write}\n\n"
        "Should the agent continue to the next turn?\n"
        "Answer CONTINUE if:\n"
        "- There is clearly more work to do (plan steps remaining, ongoing implementation)\n"
        "- The agent is making progress (reading files, writing code, running commands)\n\n"
        "Answer STOP if:\n"
        "- The task appears complete (all plan steps done, final verification passed)\n"
        "- The agent is stuck or looping without progress\n"
        "- The agent needs user input or a decision\n"
        "- The agent explicitly said it's done or waiting\n\n"
        "Reply with ONLY one word: CONTINUE or STOP"
    )

    def _should_auto_continue(self) -> bool:
        if self._auto_turn_count >= self.config.max_auto_turns:
            logger.info("Auto-continue: max turns reached (%d)", self.config.max_auto_turns)
            return False

        last_text = self._get_last_assistant_text()
        if not last_text:
            return False

        # Fast path: explicit tags from agent
        if "[TASK_COMPLETE]" in last_text:
            logger.info("Auto-continue: TASK_COMPLETE tag")
            return False
        if "[NEED_USER_INPUT]" in last_text:
            logger.info("Auto-continue: NEED_USER_INPUT tag")
            return False

        # Hard stagnation limit — more lenient during porting analysis phase
        stagnation_turn_limit = 5
        stagnation_write_gap = 5
        if self._porting_mode and self._current_phase in ("idle", "analysis"):
            stagnation_turn_limit = 12
            stagnation_write_gap = 12
        if (self._auto_turn_count >= stagnation_turn_limit
                and self._turn_count - self._last_write_turn >= stagnation_write_gap):
            logger.info("Auto-continue: hard stagnation (no writes in %d turns)", stagnation_write_gap)
            return False

        # Plan completion check (cheap, no LLM call needed)
        active_plan = self.task_plan.get_active()
        if active_plan:
            steps = active_plan.get("steps", [])
            if steps and all(
                s.get("status") in ("done", "skipped") for s in steps
            ):
                logger.info("Auto-continue: plan fully completed")
                return False

        # LLM judge fallback
        return self._auto_continue_judge(last_text, active_plan)

    def _auto_continue_judge(self, last_text: str, active_plan) -> bool:
        plan_status = "No active plan"
        if active_plan:
            steps = active_plan.get("steps", [])
            done = sum(1 for s in steps if s.get("status") in ("done", "skipped"))
            plan_status = f"{done}/{len(steps)} steps done — {active_plan.get('title', '')}"

        truncated = last_text[:2000] if len(last_text) > 2000 else last_text
        prompt = self._AUTO_CONTINUE_JUDGE_PROMPT.format(
            last_response=truncated,
            plan_status=plan_status,
            auto_turns=self._auto_turn_count,
            max_turns=self.config.max_auto_turns,
            turns_since_write=self._turn_count - self._last_write_turn,
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.provider.chat(messages, tools=[])
            text = result.get("content") or ""
            logger.info("Auto-continue judge: %s", text.strip()[:40])
            return "CONTINUE" in text.upper()
        except Exception as e:
            logger.warning("Auto-continue judge failed: %s — defaulting to continue", e)
            return True

    def _get_last_assistant_text(self) -> str:
        for m in reversed(self.history.messages):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        str(c.get("text", "")) for c in content if isinstance(c, dict)
                    )
                return str(content)
        return ""

    def _generate_continuation_prompt(self) -> str:
        """Generate continuation prompt with plan context for better direction."""
        parts = []

        # Include current plan step if available
        active_plan = self.task_plan.get_active()
        if active_plan:
            steps = active_plan.get("steps", [])
            doing = [s for s in steps if s.get("status") == "doing"]
            pending = [s for s in steps if s.get("status") == "pending"]
            if doing:
                parts.append(f"Current step: {doing[0].get('text', '')[:80]}")
            elif pending:
                parts.append(f"Next step: {pending[0].get('text', '')[:80]}")

        # Include current experiment status if relevant
        exp_name = self._workspace_manager.get_current_experiment()
        if exp_name:
            exp = self._workspace_manager.read_experiment(exp_name)
            if exp and exp.get("status") == "running":
                attempts = exp.get("attempts", [])
                if attempts:
                    parts.append(f"Last attempt: {attempts[-1].get('result', '')[:80]}")

        # Detect language
        last_user = ""
        for m in reversed(self.history.messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                last_user = c if isinstance(c, str) else str(c)
                break
        has_cjk = any('一' <= ch <= '鿿' for ch in last_user)

        if parts:
            context = " | ".join(parts)
            if has_cjk:
                return f"继续。{context}"
            return f"Continue. {context}"

        if has_cjk:
            return "继续。按照你的计划推进任务。"
        return "Continue. Proceed with your plan."

    def _exit(self):
        atexit.unregister(self._atexit_hook)
        self._ensure_memory_written()
        self._auto_update_workspace_state()
        self._archive_session()
        self._clear_autosave()
        session_elapsed = time.time() - self._session_start
        display.session_summary(
            self._turn_count, session_elapsed,
            self._session_input_tokens, self._session_output_tokens,
        )
        print("Bye!")

    _MEMORY_JUDGE_PROMPT = (
        "You are reviewing a conversation to decide if anything is worth remembering "
        "for future sessions.\n\n"
        "Conversation summary:\n{summary}\n\n"
        "Decide: is there any finding, decision, context, or todo worth persisting?\n"
        "Consider:\n"
        "- User explicitly said not to remember → return empty\n"
        "- Casual chat, simple Q&A, no lasting value → return empty\n"
        "- Important discovery (e.g. GPU topology, driver version) → type: finding\n"
        "- Configuration decision (e.g. chose TP=4 PP=2) → type: decision\n"
        "- Unfinished work or next steps → type: todo\n"
        "- General context worth keeping (e.g. cluster setup, user workflow) → type: context\n"
        "- Only save VERIFIED facts, not unconfirmed conclusions\n\n"
        "Reply with ONLY a JSON object (no markdown, no explanation):\n"
        '  {{"save": true, "key": "short_key", "type": "finding|decision|todo|context", '
        '"content": "concise fact under 200 chars"}}\n'
        "or:\n"
        '  {{"save": false}}'
    )

    def _ensure_memory_written(self):
        """Use LLM to decide whether this session is worth remembering."""
        if self._turn_count < 2:
            return
        entries = self.session_memory.list_entries()
        session_entries = [e for e in entries if e.get("session_id") == self._session_id]
        if session_entries:
            return
        # Build conversation summary
        user_msgs = [
            m["content"] for m in self.history.messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if not user_msgs:
            return
        total_len = sum(len(m.strip()) for m in user_msgs)
        if total_len < 20:
            return
        recap_parts = []
        for msg in user_msgs[-5:]:
            line = msg.strip().replace("\n", " ")
            if len(line) > 120:
                line = line[:117] + "..."
            recap_parts.append(line)
        summary = "User: " + " | ".join(recap_parts)
        assistant_msgs = []
        for m in self.history.messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    assistant_msgs.append(content.strip())
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                            assistant_msgs.append(block["text"].strip())
        if assistant_msgs:
            last_replies = assistant_msgs[-2:]
            reply_parts = []
            for r in last_replies:
                line = r.replace("\n", " ")
                if len(line) > 120:
                    line = line[:117] + "..."
                reply_parts.append(line)
            summary += "\nAgent: " + " | ".join(reply_parts)
        # Ask LLM to judge
        try:
            prompt = self._MEMORY_JUDGE_PROMPT.format(summary=summary)
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = "\n".join(text.split("\n")[:-1])
            text = text.strip()
            result = json.loads(text)
            if not result.get("save"):
                return
            key = str(result.get("key", "session_recap"))
            mem_type = result.get("type", "context")
            if mem_type not in ("finding", "decision", "todo", "context"):
                mem_type = "context"
            content = str(result.get("content", ""))[:200]
            if not content:
                return
            # Clean old recaps
            old_recaps = [e for e in entries if e.get("key", "").startswith("session_recap")]
            for old in old_recaps:
                self.session_memory.delete(old["key"])
            self.session_memory.put(key, mem_type, content, self._session_id)
        except Exception as e:
            logger.debug("Memory judge skipped: %s", e)

    _SESSION_SUMMARY_PROMPT = (
        "Summarize this session in 3-5 bullet points. Focus on:\n"
        "- What was accomplished (concrete results, not process)\n"
        "- What blockers remain\n"
        "- What should happen next\n\n"
        "Current task: {task}\n"
        "User messages: {user_msgs}\n"
        "Agent's last actions: {last_actions}\n\n"
        "Reply with ONLY the bullet points, no preamble."
    )

    def _auto_update_workspace_state(self):
        """Auto-update workspace state at session end.

        Updates current.yaml with LLM-generated session summary.
        """
        if not self._session_output_tokens:
            return

        user_msgs = []
        for m in self.history.messages:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                user_msgs.append(content.strip())
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                        user_msgs.append(block["text"].strip())
                        break
        if not user_msgs:
            return

        # Collect last few assistant text outputs for context
        assistant_texts = []
        for m in self.history.messages:
            if m.get("role") != "assistant":
                continue
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                assistant_texts.append(content.strip())
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                        assistant_texts.append(block["text"].strip())
        last_actions = " | ".join(t[:120] for t in assistant_texts[-3:])

        # Generate LLM summary
        task = self._workspace_manager.get_current_task() or user_msgs[0][:100]
        user_msgs_summary = " | ".join(m[:80] for m in user_msgs[-5:])
        summary = ""
        try:
            prompt = self._SESSION_SUMMARY_PROMPT.format(
                task=task,
                user_msgs=user_msgs_summary,
                last_actions=last_actions[:500],
            )
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            summary = (response.get("content") or "").strip()[:500]
        except Exception as e:
            logger.debug("LLM session summary failed: %s", e)

        # Add metadata
        elapsed = time.time() - self._session_start
        elapsed_str = f"{int(elapsed // 60)}m" if elapsed > 60 else f"{int(elapsed)}s"
        metadata = f"Turns: {self._turn_count}, Duration: {elapsed_str}"
        if self._session_input_tokens or self._session_output_tokens:
            metadata += f", Tokens: {self._session_input_tokens}in/{self._session_output_tokens}out"

        full_summary = f"{metadata}\n{summary}" if summary else metadata

        # Update current.yaml
        try:
            self._workspace_manager.update_current(session_summary=full_summary)
        except Exception as e:
            logger.debug("Workspace current.yaml update skipped: %s", e)

        # Append to session history (rolling, never overwritten)
        try:
            self._workspace_manager.append_session_summary(
                session_id=self._session_id,
                task=task,
                summary=summary or "(no summary generated)",
                metadata=metadata,
            )
        except Exception as e:
            logger.debug("Session history append skipped: %s", e)

    def _archive_session(self):
        """Archive the current session to disk (no LLM call)."""
        msgs = [m for m in self.history.full_log if m.get("role") != "system"]
        if not msgs or self._turn_count == 0:            return
        try:
            metadata = {
                "provider": self.config.provider,
                "model": self.config.model,
                "turns": self._turn_count,
                "session_id": self._session_id,
                "input_tokens": self._session_input_tokens,
                "output_tokens": self._session_output_tokens,
            }
            save_session(msgs, self.config.session_dir, f"session_{self._session_id}", metadata)
        except Exception as e:
            logger.warning("Session archive failed: %s", e)

    # ── Autosave / Resume ───────────────────────────────────────────────

    def _autosave_path(self):
        d = self.config.session_dir or os.path.join(Path.home(), ".flagscale", "sessions")
        return os.path.join(d, "autosave.json")

    def _autosave(self):
        try:
            from flagscale.agent.react.history import _truncate_message
            msgs = [m for m in self.history.messages if m.get("role") != "system"]
            if not msgs:
                return
            keep_recent = min(20, len(msgs))
            compact_msgs = []
            for i, m in enumerate(msgs):
                if i < len(msgs) - keep_recent:
                    compact_msgs.append(_truncate_message(m))
                else:
                    compact_msgs.append(m)
            full_log = [m for m in self.history.full_log if m.get("role") != "system"]
            metadata = {
                "provider": self.config.provider,
                "model": self.config.model,
                "turns": self._turn_count,
                "loaded_skills": list(self._loaded_skills),
                "input_tokens": self._session_input_tokens,
                "output_tokens": self._session_output_tokens,
            }
            path = self._autosave_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "id": "autosave",
                "timestamp": time.time(),
                "metadata": metadata,
                "messages": compact_msgs,
                "full_log": full_log,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Autosave failed: %s", e)

    def _clear_autosave(self):
        try:
            path = self._autosave_path()
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

    def _check_autosave(self):
        path = self._autosave_path()
        if not os.path.isfile(path):
            return
        try:
            data = load_session(path)
        except Exception:
            return

        msgs = data.get("messages", [])
        if not msgs:
            return

        meta = data.get("metadata", {})
        turn_count = meta.get("turns", 0)
        user_msgs = [m for m in msgs if m.get("role") == "user"
                     and isinstance(m.get("content"), str)]
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        timestamp = data.get("timestamp", 0)

        display.autosave_found(turn_count, len(user_msgs), last_user, timestamp)

        try:
            answer = input("Resume previous session? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if answer in ("n", "no"):
            self._clear_autosave()
            print("Cleared. Starting new session.\n")
            return

        # Restore state
        from flagscale.agent.react.history import _validate_tool_pairs
        sys_msg = self.history.messages[0] if self.history.messages and self.history.messages[0].get("role") == "system" else None
        self.history._messages = [sys_msg] if sys_msg else []
        self.history._messages.extend(_validate_tool_pairs(msgs))
        full_log = data.get("full_log", [])
        if full_log:
            self.history._full_log = [sys_msg] if sys_msg else []
            self.history._full_log.extend(full_log)
        else:
            self.history._full_log = list(self.history._messages)
        self._turn_count = turn_count
        self._loaded_skills = set(meta.get("loaded_skills", []))
        self._session_input_tokens = meta.get("input_tokens", 0)
        self._session_output_tokens = meta.get("output_tokens", 0)
        display.autosave_resumed(turn_count)

    # ── Poll mode (token-saving optimization) ────────────────────────────

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

        # Track shell errors for progress gate
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

    _MONITOR_CMD_RE = re.compile(
        r'^(tail|head|cat|wc|grep|ls)\b.*'
        r'(log|output|stdout|stderr|nohup\.out|train.*\.log)',
        re.IGNORECASE,
    )

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
        """Check if recent iterations form a polling pattern.

        Detects both exact-command repetition and fuzzy monitoring patterns
        (e.g., tail -20 then tail -30 on the same log file).
        """
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
        # Exact match
        if len(set(commands)) == 1 and commands[0]:
            return True
        # Fuzzy match: all commands target the same log file
        if all(self._MONITOR_CMD_RE.match(c) for c in commands):
            targets = [self._normalize_monitor_cmd(c) for c in commands]
            if len(set(targets)) == 1:
                return True
        return False

    _INTERESTING_CHANGE_RE = re.compile(
        r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
        r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
        r'torch\.cuda\.OutOfMemoryError|CUDA error|'
        r'loss[=:\s]|grad.norm|throughput|step\s+\d|iteration\s+\d|'
        r'training\s+complete|finished|saved\s+checkpoint',
        re.IGNORECASE,
    )

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
        """Check if the change is interesting enough to return to LLM.

        Routine changes (e.g., wc -l going from 24 to 25) should keep
        polling.  Interesting changes (errors, training metrics, large
        jumps) should break out so the LLM can react.
        """
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

    _TRAIN_CMD_RE = re.compile(r'flagscale\s+train|torchrun|python.*(?:train|verify|dryrun|test_model)')
    _TRAIN_LAUNCH_RE = re.compile(
        r'flagscale\s+train|torchrun\s|deepspeed\s|'
        r'python\s+.*(?:pretrain|finetune|train).*\.py',
    )
    _TRAIN_FAIL_RE = re.compile(
        r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
        r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
        r'torch\.cuda\.OutOfMemoryError|CUDA error',
        re.IGNORECASE,
    )

    # Error pattern classification for smart escalation
    _ERROR_PATTERNS = {
        "import_error": re.compile(r'ModuleNotFoundError|ImportError|No module named', re.I),
        "oom": re.compile(r'OutOfMemoryError|CUDA out of memory|OOM', re.I),
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

    # Map error patterns to skills that can help
    _ERROR_SKILL_MAP = {
        "nccl_timeout": "parallel-strategy",
        "oom": "parallel-strategy",
        "shape_mismatch": "model-porter",
        "checkpoint_load": "model-porter",
        "config_error": "train-run",
        "data_pipeline": "data-prep",
        "import_error": "env-setup",
    }

    def _classify_error_pattern(self, error_text):
        """Classify error into pattern categories."""
        for pattern_name, pattern_re in self._ERROR_PATTERNS.items():
            if pattern_re.search(error_text):
                return pattern_name
        return "unknown"

    _CHECKPOINT_LOAD_RE = re.compile(
        r'--resume[_-]from|--finetune[_-]from|--load\s|--pretrained[_-]model|'
        r'--init[_-]checkpoint|--restore[_-]file',
    )

    _DRY_RUN_WARNING = (
        "\n⚠️ PRE-LAUNCH CHECK: This command loads a checkpoint but no dry-run was done first.\n"
        "Principle: validate cheap things before expensive things. Checkpoint loading is slow — "
        "verify the pipeline works with random init first, then add checkpoint loading.\n"
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
        "3. Update workspace_current with new blockers/next_steps if needed.\n"
    )

    _WORKAROUND_MEMORY_HINT = (
        "\n💡 MEMORY HINT: You just succeeded after a prior failure on a similar operation. "
        "If a future session could hit the same issue, memorize the fix now with memory_write. "
        "Include: what failed, why, and the exact workaround (flag, path, version, etc.).\n"
    )

    _SESSION_MEMORY_REVIEW = (
        "\n📝 SESSION REVIEW: Before this session ends, consider: did you discover any "
        "env quirks, version constraints, tool incompatibilities, or workarounds during this session? "
        "If so, save them with memory_write so future sessions don't repeat the same trial-and-error.\n"
    )

    _TRAINING_MEMORY_HINT = (
        "\n💡 MEMORY HINT: Training is running. Review what you learned getting here — "
        "if you figured something out through trial-and-error, memorize it so the next session doesn't repeat the work.\n"
    )

    _STALE_MEMORY_WARNING_TEMPLATE = (
        "\n⚠️ STALE MEMORIES: {count} memory entries are older than {days} days: {keys}. "
        "When you encounter these during work, verify they still hold. "
        "If outdated, update or delete them with memory_write / memory_read.\n"
    )

    # ── Checkpoint Capture ──────────────────────────────────────────────

    def _checkpoint_training_launch(self, cmd: str, result: str):
        """Checkpoint: training launched successfully. Auto-record to experiment only.

        Training launches are recorded in the experiment's attempt list, not in memory.
        Memory is reserved for durable cross-session knowledge (workarounds, env quirks).
        """
        current_exp = self._workspace_manager.get_current_experiment()
        if not current_exp:
            return ""

        cmd_summary = cmd[:200] if len(cmd) <= 200 else cmd[:197] + "..."
        warning = ""
        try:
            self._workspace_manager.add_attempt(current_exp, f"Training launched: {cmd_summary}", "Running...")
        except Exception as e:
            logger.warning("Failed to add experiment attempt: %s", e)
            warning = f"\n⚠️ Experiment update failed: {e}. Attempt not recorded.\n"

        return warning

    def _checkpoint_training_failure(self, cmd: str, result: str):
        """Checkpoint: training failed. Record to experiment only.

        Failures are tracked in the experiment's attempt list.
        Only persistent/reusable knowledge (workarounds) goes to memory.
        """
        current_exp = self._workspace_manager.get_current_experiment()
        if not current_exp:
            return ""

        error_summary = self._extract_error_summary(result)
        warning = ""
        try:
            self._workspace_manager.update_last_attempt(current_exp, f"FAILED: {error_summary}")
        except Exception as e:
            logger.warning("Failed to update experiment attempt: %s", e)
            warning = f"\n⚠️ Experiment update failed: {e}. Failure not recorded.\n"

        return warning

    def _checkpoint_workaround(self, tool_name: str, prev_error: str, curr_cmd: str):
        """Checkpoint: workaround found. Save to memory with semantic key.

        Workarounds are genuinely reusable cross-session knowledge.
        Use a semantic key derived from the error so duplicates auto-merge.
        """
        prev_summary = prev_error[:100] if len(prev_error) <= 100 else prev_error[:97] + "..."
        cmd_summary = curr_cmd[:100] if len(curr_cmd) <= 100 else curr_cmd[:97] + "..."
        content = f"Workaround: {tool_name} failed with [{prev_summary}], fixed by [{cmd_summary}]"

        # Derive semantic key from error signature
        error_words = re.sub(r'[^a-z0-9\s]', '', prev_error[:60].lower()).split()[:4]
        semantic_key = "workaround_" + "_".join(error_words) if error_words else "workaround_unknown"
        semantic_key = SessionMemory.sanitize_key(semantic_key)

        task = self._workspace_manager.get_current_task()
        warning = ""
        try:
            self.session_memory.put(
                key=semantic_key,
                mem_type="finding",
                content=content,
                session_id=self._session_id,
                task=task,
            )
        except Exception as e:
            logger.warning("Failed to save workaround to memory: %s", e)
            warning = f"\n⚠️ Memory write failed: {e}. Workaround not recorded.\n"

        return warning

    def _checkpoint_new_error(self, error_signature: str, full_error: str):
        """Checkpoint: new unique error encountered.

        Errors are tracked in-session only (via _seen_errors set) for dedup.
        They are NOT auto-saved to persistent memory — the experiment's attempt
        list captures them. Only workarounds (solutions) deserve memory slots.
        """
        if not hasattr(self, "_seen_errors"):
            self._seen_errors = set()

        if error_signature in self._seen_errors:
            return ""
        self._seen_errors.add(error_signature)
        return ""

    def _extract_error_summary(self, result: str) -> str:
        """Extract first meaningful error line from tool result."""
        lines = result.split("\n")
        for line in lines:
            line = line.strip()
            if any(kw in line.lower() for kw in ("error", "exception", "failed", "traceback", "oom", "cuda")):
                return line[:200] if len(line) <= 200 else line[:197] + "..."
        # No error keyword found, return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()[:200]
        return "Unknown error"

    def _mid_turn_autosave(self):
        """Save critical state mid-turn to survive crashes (OOM, SIGKILL)."""
        try:
            state = {
                "files_read": list(self._files_read_this_session),
                "reading_categories": list(self._reading_categories),
                "verification_stage": self._verification_stage,
                "current_phase": self._current_phase,
                "error_pattern_history": list(self._error_pattern_history),
                "output_tokens": self._session_output_tokens,
                "turn_count": self._turn_count,
                "consecutive_train_failures": self._consecutive_train_failures,
                "gates_unlocked": {
                    "understanding": getattr(self, '_understanding_verified', False),
                    "component_plan": getattr(self, '_component_plan_created', False),
                    "imports": getattr(self, '_imports_verified', False),
                    "reference_comparison": getattr(self, '_reference_comparison_planned', False),
                    "failure_mode": getattr(self, '_failure_mode_analyzed', False),
                    "sanity_checks": getattr(self, '_sanity_checks_passed', False),
                    "config_model": getattr(self, '_config_model_verified', False),
                    "env": getattr(self, '_env_verified', False),
                    "component_integration": getattr(self, '_component_integration_verified', False),
                },
                "session_id": self._session_id,
                "timestamp": time.time(),
            }
            state_path = os.path.join(self._workspace_manager._dir, ".agent_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            import json
            with open(state_path, "w") as f:
                json.dump(state, f)
            logger.info("Mid-turn autosave: %d files, phase=%s, verification=%s, turn=%d",
                        len(self._files_read_this_session), self._current_phase,
                        self._verification_stage, self._turn_count)
        except Exception as e:
            logger.warning("Mid-turn autosave failed: %s", e)

    def _record_verification_advance(self, new_stage, command_snippet):
        """Auto-record verification stage advancement to workspace experiment."""
        try:
            exp_name = self._workspace_manager.get_current_experiment()
            if exp_name:
                change = f"Verification advanced to {new_stage}"
                result = f"PASS — via: {command_snippet}"
                self._workspace_manager.add_attempt(exp_name, change, result)
                logger.info("Auto-recorded verification advance: %s", new_stage)
        except Exception as e:
            logger.warning("Failed to auto-record verification advance: %s", e)

    # ── Enforcement Mechanisms ─────────────────────────────────────────

    # Progress gate is now staleness-based (see _check_progress_gate)

    _PRODUCTIVE_TOOLS = frozenset({
        "memory_write", "write_file", "edit_file",
        "workspace_experiment", "workspace_current",
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
        else:
            self._reads_since_last_new_file += 1

        # Detect repeated shell errors (same error appearing multiple times)
        repeated_errors = self._count_repeated_recent_errors()

        # === Intervention logic ===

        # Pattern 1: Re-reading without discovering anything new for a long time
        stale_threshold = 12
        if self._porting_mode:
            stale_threshold = 30  # Porting requires reading many source files
        elif self._consecutive_train_failures >= 2:
            stale_threshold = 18  # More lenient during debugging

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
        reads_hard_cap = 60 if self._porting_mode else 40
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

    _PLAN_GATE_MAX_EXPLORATORY = 20
    _PLAN_GATE_INDEPENDENT_WARN = 25
    _PLAN_GATE_INDEPENDENT_BLOCK = 35

    def _check_plan_creation_gate(self, tool_name):
        """Gate: encourage plan creation for sustained exploration.

        Two activation modes:
        1. Complexity judge fired → _complex_task_no_plan = True, hard block at 20
        2. Independent: warn at 25 consecutive reads, hard block at 35

        Returns block/warning message or empty string.
        Hard block (non-empty + "TOOL NOT EXECUTED") means tool must NOT execute.
        """
        # Plan already exists — no gate needed
        if self.task_plan.get_active() is not None:
            self._complex_task_no_plan = False
            return ""

        # Productive tools are always allowed
        if tool_name in ("plan_create", "memory_write", "workspace_experiment", "workspace_current"):
            return ""

        # Mode 1: complexity judge fired — hard block at 6
        if self._complex_task_no_plan:
            self._pre_plan_tool_calls += 1
            if self._pre_plan_tool_calls > self._PLAN_GATE_MAX_EXPLORATORY:
                return (
                    f"⛔ [PLAN GATE — TOOL NOT EXECUTED] This task was flagged as complex. "
                    f"You've used {self._pre_plan_tool_calls} exploratory calls "
                    f"(limit: {self._PLAN_GATE_MAX_EXPLORATORY}) without creating a plan.\n"
                    f"This tool call was BLOCKED. You MUST call plan_create NOW.\n"
                    f"Use what you've gathered so far to create a concrete step-by-step plan."
                )

        # Mode 2: independent — soft warn at 8, hard block at 12
        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_BLOCK:
            return (
                f"⛔ [PLAN GATE — TOOL NOT EXECUTED] You've made {self._consecutive_reads} "
                f"consecutive exploratory calls without creating a plan.\n"
                f"This tool call was BLOCKED. You MUST call plan_create NOW to organize your approach."
            )
        if self._consecutive_reads >= self._PLAN_GATE_INDEPENDENT_WARN:
            return (
                f"\n\n[PLAN REMINDER] You've made {self._consecutive_reads} exploratory calls "
                f"without a plan. Consider calling plan_create soon to organize your findings. "
                f"You will be BLOCKED at {self._PLAN_GATE_INDEPENDENT_BLOCK} calls."
            )

        return ""

    def _check_dry_run_gate(self, cmd, result):
        """Enforce dry-run before full training."""
        if not self._TRAIN_LAUNCH_RE.search(cmd):
            return result
        if self._is_quick_test_command(cmd):
            self._dry_run_passed = True
            # Check if using synthetic data — remind about real data verification
            uses_synthetic = bool(re.search(r'synthetic|/dev/null|mock.data|fake', cmd, re.I))
            synthetic_note = ""
            if uses_synthetic:
                synthetic_note = (
                    "\n\n[SYNTHETIC DATA NOTE] This dry-run used synthetic/mock data. "
                    "Before full training with real data, you MUST verify:\n"
                    "1. Real data loads without error (1 batch)\n"
                    "2. Batch shapes/dtypes match what model expects\n"
                    "3. Tokenization/preprocessing produces expected output\n"
                    "Do NOT skip this — synthetic dry-run passing does NOT guarantee real data works."
                )
            return result + (
                "\n\n[DRY RUN COMPLETE] Verify: model loaded? data flowing? "
                "no crashes? If OK, proceed to full run."
            ) + synthetic_note
        if not self._dry_run_passed:
            return result + (
                "\n\n[WARNING: NO DRY RUN] This is a full training run without "
                "prior dry-run verification. Issues like unloaded checkpoints, "
                "broken data pipelines, or config errors will waste GPU hours. "
                "Consider stopping and running with --max-steps=2 first."
            )
        return result

    def _check_kill_retry_loop(self, cmd):
        """Detect kill+relaunch cycles. 3 kills in 20 minutes = forced audit."""
        now = time.time()
        is_kill = bool(re.search(r'pkill|kill\s+-9|killall', cmd))
        is_launch = bool(self._TRAIN_LAUNCH_RE.search(cmd))

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
        if elapsed < 120:
            return ""
        # Check if result shows hang indicators
        hang_indicators = [
            "before the start of training step" in result and "iteration" not in result.split("before the start")[-1],
            "0%" in result and "utilization" in result.lower(),
            elapsed > 180 and "iteration" not in result[-500:],
        ]
        if any(hang_indicators):
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
        """After error, require root cause before big changes."""
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
            self._update_snapshot()
            # Set anchors so the summary preserves critical info
            anchors = self._get_compaction_anchors()
            if anchors:
                self.history.set_compaction_anchors(anchors)
            self.history.force_compact(target_ratio=0.60)
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
                "4. workspace_current: update next_steps\n"
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

    def _pre_compaction_memory_dump(self):
        """Auto-save critical context before compaction destroys it."""
        try:
            recent = self.history.messages[-20:]
            errors_found = []
            solutions_found = []
            files_modified = []
            current_approach = ""

            for msg in recent:
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
                if "write_file" in text or "edit_file" in text:
                    import re as _re
                    paths = _re.findall(r'["\']([/\w._-]+\.(py|yaml|sh))["\']', text)
                    files_modified.extend(p[0] for p in paths[:5])
                if "approach" in text.lower() or "strategy" in text.lower() or "plan" in text.lower():
                    if len(text) < 500:
                        current_approach = text[:300]

            # Build checkpoint content
            parts = []
            if errors_found:
                parts.append(f"Errors: {'; '.join(errors_found[:5])}")
            if solutions_found:
                parts.append(f"Solutions: {'; '.join(solutions_found[:5])}")
            if files_modified:
                parts.append(f"Files modified: {', '.join(set(files_modified)[:10])}")
            if current_approach:
                parts.append(f"Approach: {current_approach}")

            parts.append(f"Files read this session: {len(self._files_read_this_session)}")
            parts.append(f"Phase: {self._current_phase}, Verification: {self._verification_stage}")
            parts.append(f"Turn: {self._turn_count}")

            checkpoint_content = "\n".join(parts)

            # Save to memory with high priority (survives TTL)
            key = f"compaction_checkpoint_{int(time.time())}"
            self.session_memory.put(
                key, "context", checkpoint_content,
                self._session_id,
                task=self._workspace_manager.get_current_task(),
                priority="high",
            )

            # Update workspace current
            self._update_workspace_current_context(
                context_update=f"Pre-compaction: {len(self._files_read_this_session)} files read, "
                               f"phase={self._current_phase}, verification={self._verification_stage}",
            )

            # Add to experiment if active
            exp = self._workspace_manager.get_current_experiment()
            if exp:
                self._workspace_manager.add_attempt(
                    exp, "pre-compaction checkpoint",
                    f"Turn {self._turn_count}: {checkpoint_content[:500]}"
                )

            logger.info("Pre-compaction memory dump: saved checkpoint with %d errors, %d solutions",
                        len(errors_found), len(solutions_found))
        except Exception as e:
            logger.warning("Pre-compaction memory dump failed: %s", e)

    def _update_workspace_current_context(self, context_update: str = "", next_steps: list = None):
        """Incrementally update current.yaml with new context or next_steps."""
        try:
            current = self._workspace_manager.read_current() or {}
            ctx = current.get("context", [])
            if isinstance(ctx, str):
                ctx = [ctx]
            if context_update:
                ctx.append(context_update)
                # Keep last 10 context entries
                ctx = ctx[-10:]
            if next_steps is not None:
                current["next_steps"] = next_steps
            current["context"] = ctx
            current["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._workspace_manager.write_current(current)
        except Exception as e:
            logger.warning("Failed to update workspace current: %s", e)

    def _auto_update_workspace_on_progress(self, tool_name, arguments):
        """Auto-update workspace current after significant progress events."""
        try:
            # After plan creation/update
            if tool_name in ("plan_create", "plan_update"):
                plan = self.task_plan.get_active()
                if plan:
                    pending = [s.get("text", "") for s in plan.get("steps", []) if s.get("status") == "pending"]
                    self._update_workspace_current_context(
                        context_update=f"Plan updated (turn {self._turn_count})",
                        next_steps=pending[:5],
                    )

            # After verification stage advance
            elif tool_name == "workspace_experiment":
                action = arguments.get("action", "")
                if action in ("advance_verification", "add_attempt"):
                    self._update_workspace_current_context(
                        context_update=f"Verification: {self._verification_stage} (turn {self._turn_count})",
                    )

            # After successful training launch
            elif tool_name == "shell":
                cmd = arguments.get("command", "")
                if self._is_training_launch(cmd):
                    self._update_workspace_current_context(
                        context_update=f"Training launched (turn {self._turn_count}): {cmd[:100]}",
                    )
        except Exception as e:
            logger.debug("Auto workspace update failed: %s", e)

    # ── Auto-Persistence Layer ─────────────────────────────────────────

    def _auto_persist_on_event(self, tool_name, arguments, result, error):
        """Automatic persistence after key events — no agent decision needed."""
        try:
            cmd = arguments.get("command", "") if tool_name == "shell" else ""

            # Training failure → auto-record
            if tool_name == "shell" and self._is_training_launch(cmd):
                if error or (result and ("Error" in result or "Traceback" in result)):
                    self._auto_record_training_attempt(cmd, result, success=False)
                elif result and ("iteration" in result.lower() or "loss" in result.lower()):
                    self._auto_record_training_attempt(cmd, result, success=True)

            # Kill command → auto-record
            if tool_name == "shell" and re.search(r'pkill|kill\s+-?\d|killall', cmd):
                self._auto_record_kill_event(cmd)

            # File write to model code → auto-record
            if tool_name in ("write_file", "edit_file") and not error:
                path = arguments.get("path", "") or arguments.get("file_path", "")
                if path and self._porting_mode and re.search(r'model|train|layer|attention|mlp|embed', path, re.I):
                    self._auto_record_code_change(path)

            # Plan step done → auto-add experiment attempt
            if tool_name == "plan_update" and arguments.get("status") == "done":
                self._auto_bind_plan_to_experiment(arguments)

        except Exception as e:
            logger.debug("Auto-persist failed: %s", e)

    def _auto_record_training_attempt(self, cmd, result, success):
        """Record training attempt to experiment and update snapshot."""
        exp_name = self._workspace_manager.get_current_experiment()
        if not exp_name:
            return
        if success:
            # Extract first iteration info
            lines = (result or "").split("\n")
            info_lines = [l for l in lines if "iteration" in l.lower() or "loss" in l.lower()][:3]
            summary = "\n".join(info_lines)[:200] if info_lines else "Training started successfully"
            self._workspace_manager.add_attempt(exp_name, f"LAUNCH: {cmd[:80]}", f"SUCCESS: {summary}")
        else:
            # Extract error
            err_lines = []
            for line in (result or "").split("\n"):
                if "error" in line.lower() or "traceback" in line.lower() or "assert" in line.lower():
                    err_lines.append(line.strip())
            error_summary = "\n".join(err_lines[-3:])[:200] if err_lines else "Unknown error"
            self._workspace_manager.add_attempt(exp_name, f"LAUNCH: {cmd[:80]}", f"FAIL: {error_summary}")
        self._update_snapshot()

    def _auto_record_kill_event(self, cmd):
        """Record kill event with inferred reason from recent context."""
        reason = "unknown"
        for msg in reversed(self.history.messages[-5:]):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            text = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            for line in text.split("\n"):
                if any(kw in line.lower() for kw in ("hang", "stuck", "oom", "error", "wrong", "fix", "retry")):
                    reason = line.strip()[:150]
                    break
            if reason != "unknown":
                break
        exp_name = self._workspace_manager.get_current_experiment()
        if exp_name:
            self._workspace_manager.add_attempt(exp_name, f"KILL: {cmd[:80]}", f"Reason: {reason}")
        self._update_snapshot()

    def _auto_record_code_change(self, path):
        """Record model code change to experiment."""
        exp_name = self._workspace_manager.get_current_experiment()
        if exp_name:
            self._workspace_manager.add_attempt(
                exp_name, f"CODE: modified {os.path.basename(path)}", f"File: {path}"
            )

    def _auto_bind_plan_to_experiment(self, arguments):
        """When plan step is done, auto-add experiment attempt."""
        step_id = arguments.get("step_id", "")
        notes = arguments.get("notes", "done")
        exp_name = self._workspace_manager.get_current_experiment()
        if exp_name:
            self._workspace_manager.add_attempt(
                exp_name, f"Plan step {step_id} completed", notes[:200]
            )
        self._update_snapshot()

    # ── Unified Snapshot ───────────────────────────────────────────────

    def _update_snapshot(self):
        """Generate and write unified snapshot from current state."""
        try:
            plan = self.task_plan.get_active() if hasattr(self, 'task_plan') else None
            plan_progress = ""
            current_step = ""
            if plan:
                steps = plan.get("steps", [])
                done = sum(1 for s in steps if s.get("status") == "done")
                total = len(steps)
                plan_progress = f"{done}/{total} steps done"
                doing = [s for s in steps if s.get("status") == "doing"]
                if doing:
                    current_step = doing[0].get("text", doing[0].get("title", ""))[:100]

            # Get last 3 attempts from experiment
            last_attempts = []
            exp_name = self._workspace_manager.get_current_experiment()
            if exp_name:
                exp = self._workspace_manager.read_experiment(exp_name)
                if exp:
                    for a in (exp.get("attempts", []) or [])[-3:]:
                        last_attempts.append({
                            "change": a.get("change", "")[:100],
                            "result": a.get("result", "")[:100],
                        })

            # Get key decisions from memory
            key_decisions = []
            decisions = [e for e in self.session_memory.list_entries()
                         if e.get("type") == "decision" or e.get("priority") == "high"]
            for d in decisions[-5:]:
                key_decisions.append(d.get("content", "")[:120])

            current = self._workspace_manager.read_current()
            snapshot = {
                "task": current.get("task", ""),
                "status": current.get("status", "running"),
                "phase": getattr(self, '_current_phase', 'unknown'),
                "verification_stage": getattr(self, '_verification_stage', 'none'),
                "plan_progress": plan_progress,
                "current_step": current_step,
                "last_3_attempts": last_attempts,
                "key_decisions": key_decisions,
                "current_errors": current.get("blockers", []),
                "files_modified_recently": list(getattr(self, '_files_written_this_session', set()))[:10],
                "next_action": current.get("next_steps", [""])[0] if current.get("next_steps") else "",
                "turn_count": self._turn_count,
            }
            self._workspace_manager.write_snapshot(snapshot)
        except Exception as e:
            logger.debug("Snapshot update failed: %s", e)

    def _get_compaction_anchors(self) -> list:
        """Extract mandatory anchors for summary preservation."""
        anchors = []
        try:
            plan = self.task_plan.get_active() if hasattr(self, 'task_plan') else None
            if plan:
                doing = [s for s in plan.get("steps", []) if s.get("status") == "doing"]
                if doing:
                    anchors.append(f"Current plan step: {doing[0].get('text', '')[:80]}")

            entries = self.session_memory.list_entries()
            high_pri = [e for e in entries if e.get("priority") == "high"]
            for e in high_pri[-3:]:
                anchors.append(f"Key [{e['key']}]: {e['content'][:80]}")

            current = self._workspace_manager.read_current()
            blockers = current.get("blockers", [])
            if blockers:
                anchors.append(f"Blocker: {blockers[-1][:80]}")

            if hasattr(self, '_files_read_this_session') and self._files_read_this_session:
                files_list = list(self._files_read_this_session)[:8]
                anchors.append(f"Files already read: {', '.join(os.path.basename(f) for f in files_list)}")

            # Porting-specific anchors: model mapping and verification stage
            if self._porting_mode:
                if hasattr(self, '_verification_stage') and self._verification_stage != "none":
                    anchors.append(f"Verification stage: {self._verification_stage}")
                # Include porting-related memory entries (component mapping, architecture decisions)
                porting_entries = [
                    e for e in entries
                    if any(kw in (e.get("content") or "").lower()
                           for kw in ("mapping", "component", "architecture", "porting", "model structure"))
                ]
                for e in porting_entries[-2:]:
                    anchors.append(f"Porting [{e['key']}]: {e['content'][:100]}")
                # Current experiment status
                exp_name = self._workspace_manager.get_current_experiment()
                if exp_name:
                    exp = self._workspace_manager.read_experiment(exp_name)
                    if exp:
                        attempts = exp.get("attempts", [])
                        if attempts:
                            last = attempts[-1]
                            anchors.append(f"Last experiment attempt: {last.get('change', '')[:60]} → {last.get('result', '')[:60]}")
        except Exception:
            pass
        return anchors

    def _format_snapshot_as_resume(self, snapshot: dict = None) -> str:
        """Format snapshot as a resume hint for injection after compaction or session start."""
        if snapshot is None:
            snapshot = self._workspace_manager.read_snapshot()
        if not snapshot:
            # Fallback to current.yaml
            try:
                current = self._workspace_manager.read_current()
            except Exception:
                current = None
            if not current:
                return ""
            task = current.get("task", "")
            status = current.get("status", "")
            if isinstance(task, dict):
                status = task.get("status", status)
                task_name = task.get("name", str(task))
            else:
                task_name = str(task) if task else ""
            if not task_name or status in ("completed", "abandoned", ""):
                return ""
            return (
                f"\n<system-hint>[SESSION RESUME] Task: {task_name} (status: {status})\n"
                f"Run workspace_current and memory_read to load full state.</system-hint>\n"
            )

        parts = []
        if snapshot.get("task"):
            parts.append(f"Task: {snapshot['task']}")
        if snapshot.get("status"):
            parts.append(f"Status: {snapshot['status']}")
        if snapshot.get("phase"):
            parts.append(f"Phase: {snapshot['phase']}, Verification: {snapshot.get('verification_stage', 'none')}")
        if snapshot.get("plan_progress"):
            parts.append(f"Plan: {snapshot['plan_progress']}")
        if snapshot.get("current_step"):
            parts.append(f"Current step: {snapshot['current_step']}")
        if snapshot.get("last_3_attempts"):
            parts.append("Recent attempts:")
            for a in snapshot["last_3_attempts"]:
                parts.append(f"  - {a.get('change', '')} → {a.get('result', '')}")
        if snapshot.get("key_decisions"):
            parts.append("Key decisions:")
            for d in snapshot["key_decisions"]:
                parts.append(f"  - {d}")
        if snapshot.get("current_errors"):
            parts.append(f"Blockers: {snapshot['current_errors']}")
        if snapshot.get("next_action"):
            parts.append(f"Next action: {snapshot['next_action']}")
        if snapshot.get("files_modified_recently"):
            parts.append(f"Files modified: {', '.join(snapshot['files_modified_recently'][:5])}")

        if not parts:
            return ""
        body = "\n  ".join(parts)
        return (
            f"\n<system-hint>[SESSION RESUME] Unified snapshot:\n  {body}\n\n"
            f"RECOVERY: Do NOT re-read files already analyzed — check memory first. "
            f"Continue from next_action above.</system-hint>\n"
        )

    def _check_progress_checkpoint(self):
        """Force progress review every 10K output tokens."""
        if self._session_output_tokens - self._last_checkpoint_tokens > 10000:
            self._last_checkpoint_tokens = self._session_output_tokens
            return (
                "\n\n[PROGRESS CHECKPOINT] You've generated 10K+ tokens since last checkpoint.\n"
                "Before continuing, answer briefly:\n"
                "1. What have you accomplished in the last ~10 turns?\n"
                "2. What's the current blocker (if any)?\n"
                "3. What's the next concrete action?\n"
                "Write your answers to memory, then continue."
            )
        return ""

    _PORTING_WRITE_PATHS = re.compile(
        r'(tools/checkpoint/|flagscale/models/|megatron/.*model|model_provider|spec\.py|'
        r'pretrain_|train_.*\.py|data/.*dataset)'
    )
    _MIN_READS_BEFORE_PORTING_WRITE = 8

    def _check_porting_path_gate(self, tool_name, arguments):
        """Block porting code writes until user has confirmed the porting path (Mode B/C)."""
        if not self._porting_mode or self._porting_path_confirmed:
            return ""
        if tool_name not in ("write_file", "edit_file"):
            return ""
        target = arguments.get("path", "") or arguments.get("file_path", "")
        if not self._PORTING_WRITE_PATHS.search(target):
            return ""
        return (
            "\n\n[PORTING PATH GATE] You must confirm the porting path with the user before "
            "writing any porting code.\n\n"
            "FlagScale supports two paths for models with custom components:\n"
            "- Mode B (Megatron Native): Full parallelism (TP/PP/EP/CP), best performance, "
            "higher implementation effort\n"
            "- Mode C (HuggingFace Wrapper): FSDP2 distribution, HF model as-is, "
            "fastest to implement, limited parallelism\n\n"
            "Present the trade-offs to the user and get their explicit choice.\n"
            "Then create a plan (plan_create) that includes the chosen path, or record the "
            "decision in workspace_experiment with a 'porting_path' field."
        )

    def _check_reading_depth_gate(self, tool_name, arguments):
        """Block porting code writes if insufficient reading has been done."""
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
            f"\n\n[READING DEPTH CHECK] You're writing porting code but have only read "
            f"{read_count} files this session (minimum: {self._MIN_READS_BEFORE_PORTING_WRITE}). "
            f"Model porting failures are almost always caused by incomplete understanding. "
            f"Before writing:\n"
            f"1. Have you read the COMPLETE source model code (modeling_*.py, config.json)?\n"
            f"2. Have you read the target Megatron base classes you're subclassing?\n"
            f"3. Have you read existing similar implementations in flagscale/models/?\n"
            f"4. Have you built a component mapping table?\n"
            f"Record your analysis with workspace_experiment or memory_write, then continue."
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
        if not is_full_training:
            return ""
        if self._verification_stage in ("distributed_ok", "full_training"):
            return ""
        return (
            f"\n\n[VERIFICATION LADDER] Stage is '{self._verification_stage}'. "
            f"Verify incrementally before full training:\n"
            f"1. Model init: python -c 'from <module> import <Model>; m = <Model>(cfg)'\n"
            f"2. Forward: 1 forward pass, compare loss with HF reference\n"
            f"3. Backward: --train-iters 1, check no crash\n"
            f"4. Distributed: 2 steps at target TP/PP, check no hang\n"
            f"Record each stage with workspace_experiment update."
        )

    _PORTING_READ_CATEGORIES = {
        "source_model": re.compile(r'modeling_|model\.py|config\.json|configuration_'),
        "megatron_base": re.compile(r'megatron/.*(transformer|attention|mlp|language_model|gpt_model|spec)'),
        "existing_impl": re.compile(r'flagscale/models/'),
        "checkpoint": re.compile(r'tools/checkpoint/|checkpoint_loader|convert'),
    }
    _MIN_CATEGORIES_BEFORE_WRITE = 3

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
            return ""  # No recent conversion, assume checkpoint is pre-verified

        # Check if verification was done after conversion
        has_verification = any(
            ("torch.load" in str(rest) or "state_dict" in str(rest) or "verify" in str(name).lower())
            and i > max(idx for idx, (n, *_r) in enumerate(recent_tools) if "convert" in str(n).lower())
            for i, (name, *rest) in enumerate(recent_tools)
        )

        if has_verification:
            return ""

        # Extract checkpoint path from command
        ckpt_match = re.search(r'--load[=\s]+([^\s]+)', cmd)
        ckpt_path = ckpt_match.group(1) if ckpt_match else "<checkpoint_path>"

        return f"""
[CHECKPOINT VERIFICATION GATE]

You just converted a checkpoint but haven't verified it before training. Checkpoint bugs waste 10+ minutes of model loading time.

**Critical checks** (30 seconds, catches 90% of conversion bugs):

```python
import torch

# 1. Reload and verify structure
ckpt = torch.load("{ckpt_path}/mp_rank_00/model_optim_rng.pt", map_location="cpu")
state = ckpt.get("model", ckpt)
print(f"Checkpoint keys: {{len(state)}}")
if len(state) == 0:
    print("ERROR: Empty checkpoint!")

# 2. Sample shapes and dtypes
for k, v in list(state.items())[:5]:
    print(f"  {{k}}: {{v.shape}} {{v.dtype}}")

# 3. Verify norms (not random init)
norms = [v.float().norm().item() for v in list(state.values())[:10]]
print(f"Sample norms: {{norms}}")
if all(0.01 < n < 0.1 for n in norms):
    print("WARNING: Norms look like random init, not converted weights")

# 4. Cross-check against model state_dict
with torch.device("meta"):
    model = build_model(config)  # Use your model builder
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
shape_mismatch = {{k for k in model_keys & ckpt_keys if model.state_dict()[k].shape != state[k].shape}}

print(f"Missing: {{len(missing)}}, Unexpected: {{len(unexpected)}}, Shape mismatch: {{len(shape_mismatch)}}")
if missing or shape_mismatch:
    print("ERROR: Key/shape mismatch detected!")
    for k in list(missing)[:5]:
        print(f"  Missing: {{k}}")
    for k in list(shape_mismatch)[:5]:
        print(f"  Shape: {{k}} model={{model.state_dict()[k].shape}} ckpt={{state[k].shape}}")
```

**Your decision**:
- Run verification (recommended) — 30 seconds now saves 10+ minutes later
- Skip if you've verified this exact conversion path before (state why)

Think carefully: is this checkpoint conversion new or have you tested it before?
"""

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

        # Try to extract model info from workspace or config
        model_type = "unknown"
        model_size = "unknown"
        custom_components = []

        try:
            ws_state = self._workspace_manager.get_current()
            if isinstance(ws_state, dict):
                model_type = ws_state.get("model_type", "unknown")
                model_size = ws_state.get("model_size", "unknown")
                custom_components = ws_state.get("custom_components", [])
        except:
            pass

        return f"""
[DISTRIBUTED PREREQUISITE GATE]

You're launching distributed training (TP={tp}, PP={pp}) without single-GPU verification.

**Model context**:
- Type: {model_type}
- Size: {model_size}
- Custom components: {custom_components if custom_components else "none detected"}

**Why single-GPU verification matters**:
When distributed training fails, it's hard to tell if the issue is:
- Model/data/config problem (would fail on single-GPU too)
- Parallelism implementation (TP sharding, PP stage assignment, NCCL)

Single-GPU verification isolates the first category in minutes instead of hours.

**Your options** (think about model characteristics):

1. **Run single-GPU first** (recommended for):
   - Dense models <30B params
   - First time porting this architecture
   - Custom layers that need TP implementation

   For large models: use smaller variant (reduce num_layers in config) or smaller batch

2. **Skip to distributed** (acceptable for):
   - MoE models where single-GPU is impractical (routing needs ≥2 GPUs)
   - Models >30B where even minimal config won't fit one GPU
   - You've verified similar model architecture before
   - This is iteration N of a working model (just config changes)

3. **Verify components separately** (middle ground):
   - Test custom layers with dummy model on single-GPU
   - Then launch full distributed training

**Decision required**: Choose an option and explain your reasoning. If skipping single-GPU, document why in workspace_experiment (action='add_attempt', change='Skip single-GPU verification', result='<your reason>').

Don't rush this decision — distributed debugging is expensive.
"""

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
        return """
[UNDERSTANDING VERIFICATION GATE]

**Context**: You're about to write training/model code, but haven't demonstrated understanding of the target system.

**Why this gate exists**: Writing code without understanding leads to "implement everything then debug" — the #1 cause of wasted hours in model migration.

**What you must do** — answer these 3 categories in workspace_experiment before writing code:

1. **Data flow**: What is the exact input format? What preprocessing happens? What does get_batch return (keys, shapes, dtypes)?

2. **Model signature**: What does the model's forward() expect? What does it return? What loss function is used?

3. **Parallelism context**: What parallelism does FlagScale apply to this model type? Which layers get tensor-parallelized? How does the data pipeline handle DP?

**Your options**:
- Write answers in workspace_experiment (action='create' or action='add_attempt') — unlocks this gate
- Skip if this is a trivial edit to existing working code (not a new implementation)

**Decision required**: Write your understanding to workspace_experiment, then proceed with implementation.

Prove you understand before you implement.
"""

    def _check_component_isolation_gate(self, tool_name, arguments):
        """A3: For multi-component models, require component-by-component plan following data flow order."""
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

        return """
[COMPONENT ISOLATION PLAN GATE]

**Context**: This appears to be a multi-component model (multimodal, MoE, or multi-encoder). You're about to write the training script without a component isolation plan.

**Why this gate exists**: Multi-component models fail in subtle ways when all components are implemented simultaneously. A bug in Component A manifests as wrong output in Component C — impossible to debug without isolation.

**What you must do** — create a plan (plan_create) with explicit component phases in DATA FLOW ORDER:

```
Phase 1: [First component in data flow] (e.g., ViT/encoder)
  - Implement
  - Verify forward (non-zero output, correct shape)
  - Verify backward (non-zero gradients)

Phase 2: [Next component] (e.g., LLM backbone)
  - Implement
  - Verify forward
  - Verify backward

Phase 3: [Integration]
  - Connect components
  - Verify end-to-end data flow
  - Verify gradient flow through all components
```

**Your options**:
- Create component plan via plan_create — unlocks this gate
- Document in workspace_experiment why isolation isn't needed (e.g., all components already verified in FlagScale)

**Decision required**: Create the component isolation plan before writing the training script.

Data flows in one direction. Verify in that same direction.
"""

    def _check_failure_mode_analysis_gate(self, tool_name, arguments):
        """B1: After writing training code, require failure mode analysis before first launch."""
        if self._failure_mode_analyzed or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Check if training code was recently written
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

        return """
[FAILURE MODE ANALYSIS GATE]

**Context**: You just wrote training code and are about to launch. But you haven't analyzed what's most likely to go wrong.

**Why this gate exists**: 80% of first-launch failures fall into predictable categories. Spending 2 minutes thinking about failure modes saves 20 minutes of debugging.

**What you must do** — document in workspace_experiment (action='add_attempt'):

1. **Top 3 most likely failure modes** for this specific model:
   - What could go wrong? (e.g., "shape mismatch in cross-attention between ViT and LLM")
   - How would you detect it? (e.g., "RuntimeError with shape info" or "loss = NaN")
   - What's the fix? (e.g., "check projection layer dimensions")

2. **One "what if I'm wrong" check**:
   - What assumption are you least confident about?
   - How can you verify it before launching?

**Your options**:
- Write failure mode analysis to workspace_experiment — unlocks this gate
- Skip if this is a re-launch of previously working code (no new code written)

**Decision required**: Document your failure mode analysis, then launch.

Think before you run. Debug before you debug.
"""

    # ── Phase 2 Gates: Pre-Launch ────────────────────────────────────────

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
            return ""  # Don't block dry-runs

        # Only trigger on first real training launch
        recent_launches = sum(
            1 for name, *rest in list(self._recent_tool_calls)[-30:]
            if name == "shell" and self._is_training_launch(str(rest))
            and not self._is_quick_test_command(str(rest))
        )
        if recent_launches > 0:
            return ""  # Already launched before, don't re-gate

        return """
[SANITY CHECK GATE]

**Context**: First real training launch detected. Have you verified the 4 critical components?

**Why this gate exists**: First launches fail 70%+ of the time. These 4 checks catch the most common issues in under 60 seconds total.

**Checklist** (run each, document results):

1. **Data check**: `python -c "from your_get_batch import get_batch; batch = get_batch(...); print({k: v.shape for k,v in batch.items()})"` — verify shapes match model expectations

2. **Model init check**: `python -c "model = YourModel(config); print(sum(p.numel() for p in model.parameters()))"` — verify model builds without error

3. **Config check**: Verify TP × PP × DP = world_size, and batch_size is divisible by DP

4. **Checkpoint check** (if loading): Verify checkpoint keys match model state_dict keys (use the verification script from checkpoint gate)

**Your options**:
- Run all 4 checks (recommended for new models)
- Run checks 1+3 only (acceptable if model init already verified via dry-run)
- Skip all (only if this exact config+code ran successfully before)

**Decision required**: Run the checks or explain why each skipped check is safe to skip.
"""

    def _check_config_model_consistency_gate(self, tool_name, arguments):
        """B4: After generating config, verify config keys match model __init__ parameters."""
        if self._config_model_verified or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Check if config was recently written
        recent_writes = [
            str(rest) for name, *rest in list(self._recent_tool_calls)[-15:]
            if name == "write_file" and re.search(r'\.yaml|\.yml|config', str(rest))
        ]
        if not recent_writes:
            return ""

        return """
[CONFIG-MODEL CONSISTENCY GATE]

**Context**: You recently wrote a config file and are launching training. Config-model mismatches are a top-5 failure cause.

**Why this gate exists**: YAML config keys must exactly match what the model's __init__ and argument parser expect. A typo or wrong key name silently uses defaults — causing subtle bugs that waste hours.

**What you must do**:
1. Read the model's `__init__` signature (or `add_model_args` function)
2. Compare each config key against the expected parameter names
3. Verify value types match (int vs str, list vs scalar)

**Common mismatches to check**:
- `hidden_size` vs `hidden_dim` vs `d_model`
- `num_attention_heads` vs `n_heads` vs `num_heads`
- `ffn_hidden_size` vs `intermediate_size` vs `d_ff`
- `num_layers` vs `n_layers` vs `num_hidden_layers`
- Boolean flags that default to False if misspelled

**Your options**:
- Verify manually (read model code, compare with config)
- Run quick script: `python -c "import yaml; cfg = yaml.safe_load(open('config.yaml')); print(cfg.keys())"` then compare
- Skip if using an existing verified config (no new keys added)

**Decision required**: Confirm zero mismatches or explain each mismatch.
"""

    def _check_environment_consistency_gate(self, tool_name, arguments):
        """C2: Before training launch, verify installed packages are from correct paths."""
        if self._env_verified or not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Only trigger once per session
        self._env_verified = True

        return """
[ENVIRONMENT CONSISTENCY GATE]

**Context**: About to launch training. Package version/path conflicts are a silent killer.

**Common issue**: Megatron-LM-FL updates faster than FlagScale adapts. If you installed Megatron-LM-FL from `main` branch but FlagScale expects an older tag, you'll hit import errors or API mismatches that look like your code is wrong but are actually version drift.

**Quick verification** (10 seconds):
```python
python -c "
import megatron; print('megatron:', megatron.__file__)
import transformer_engine; print('TE:', transformer_engine.__file__)
try:
    import flagscale; print('flagscale:', flagscale.__file__)
except: pass
"
```

**What to check**:
- All paths should point to your current working environment
- If FlagScale wrapper fails on imports: check FlagScale's install docs for the compatible Megatron-LM-FL tag, then `pip install git+...@<tag>` to roll back
- If rolling back is impractical, use direct torchrun instead of the FlagScale wrapper

**If paths are wrong**: `pip install -e .` in the correct directory, or adjust PYTHONPATH.

This is informational — proceed with training, but investigate version compatibility if you get unexpected import or API errors.
"""

    def _check_component_integration_gate(self, tool_name, arguments):
        """C4: Before full training with all components, verify each was tested individually."""
        if self._component_integration_verified or not self._porting_mode:
            return ""
        if not self._component_plan_created:
            return ""  # Only applies if component plan was required
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""
        if self._is_quick_test_command(cmd):
            return ""

        # Check verification stage — should be at least forward_aligned
        if self._verification_stage in ("forward_aligned", "backward_ok", "distributed_ok", "full_training"):
            self._component_integration_verified = True
            return ""

        return """
[COMPONENT INTEGRATION GATE]

**Context**: Launching full training with a multi-component model, but individual component verification is incomplete.

**Why this gate exists**: When multiple components fail simultaneously, it's nearly impossible to isolate which one is broken. Verify each component works alone before combining.

**What you must verify** (check workspace_experiment or logs for evidence):
- Each component's forward pass produces non-zero output
- Each component's backward pass produces non-zero gradients
- Data flows correctly between components (shapes match at boundaries)

**Your options**:
- Show evidence of per-component verification (logs, workspace records)
- Run quick per-component tests now before full launch
- Skip if all components are standard FlagScale modules (already tested upstream)

**Decision required**: Provide evidence of component verification or explain why it's safe to skip.
"""

    # ── Phase 3 Gates: Post-Launch (Informational) ───────────────────────

    def _check_gradient_health(self, cmd, result):
        """D1: After first iteration, check grad_norm is non-zero and finite."""
        if not self._porting_mode:
            return ""
        if not self._is_training_launch(cmd):
            return ""
        if not result:
            return ""

        # Look for gradient info in output
        grad_norm_match = re.search(r'grad.norm[:\s]+([0-9.eE+\-]+|nan|inf)', result, re.I)
        if not grad_norm_match:
            return ""

        grad_val = grad_norm_match.group(1).lower()
        issues = []

        if grad_val in ("nan", "inf"):
            issues.append(f"grad_norm is {grad_val} — likely exploding gradients or NaN in forward pass")
        elif grad_val == "0.0" or grad_val == "0":
            issues.append("grad_norm is 0 — model may not be receiving gradients (frozen params? detached tensor?)")
        else:
            try:
                val = float(grad_val)
                if val > 1000:
                    issues.append(f"grad_norm={val} is very large — consider gradient clipping or lower learning rate")
                elif val < 1e-10:
                    issues.append(f"grad_norm={val} is near-zero — check if loss is connected to all parameters")
            except ValueError:
                pass

        # Check zero_grad_ratio if present
        zero_grad_match = re.search(r'zero.grad.*?([0-9.]+)%', result, re.I)
        if zero_grad_match:
            ratio = float(zero_grad_match.group(1))
            if ratio > 50:
                issues.append(f"zero_grad_ratio={ratio}% — over half of parameters have zero gradients")

        if not issues:
            return ""

        return f"""
[GRADIENT HEALTH CHECK — INFORMATIONAL]

Issues detected in training output:
{chr(10).join(f'- {issue}' for issue in issues)}

**Common causes**:
- NaN/Inf: dtype overflow (try bf16→fp32 for problematic layers), bad initialization, division by zero
- Zero gradients: frozen parameters, detached tensors in forward, loss not connected to all components
- Very large: missing gradient clipping, learning rate too high for model size

**Suggested action**: Check which parameters have zero/abnormal gradients:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{{name}}: grad_norm={{param.grad.norm().item():.6f}}")
    else:
        print(f"{{name}}: NO GRADIENT")
```
"""

    def _check_loss_sanity(self, cmd, result):
        """D2: After step 0 and step 10, check loss is reasonable."""
        if not self._porting_mode:
            return ""
        if not result:
            return ""

        # Extract loss values
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

        # Check step 0 loss against expected random init value
        # For random init: loss ≈ ln(vocab_size). Common vocab sizes: 32k→10.4, 64k→11.1, 128k→11.8, 256k→12.4
        if first_loss > 20:
            issues.append(f"Initial loss={first_loss:.2f} is unusually high (expected ~10-12 for random init with typical vocab)")
        elif first_loss < 0.1:
            issues.append(f"Initial loss={first_loss:.4f} is suspiciously low — possible data leak or label issue")
        elif 0.1 < first_loss < 5 and len(losses) == 1:
            # Could be checkpoint-loaded, which is fine
            pass

        # Check if loss is decreasing (if we have multiple values)
        if len(losses) >= 3:
            if all(losses[i] >= losses[i-1] for i in range(1, min(5, len(losses)))):
                issues.append(f"Loss is not decreasing over first {len(losses)} steps: {losses[:5]} — check learning rate or data pipeline")
            if any(l != l or l == float('inf') for l in losses):  # NaN check
                issues.append("Loss contains NaN or Inf values — critical issue")

        if not issues:
            return ""

        return f"""
[LOSS SANITY CHECK — INFORMATIONAL]

Issues detected:
{chr(10).join(f'- {issue}' for issue in issues)}

**Reference values**:
- Random init (no checkpoint): loss ≈ ln(vocab_size) — typically 10-12
- Pretrained checkpoint: loss should be much lower (2-5 typical)
- If loss = ln(vocab_size) WITH checkpoint loading: checkpoint was NOT loaded correctly

**If loss is not decreasing**:
- Learning rate too low or too high
- Data pipeline returning constant/wrong data
- Model parameters frozen or not connected to optimizer
- Wrong loss function for the task
"""

    def _check_component_gradient_flow(self, cmd, result):
        """D3: For multi-component models, check all components receive gradients."""
        if not self._porting_mode or not self._component_plan_created:
            return ""
        if not result:
            return ""

        # Look for component-specific gradient info
        no_grad_components = re.findall(r'(vision|vit|encoder|decoder|vae|router|expert).*?grad.*?(?:None|0\.0)', result, re.I)
        if no_grad_components:
            components = list(set(c.lower() for c in no_grad_components))
            return f"""
[COMPONENT GRADIENT FLOW CHECK — INFORMATIONAL]

Components with zero/missing gradients detected: {', '.join(components)}

**This means**: These components are not learning. Possible causes:
- Component output is detached from the computation graph (`.detach()` or `torch.no_grad()`)
- Component is frozen (`requires_grad=False`) unintentionally
- Loss function doesn't flow through this component
- Projection layer between components breaks gradient flow

**Verification**:
```python
# Add after loss.backward():
for name, param in model.named_parameters():
    if any(comp in name.lower() for comp in {components}):
        has_grad = param.grad is not None and param.grad.norm() > 0
        print(f"{{name}}: has_grad={{has_grad}}")
```

**Fix**: Ensure the forward pass maintains gradient flow through all components. Check for `.detach()`, `@torch.no_grad()`, or missing connections.
"""
        return ""

    def _check_checkpoint_numerical_verification(self, cmd, result):
        """D4: After checkpoint loading, verify tensor statistics match HF originals."""
        if not self._porting_mode:
            return ""
        if not result:
            return ""
        # Trigger on checkpoint load indicators
        if not re.search(r'(?:loaded|loading|checkpoint|ckpt).*(?:success|done|complete)|successfully loaded', result, re.I):
            return ""
        # Check if numerical verification was already done
        if re.search(r'mean.*std.*match|tensor.*verification.*pass|numerical.*check.*ok', result, re.I):
            return ""
        return """
[CHECKPOINT NUMERICAL VERIFICATION — ACTION REQUIRED]

Checkpoint loaded, but numerical correctness is NOT verified.
Key mapping alone does NOT guarantee correct loading — tensors can be transposed, permuted, or scaled differently.

**MANDATORY verification** (add to your conversion/loading script):
```python
import torch
# After loading converted checkpoint into Megatron model:
# Compare a few key tensors against HF originals
hf_state = torch.load("hf_model.pt", map_location="cpu")
mg_state = model.state_dict()

checks = [
    ("embed_tokens", "language_model.embedding.word_embeddings.weight"),
    ("layers.0.self_attn.q_proj", "decoder.layers.0.self_attention.linear_qkv.weight"),
    # Add model-specific mappings
]
for hf_key, mg_key in checks:
    hf_t = hf_state[hf_key]
    mg_t = mg_state[mg_key]
    # For TP-sharded: compare full tensor or first shard
    print(f"{hf_key}: HF mean={hf_t.float().mean():.6f} std={hf_t.float().std():.6f}")
    print(f"{mg_key}: MG mean={mg_t.float().mean():.6f} std={mg_t.float().std():.6f}")
    assert torch.allclose(hf_t.float().mean(), mg_t.float().mean(), atol=1e-4), f"MISMATCH: {hf_key}"
```

**Why this matters**: In the Bagel migration, key mapping appeared correct but tensor shapes were wrong (QKV packed differently), causing silent numerical divergence.
"""

    def _check_gpu_zombie_escalation(self, cmd, result):
        """Detect GPU zombie processes and provide escalation strategy."""
        if not result:
            return ""
        # Trigger on nvidia-smi showing memory used but no active process, or CUDA OOM
        zombie_indicators = [
            re.search(r'CUDA out of memory', result, re.I),
            re.search(r'RuntimeError.*CUDA.*OOM', result, re.I),
            re.search(r'No running processes found.*MiB.*[1-9]', result),
            re.search(r'memory.used.*[1-9]\d{3,}.*MiB.*\|\s*0%', result),
        ]
        if not any(zombie_indicators):
            return ""
        return """
[GPU ZOMBIE DETECTED — ESCALATION STRATEGY]

GPU memory is occupied but no active training process is running. This blocks all future launches.

**Escalation steps** (try in order):
1. `nvidia-smi` — identify PIDs using GPU memory
2. `kill -9 <PID>` — kill specific zombie processes
3. `fuser -v /dev/nvidia*` — find ALL processes holding GPU device files
4. `fuser -k /dev/nvidia*` — force-kill all GPU-holding processes
5. `python -c "import torch; torch.cuda.empty_cache()"` — release cached memory
6. If still stuck: `nvidia-smi --gpu-reset -i <gpu_id>` (CAUTION: resets GPU state)
7. Last resort: container/node restart may be needed

**Prevention**: Always use `pkill -f torchrun` or `pkill -f python.*train` before relaunching.
Avoid `kill -9` on the parent process without also killing children — orphaned workers hold GPU memory.
"""

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
        # Extract imports from the file being written
        imports = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
        critical_imports = [
            imp for imp in imports
            if any(kw in imp for kw in ("megatron", "transformer_engine", "flagscale", "apex"))
        ]

        if not critical_imports:
            return ""

        import_list = "\n".join(f"  python -c \"import {imp.split('.')[0]}; print({imp.split('.')[0]}.__file__)\"" for imp in set(imp.split('.')[0] for imp in critical_imports))

        return f"""
[IMPORT VERIFICATION GATE]

**Context**: Writing training script with critical infrastructure imports. Unresolved imports waste the entire first launch attempt.

**Why this gate exists**: Import errors are the #1 cause of "launch, wait 30s for model load, crash" cycles. 5 seconds of verification prevents this.

**Critical imports detected**:
{chr(10).join(f'- {imp}' for imp in critical_imports[:10])}

**Quick verification** (run before writing):
```bash
{import_list}
```

**Your options**:
- Run import verification (recommended) — 5 seconds, catches path issues
- Skip if you already verified these imports earlier in this session
- Skip if using exact same imports as a working reference implementation

**Decision required**: Verify imports resolve or explain why you're confident they will.
"""

    def _check_tp_compatibility_gate(self, tool_name, arguments):
        """C3: Before TP>1 with custom layers, verify sharded_state_dict() exists."""
        if not self._porting_mode:
            return ""
        if tool_name != "shell":
            return ""
        cmd = arguments.get("command", "")
        if not self._is_training_launch(cmd):
            return ""

        # Check if TP > 1
        tp = self._extract_arg_value(cmd, r'tensor.model.parallel.size[=\s]+(\d+)')
        if tp <= 1:
            tp = self._extract_arg_value(cmd, r'--tp[=\s]+(\d+)')
        if tp <= 1:
            return ""

        # Check if custom layers were written recently
        recent_writes = [
            str(rest) for name, *rest in list(self._recent_tool_calls)[-30:]
            if name == "write_file" and re.search(r'model|layer|module|attention', str(rest), re.I)
        ]
        has_custom_layers = any(
            re.search(r'class.*Module|class.*Layer|class.*Attention', w)
            for w in recent_writes
        )

        if not has_custom_layers:
            return ""

        return f"""
[TP COMPATIBILITY GATE]

**Context**: Launching with TP={tp} and custom layers detected. Custom layers need explicit tensor-parallel support.

**Why this gate exists**: Custom layers without `sharded_state_dict()` or proper column/row parallel linear layers will either crash or silently produce wrong results with TP>1.

**What to verify for each custom layer**:
1. Does it use `ColumnParallelLinear` / `RowParallelLinear` instead of `nn.Linear`?
2. Does it implement `sharded_state_dict()` for checkpoint save/load with TP?
3. Are attention heads divisible by TP? (num_heads % TP == 0)
4. Are hidden dimensions divisible by TP where needed?

**Your options**:
- Verify custom layers have TP support (grep for sharded_state_dict, ColumnParallelLinear)
- Launch with TP=1 first to verify correctness, then add TP support
- Skip if custom layers are thin wrappers around existing Megatron modules (they inherit TP support)

**Decision required**: Verify TP compatibility or explain why custom layers are TP-safe.
"""

    def _check_reference_comparison_gate(self, tool_name, arguments):
        """A2: When source model has runnable training, require comparison plan."""
        if self._reference_comparison_planned or not self._porting_mode:
            return ""
        if tool_name != "write_file":
            return ""
        path = arguments.get("path", "")
        if not re.search(r'train_.*\.py|pretrain_', path):
            return ""

        # Check if reference implementation exists (detected from reads)
        recent_reads = " ".join(
            str(rest) for name, *rest in list(self._recent_tool_calls)[-30:]
            if name == "read_file"
        )
        has_reference = any(
            kw in recent_reads.lower()
            for kw in ("huggingface", "transformers", "reference", "original", "source_model")
        )

        if not has_reference:
            self._reference_comparison_planned = True  # No reference available
            return ""

        return """
[REFERENCE COMPARISON STRATEGY GATE]

**Context**: You've read a reference implementation and are writing the FlagScale version. Have you planned what to compare at each step?

**Why this gate exists**: Without a comparison plan, you'll finish the port and have no way to verify correctness until full training — by which time bugs are hard to isolate.

**What you must plan** (document in workspace_experiment or plan):

1. **get_batch output comparison**: Same input → same batch tensors?
2. **Forward output comparison**: Same input → same logits/loss (within fp tolerance)?
3. **Gradient comparison** (optional): Same input → similar gradient norms?

For each comparison point:
- What input will you use? (same sample, same seed)
- What tolerance is acceptable? (exact match? 1e-5? 1e-3?)
- How will you run the reference? (HF script? saved tensors?)

**Your options**:
- Create comparison plan in workspace_experiment — unlocks this gate
- Skip if no runnable reference exists (only config.json available)
- Skip if this is a standard architecture with known-correct FlagScale implementation

**Decision required**: Document your comparison strategy or explain why comparison isn't feasible.
"""

    def _extract_arg_value(self, cmd: str, pattern: str) -> int:
        """Extract integer argument value from command."""
        match = re.search(pattern, cmd)
        return int(match.group(1)) if match else 1

    # ── Infinite loop / duplicate tool call detection ──────────────────

    _LOOP_DETECTION_WINDOW = 10
    _LOOP_DETECTION_THRESHOLD = 3
    _AUTOSAVE_INTERVAL = 5  # Save state every N tool calls within a turn

    def _get_tool_call_key(self, tool_name, arguments):
        """Generate a hashable key for a tool call.

        For read_file, include start_line/end_line to distinguish
        different parts of the same file.
        """
        if tool_name == "shell":
            return (tool_name, arguments.get("command", ""))
        elif tool_name == "read_file":
            path = arguments.get("path", "")
            start_line = arguments.get("start_line", 0)
            end_line = arguments.get("end_line", 0)
            return (tool_name, path, start_line, end_line)
        elif tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "") or arguments.get("file_path", "")
            # For edit_file, include a hash of old_string to distinguish different edits
            if tool_name == "edit_file":
                old_str = arguments.get("old_string", "")
                old_hash = hash(old_str[:200]) if old_str else 0
                return (tool_name, path, old_hash)
            return (tool_name, path)
        elif tool_name == "load_skill":
            return (tool_name, arguments.get("name", ""))
        else:
            # For other tools, use first 2 args
            key_parts = []
            for k, v in list(arguments.items())[:2]:
                key_parts.append(f"{k}={str(v)[:100]}")
            return (tool_name, "|".join(key_parts))

    def _check_loop_detection(self, tool_name, arguments):
        """Detect repeated identical tool calls that indicate the agent is stuck."""
        key = self._get_tool_call_key(tool_name, arguments)
        self._recent_tool_calls.append(key)
        if len(self._recent_tool_calls) > self._LOOP_DETECTION_WINDOW:
            self._recent_tool_calls = self._recent_tool_calls[-self._LOOP_DETECTION_WINDOW:]

        # Count occurrences of this exact call in recent window
        count = self._recent_tool_calls.count(key)
        if count >= self._LOOP_DETECTION_THRESHOLD:
            return (
                f"\n\n⚠️ [LOOP DETECTION] You've called {tool_name} with the same arguments "
                f"{count} times in the last {self._LOOP_DETECTION_WINDOW} tool calls. "
                f"This suggests you're stuck in a loop.\n"
                f"STOP and take a different approach:\n"
                f"1. Diagnose WHY the previous attempts failed\n"
                f"2. Try a fundamentally different strategy\n"
                f"3. If blocked, write what you know to workspace and ask the user\n"
            )
        return ""

    def _check_duplicate_read(self, tool_name, arguments):
        """Detect duplicate tool calls — within-turn cache hit or cross-compaction re-read."""
        if tool_name == "read_file":
            path = arguments.get("path", "")
            if not path:
                return None
            # Include line range in cache key so different ranges aren't treated as duplicates
            start = arguments.get("start_line", "")
            end = arguments.get("end_line", "")
            key = ("read_file", path, str(start), str(end))
            # Within-turn cache hit: return cached content
            if key in self._tool_call_cache:
                return self._tool_call_cache[key]
            # Cross-compaction re-read: file was read before but content was compacted away
            if path in self._files_read_this_session and key not in self._tool_call_cache:
                return None  # Allow the read but we'll add a note in post-processing
        elif tool_name == "memory_write":
            mem_key = arguments.get("key", "")
            if not mem_key:
                return None
            key = ("memory_write", mem_key)
        else:
            return None
        if key in self._tool_call_cache:
            return self._tool_call_cache[key]
        return None

    def _cache_tool_result(self, tool_name, arguments, result):
        """Cache tool results within a turn to avoid redundant calls."""
        if tool_name == "read_file" and "ERROR" not in result[:20]:
            path = arguments.get("path", "")
            if path:
                start = arguments.get("start_line", "")
                end = arguments.get("end_line", "")
                self._tool_call_cache[("read_file", path, str(start), str(end))] = result
        elif tool_name == "memory_write" and "ERROR" not in result[:20]:
            mem_key = arguments.get("key", "")
            if mem_key:
                self._tool_call_cache[("memory_write", mem_key)] = result

    # ── Error-to-skill auto-loading ────────────────────────────────────

    def _check_skill_lifecycle(self):
        """Unload skills that are no longer needed to save tokens."""
        if not self._active_skill_content:
            return
        to_unload = []
        for skill_name, loaded_at_iter in list(self._skill_load_iterations.items()):
            age = self._total_iterations - loaded_at_iter
            # Never unload model-porter during porting mode
            if self._porting_mode and "model-porter" in skill_name:
                continue
            # Never unload data-prep during data-prep mode
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
            self._refresh_system_prompt()  # Update prompt to remove unloaded skill content
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
            # Short placeholder in history — full content lives in system prompt
            self.history.append(self.provider.format_tool_result(
                tool_call_id, f"[Skill '{skill_name}' loaded — content available in system context]"))
            self._loaded_skills.add(skill_name)
            # Track for skill lifecycle management
            self._active_skill_content[skill_name] = content
            self._skill_load_iterations[skill_name] = self._total_iterations
            self._refresh_system_prompt()
            display.skill_auto_loaded(skill_name)
            return f"\n[Auto-loaded skill '{skill_name}' for {pattern} error]\n"
        except Exception:
            return ""

    # Phase transition gate
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
                "2. Have you run a dry-run (--train-iters 2) successfully?\n"
                "If yes, proceed to verification. If no, continue implementation."
            ),
        },
        "verification": {
            "exit_conditions": ["verification_stage >= forward_aligned"],
            "next_phase": "done",
            "message": (
                "\n\n[PHASE TRANSITION] Verification phase in progress. Checklist:\n"
                "1. Model init: instantiate model, check param count\n"
                "2. Forward: compare loss/logits with reference\n"
                "3. Backward: check gradients\n"
                "4. Distributed: verify at target TP/PP\n"
                "Record each stage with workspace_experiment."
            ),
        },
    }

    def _check_phase_transition(self, tool_name):
        """Inject phase transition reminders when conditions met."""
        if not self._porting_mode:
            return ""

        # Track tool counts per phase
        self._phase_tool_counts[self._current_phase] = self._phase_tool_counts.get(self._current_phase, 0) + 1

        # Check if current phase should transition
        if self._current_phase == "idle":
            # First read_file or load_skill transitions to analysis
            if tool_name in ("read_file", "load_skill"):
                self._current_phase = "analysis"
            return ""

        if self._current_phase == "analysis":
            # Check exit conditions
            read_count = len(self._files_read_this_session)
            categories = len(self._reading_categories)
            tool_count = self._phase_tool_counts.get("analysis", 0)

            # Transition after 15+ tools in analysis phase
            if tool_count >= 15 and read_count >= 8 and categories >= 3:
                if tool_name in ("write_file", "edit_file"):
                    self._current_phase = "implementation"
                    return self._PHASE_TRANSITIONS["analysis"]["message"]

        elif self._current_phase == "implementation":
            # Transition after dry run passes
            if self._dry_run_passed:
                self._current_phase = "verification"
                return self._PHASE_TRANSITIONS["implementation"]["message"]

        elif self._current_phase == "verification":
            # Transition after verification stage advances
            if self._verification_stage in ("forward_aligned", "backward_ok", "distributed_ok", "full_training"):
                self._current_phase = "done"
                return self._PHASE_TRANSITIONS["verification"]["message"]

        return ""

    _READ_FILE_SUMMARY_THRESHOLD = 4000
    _READ_FILE_SUMMARY_THRESHOLD_PORTING = 12000  # Porting needs full model code

    _SKILL_SUMMARY_THRESHOLD = 3000

    def _maybe_summarize_skill(self, skill_name, content):
        """Replace large skill content with summary if SUMMARY.md exists, or auto-generate one."""
        if len(content) <= self._SKILL_SUMMARY_THRESHOLD:
            return content
        summary = self.skill_manager.load_summary(skill_name)
        if summary is None:
            summary = self._auto_generate_skill_summary(content)
        return (
            f"<skill name=\"{skill_name}\" mode=\"summary\">\n{summary}\n</skill>\n\n"
            f"[Full skill content: {len(content)} chars. "
            f"Use read_file on the SKILL.md for specific sections.]"
        )

    @staticmethod
    def _auto_generate_skill_summary(content):
        """Extract headers + first paragraph of each section as a fallback summary."""
        lines = content.splitlines()
        summary_lines = []
        in_section = False
        section_para_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                summary_lines.append(stripped)
                in_section = True
                section_para_lines = 0
            elif in_section and stripped:
                if section_para_lines < 2:
                    summary_lines.append(stripped)
                    section_para_lines += 1
            elif not stripped:
                in_section = True
                section_para_lines = 0
        # Cap at ~80 lines
        return "\n".join(summary_lines[:80])

    def _summarize_file_content(self, content, path):
        """Summarize large file reads to save context budget."""
        if len(content) <= self._READ_FILE_SUMMARY_THRESHOLD:
            return content
        lines = content.splitlines()

        signatures = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('class ', 'def ', 'async def ')):
                signatures.append(f"  L{i+1}: {stripped}")
        sig_block = "\n".join(signatures[:30]) if signatures else "(no class/def found)"

        head = "\n".join(lines[:15])
        tail = "\n".join(lines[-10:])

        return (
            f"[File: {path}, {len(lines)} lines, {len(content)} chars]\n"
            f"[Structure:]\n{sig_block}\n\n"
            f"[First 15 lines:]\n{head}\n\n"
            f"[... {len(lines) - 25} lines omitted — use read_file with offset/limit for details ...]\n\n"
            f"[Last 10 lines:]\n{tail}"
        )

    def _is_context_limit_error(self, e):
        """Check if an exception is a context limit error."""
        msg = str(e).lower()
        return any(kw in msg for kw in ("400", "context length", "too many tokens", "token limit", "maximum context"))

    def _track_training_failures(self, tool_calls, results):
        """Track consecutive training failures with pattern-based escalation."""
        for tc, result in zip(tool_calls, results):
            if tc["name"] != "shell":
                continue
            cmd = tc["arguments"].get("command", "")
            if not self._TRAIN_CMD_RE.search(cmd):
                continue

            # Distinguish verification scripts from full training
            is_verification = bool(re.search(r'verify|dryrun|test_model', cmd, re.I))

            if self._TRAIN_FAIL_RE.search(result[:2000]):
                self._consecutive_train_failures += 1
                reason = result[:200].split('\n')[0]
                self._last_train_failure_reasons.append(reason)
                pattern = self._classify_error_pattern(result[:2000])
                self._error_pattern_history.append(pattern)

                # Auto-checkpoint training failure to memory
                try:
                    error_summary = self._extract_error_summary(result[:2000])
                    self.session_memory.put(
                        f"train_failure_{self._consecutive_train_failures}",
                        "finding",
                        f"Training failure #{self._consecutive_train_failures} "
                        f"(pattern: {pattern}): {error_summary[:300]}. "
                        f"Cmd: {cmd[:100]}",
                        self._session_id,
                        task=self._workspace_manager.get_current_task(),
                    )
                except Exception:
                    pass

                # For verification scripts: 2 failures → force audit (lower threshold)
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

                # Same pattern twice → force root cause analysis (earlier than generic escalation)
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
            else:
                self._consecutive_train_failures = 0
                self._last_train_failure_reasons.clear()
                self._error_pattern_history.clear()

    def _run_poll_mode(self, command, last_output, tool_call_id):
        """Execute poll loop locally without LLM calls. Returns (output, count, elapsed, reason, routine_changes).

        Only returns to the LLM when the output change is "interesting"
        (errors, training metrics, large jumps) or on timeout.  Routine
        changes (e.g., line count +1) are absorbed silently.
        """
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

    # ── ReAct loop ───────────────────────────────────────────────────────

    def _react_loop(self):
        self._turn_count += 1
        self._interrupted = False
        self._turn_iteration_count = 0
        turn_start = time.time()
        turn_input_tokens = 0
        turn_output_tokens = 0
        max_iter = self.config.max_iterations
        iteration = 0

        # Install SIGINT handler so Ctrl+C works even during non-IO phases
        _prev_handler = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            self._interrupted = True
            display.interrupted()

        signal.signal(signal.SIGINT, _sigint_handler)

        while iteration < max_iter:
            if self._interrupted:
                break

            # Dynamic schema filtering by detected phase
            phase = self._detect_tool_phase()
            schemas = self._get_filtered_schemas(phase)
            self._extra_tools_next_iter = set()

            t0 = time.time()
            messages = self.history.get_messages()

            if self.history._last_compacted_from:
                display.context_compacted(
                    self.history._last_compacted_from,
                    self.history._last_compacted_to,
                    compaction_num=self.history.compaction_count,
                    ratio=self.history.last_compaction_ratio,
                )
                # Restore enforcement state from compaction summary
                self._restore_state_from_compaction()

            display.thinking()

            try:
                response, usage = self._call_llm_stream(messages, schemas)
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break
            except Exception as e:
                # Context limit auto-recovery: compact and retry once
                if self._is_context_limit_error(e):
                    display.thinking_clear()
                    logger.warning("Context limit hit, forcing compact and retry: %s", e)
                    print(display.yellow("⚠ Context limit hit — compacting and retrying..."))
                    self.history.force_compact(target_ratio=0.60)
                    messages = self.history.get_messages()
                    try:
                        display.thinking()
                        response, usage = self._call_llm_stream(messages, schemas)
                    except Exception as e2:
                        display.thinking_clear()
                        print(display.red(f"✖ LLM error after compact: {e2}"))
                        logger.exception("LLM call failed after compact retry")
                        break
                else:
                    display.thinking_clear()
                    print(display.red(f"✖ LLM error: {e}"))
                    logger.exception("LLM call failed")
                    break

            elapsed = time.time() - t0

            input_tok = usage.get("input_tokens") or 0
            output_tok = usage.get("output_tokens") or 0
            if input_tok:
                turn_input_tokens += input_tok
                self._session_input_tokens += input_tok
                self.history.report_actual_tokens(input_tok)
            if output_tok:
                turn_output_tokens += output_tok
                self._session_output_tokens += output_tok

            display.llm_done(elapsed, input_tok, output_tok)

            # Check interrupt after LLM call completes
            if self._interrupted:
                break

            logger.info("LLM call #%d: %.1fs", iteration + 1, elapsed)

            self.history.append(self.provider.format_assistant_message(response))

            if not response["tool_calls"]:
                break

            print()  # visual gap before tool block

            tool_t0 = time.time()
            try:
                results = self._execute_tools(response["tool_calls"])
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break
            tool_elapsed_total = time.time() - tool_t0
            tool_elapsed_list = [tool_elapsed_total] if len(response["tool_calls"]) == 1 else [tool_elapsed_total]

            tool_results = [
                self.provider.format_tool_result(tc["id"], result)
                for tc, result in zip(response["tool_calls"], results)
            ]
            self._append_tool_results(tool_results)

            # Track tool calls for phase detection and skill lifecycle
            for tc in response["tool_calls"]:
                self._last_tool_calls_deque.append(tc["name"])
                # Fallback: if LLM used a tool not in current schema, ensure it's available next time
                phase_tools = self._PHASE_TOOL_SETS.get(phase)
                if phase_tools is not None and tc["name"] not in (phase_tools | self._CORE_TOOLS):
                    self._extra_tools_next_iter.add(tc["name"])
                    logger.info("Tool '%s' not in phase '%s' schema — adding for next iteration", tc["name"], phase)
            self._total_iterations += 1
            self._turn_iteration_count += 1

            # Skill lifecycle check (may unload skills and refresh prompt)
            self._check_skill_lifecycle()

            # Refresh system prompt if a skill was loaded this iteration
            if any(tc["name"] == "load_skill" for tc in response["tool_calls"]):
                self._refresh_system_prompt()

            self._record_iteration(response["tool_calls"], results, output_tok, tool_elapsed_list)

            # Intra-turn sliding window: compress old intermediate results every 5 iterations
            if iteration > 0 and iteration % 5 == 0:
                if self.history.compact_intra_turn(keep_last=4):
                    logger.info("Intra-turn compaction at iteration %d", iteration)

            # Track consecutive training launch failures
            self._track_training_failures(response["tool_calls"], results)

            if self._detect_poll_pattern():
                last_iter = self._recent_iters[-1]
                command = last_iter["command"]
                last_output = last_iter["output"]
                tc = response["tool_calls"][0]

                new_output, poll_count, poll_elapsed, reason, routine_changes = self._run_poll_mode(
                    command, last_output, tc["id"])
                display.poll_mode_end(reason, poll_count, poll_elapsed, routine_changes)

                self._replace_last_tool_result(
                    self.provider.format_tool_result(tc["id"], new_output))
                self._recent_iters.clear()

            print()  # visual gap after tool block

            # Context pressure check — AFTER tool execution, before next LLM call
            pressure_warning = self._check_context_pressure()
            if pressure_warning:
                msgs = self.history.messages
                if msgs and msgs[-1].get("role") == "user":
                    last = msgs[-1]
                    content = last.get("content", "")
                    if isinstance(content, list):
                        content.append({"type": "text", "text": pressure_warning})
                    elif isinstance(content, str):
                        last["content"] = content + "\n\n" + pressure_warning
                else:
                    self.history.append({"role": "user", "content": pressure_warning})

            # Reset per-turn caches at end of each iteration
            self._tool_call_cache = {}

            iteration += 1

            if iteration >= max_iter and not self._interrupted:
                added = self._ask_continue(iteration)
                if added == 0:
                    break
                max_iter += added
                self.config.max_iterations = max_iter

        signal.signal(signal.SIGINT, _prev_handler)
        turn_elapsed = time.time() - turn_start
        display.turn_summary(self._turn_count, turn_elapsed, turn_input_tokens, turn_output_tokens)
        self._autosave()

    def _ask_continue(self, iteration: int) -> int:
        """Ask user whether to continue after hitting iteration limit.

        Returns the number of iterations to add (0 = stop).
        In auto mode, continues automatically without prompting.
        """
        if self.config.mode == "auto":
            print(f"\n\033[33m⚠ Reached {iteration} iterations (auto mode: +50).\033[0m")
            return 50
        print(f"\n\033[33m⚠ Reached {iteration} iterations.\033[0m")
        try:
            answer = input("   Continue? [Y/n] (or enter a number to add iterations): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("   Stopping.")
            return 0
        lower = answer.lower()
        if lower in ("", "y", "yes"):
            return 50
        if answer.isdigit() and int(answer) > 0:
            added = int(answer)
            print(f"   Continuing (+{added}, new limit: {iteration + added} iterations).")
            return added
        print("   Stopping.")
        return 0

    # ── LLM streaming with error recovery (P0-3) ────────────────────────

    def _call_llm_stream(self, messages, schemas):
        content_parts = []
        tool_calls = []
        tool_calls_by_id = {}
        current_tool = None
        usage = {}
        self._streaming_in_code_block = False

        pressure = self.history.get_context_pressure()
        if pressure >= 0.85:
            logger.warning("Context pressure %.0f%%, forcing compaction before LLM call", pressure * 100)
            self.history.force_compact(target_ratio=0.60)
            messages = self.history.get_messages()
        elif pressure >= 0.75:
            logger.info("Context pressure %.0f%%, approaching limit", pressure * 100)

        def _handle_context_overflow():
            logger.warning("Context overflow recovery: forcing aggressive compaction")
            compacted = self.history.force_compact(target_ratio=0.50)
            if compacted:
                messages[:] = self.history.get_messages()
            return compacted

        stream = retry_with_backoff(
            lambda: self.provider.chat_stream(messages, schemas),
            max_retries=3,
            on_context_overflow=_handle_context_overflow,
        )

        thinking_cleared = False

        try:
            for event in stream:
                if not thinking_cleared:
                    display.thinking_done()
                    thinking_cleared = True
                if event["type"] == "text":
                    text = event["content"]
                    if display._use_color():
                        fence_count = text.count("```")
                        if self._streaming_in_code_block:
                            text = display.cyan(text)
                        elif "```" in text:
                            text = display.render_markdown(text)
                        else:
                            text = display.blue(text)
                        if fence_count % 2 == 1:
                            self._streaming_in_code_block = not self._streaming_in_code_block
                    display._write(text)
                    content_parts.append(event["content"])
                elif event["type"] == "tool_start":
                    current_tool = {
                        "id": event["id"],
                        "name": event["name"],
                        "arguments_json": "",
                    }
                    tool_calls.append(current_tool)
                    if event["id"]:
                        tool_calls_by_id[event["id"]] = current_tool
                elif event["type"] == "tool_delta":
                    delta_id = event.get("id", "")
                    target = tool_calls_by_id.get(delta_id, current_tool) if delta_id else current_tool
                    if target:
                        target["arguments_json"] += event["arguments_delta"]
                elif event["type"] == "usage":
                    usage = {
                        "input_tokens": event.get("input_tokens"),
                        "output_tokens": event.get("output_tokens"),
                    }
                elif event["type"] == "done":
                    break
        except KeyboardInterrupt:
            if not thinking_cleared:
                display.thinking_clear()
            raise
        except Exception as e:
            if not thinking_cleared:
                display.thinking_clear()
            logger.warning("Stream interrupted: %s", e)
            if not content_parts and not tool_calls:
                raise

        if content_parts:
            print()

        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = []
            for tc in tool_calls:
                try:
                    arguments = json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                parsed_tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": arguments})

        return {"content": "".join(content_parts) or None, "tool_calls": parsed_tool_calls}, usage

    # ── Tool execution ───────────────────────────────────────────────────

    @staticmethod
    def _shell_display_summary(cmd: str, max_len: int = 90) -> str:
        """Extract a clean one-line summary from a shell command for display.

        Skips comment lines and echo separators, picks the first meaningful
        command, and truncates to max_len.
        """
        lines = cmd.strip().split('\n')
        meaningful = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip pure comments
            if stripped.startswith('#'):
                continue
            # Skip echo separators like echo "===" or echo "---"
            if stripped.startswith('echo') and all(c in '="\'- |' for c in stripped[4:].strip()):
                continue
            meaningful.append(stripped)

        if not meaningful:
            # All comments — use the first comment as context
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#'):
                    summary = stripped[1:].strip()
                    if len(summary) > max_len:
                        summary = summary[:max_len - 3] + "..."
                    return summary
            return cmd[:max_len]

        # Show first meaningful command
        summary = meaningful[0]
        if len(meaningful) > 1:
            suffix = f" (+{len(meaningful) - 1} more)"
            if len(summary) + len(suffix) > max_len:
                summary = summary[:max_len - len(suffix) - 3] + "..." + suffix
            else:
                summary += suffix
        elif len(summary) > max_len:
            summary = summary[:max_len - 3] + "..."
        return summary

    def _execute_tools(self, tool_calls):
        if len(tool_calls) == 1:
            return [self._execute_tool(tool_calls[0])]

        # Gate checks for parallel calls — count all read-only tools toward progress gate
        # and block the entire batch if gates fire
        for tc in tool_calls:
            name = tc["name"]
            # Update progress gate counter for read-only tools
            if name in self._READ_ONLY_TOOLS:
                self._consecutive_reads += 1
                # Track file discovery for staleness detection
                if name == "read_file":
                    path = tc.get("arguments", {}).get("path", "")
                    if path and path not in self._files_read_this_session:
                        self._reads_since_last_new_file = 0
                        self._last_unique_file_count = len(self._files_read_this_session) + 1
                    else:
                        self._reads_since_last_new_file += 1
                else:
                    self._reads_since_last_new_file += 1
            elif name in self._PRODUCTIVE_TOOLS:
                self._consecutive_reads = 0
                self._progress_gate_triggers = 0
                self._reads_since_last_new_file = 0
            # Count toward plan gate (complexity-judge mode)
            if self._complex_task_no_plan and name not in (
                "plan_create", "memory_write", "workspace_experiment", "workspace_current"
            ):
                self._pre_plan_tool_calls += 1

        # Check plan gate after counting
        if not any(tc["name"] in self._PRODUCTIVE_TOOLS for tc in tool_calls):
            plan_block = self._check_plan_creation_gate(tool_calls[0]["name"])
            if plan_block and "TOOL NOT EXECUTED" in plan_block:
                display.warn("Plan gate: HARD BLOCK — parallel tools not executed")
                return [plan_block] * len(tool_calls)

        # Check progress gate (staleness-based) for parallel path
        stale_threshold = 12
        if self._porting_mode:
            stale_threshold = 30
        elif self._consecutive_train_failures >= 2:
            stale_threshold = 18
        if self._reads_since_last_new_file >= stale_threshold + 8:
            has_plan = self.task_plan.get_active() is not None
            if not has_plan:
                block_msg = (
                    f"⛔ [PROGRESS BLOCK — TOOL NOT EXECUTED] You've made "
                    f"{self._reads_since_last_new_file} calls without discovering "
                    "new files or producing output. "
                    "Create a plan (plan_create) to organize your approach."
                )
            else:
                block_msg = (
                    "⛔ [PROGRESS BLOCK — TOOL NOT EXECUTED] Extended re-reading "
                    "without new discoveries. Record findings with memory_write."
                )
                self._reads_since_last_new_file = 0
            display.warn("Progress gate: HARD BLOCK — parallel tools not executed")
            return [block_msg] * len(tool_calls)

        # Pre-confirm all shell commands BEFORE parallel execution.
        shell_tool = self.tool_registry.get("shell")
        denied = set()
        if shell_tool:
            for i, tc in enumerate(tool_calls):
                if tc["name"] == "shell":
                    cmd = tc["arguments"].get("command", "")
                    if shell_tool.needs_confirm(cmd):
                        if not shell_tool.pre_confirm(cmd):
                            denied.add(i)

        results = [None] * len(tool_calls)
        to_run = [
            (i, tc) for i, tc in enumerate(tool_calls) if i not in denied
        ]
        for i in denied:
            results[i] = "DENIED: User declined to execute this command."

        # Show all tool names upfront, then one spinner for the batch
        tool_summaries = []
        for _, tc in to_run:
            name = tc["name"]
            if name == "shell":
                cmd = tc["arguments"].get("command", "")
                tool_summaries.append((name, self._shell_display_summary(cmd)))
            else:
                args = tc.get("arguments", {})
                parts = []
                for k, v in list(args.items())[:2]:
                    s = str(v)
                    if len(s) > 60:
                        s = s[:57] + "..."
                    parts.append(f'{k}="{s}"' if isinstance(v, str) else f'{k}={s}')
                tool_summaries.append((name, ", ".join(parts)))
        # Map original index -> display line index
        idx_to_line = {orig_i: line_i for line_i, (orig_i, _) in enumerate(to_run)}
        display.parallel_tools_start(tool_summaries)

        def _run_quiet(idx, tc):
            tool_name = tc["name"]
            arguments = tc["arguments"]
            t0 = time.time()
            try:
                if tool_name == "shell":
                    result = self.tool_registry.execute(
                        tool_name, _skip_confirm=True,
                        _parallel_index=idx_to_line[idx], **arguments)
                else:
                    result = self.tool_registry.execute(tool_name, **arguments)
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

        with ThreadPoolExecutor(max_workers=min(len(to_run), 4)) as pool:
            futures = {
                pool.submit(_run_quiet, i, tc): i
                for i, tc in to_run
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        display.parallel_tools_finish()

        # Annotate parallel results with failure summary for LLM awareness
        failed_indices = [i for i, r in enumerate(results) if r and isinstance(r, str) and "ERROR" in r[:20]]
        if failed_indices and len(failed_indices) < len(results):
            failed_names = [tool_calls[i]["name"] for i in failed_indices]
            for i in range(len(results)):
                r = results[i]
                if i not in failed_indices and isinstance(r, str):
                    results[i] = (
                        f"[NOTE: {len(failed_indices)} parallel tool(s) failed: "
                        f"{', '.join(failed_names)}. Check if this result depends on them.]\n{r}"
                    )

        return results

    def _execute_tool(self, tool_call, skip_confirm=False):
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # Check for duplicate read (cache hit)
        cached_result = self._check_duplicate_read(tool_name, arguments)
        if cached_result is not None:
            logger.info("Cache hit for %s, skipping execution", tool_name)
            return cached_result + "\n[Cached result from earlier in this turn]"

        # Loop detection check — block execution if looping
        loop_warning = self._check_loop_detection(tool_name, arguments)
        if loop_warning:
            display.warn(f"Loop detected: {tool_name} called with same args, skipping execution")
            return loop_warning

        # Porting path gate — hard block porting writes until user confirms Mode B/C
        porting_path_warning = self._check_porting_path_gate(tool_name, arguments)
        if porting_path_warning:
            display.warn("Porting path gate: must confirm porting path (Mode B/C) with user first")
            return porting_path_warning

        # Reading depth gate — hard block writes when insufficient reading
        reading_depth_warning = self._check_reading_depth_gate(tool_name, arguments)
        if reading_depth_warning:
            display.warn("Reading depth gate: insufficient file reading before implementation")
            return reading_depth_warning

        # Reading quality gate — hard block writes when missing critical categories
        quality_warning = self._check_reading_quality(tool_name, arguments)
        if quality_warning:
            display.warn("Reading quality gate: insufficient category coverage")
            return quality_warning

        # Data pipeline comprehension gate — hard block data writes until pipeline understood
        data_gate_warning = self._check_data_pipeline_gate(tool_name, arguments)
        if data_gate_warning:
            display.warn("Data pipeline gate: must understand source→processing→model input chain first")
            return data_gate_warning

        # Understanding verification gate — hard block training code writes until understanding proven
        understanding_warning = self._check_understanding_verification_gate(tool_name, arguments)
        if understanding_warning:
            display.warn("Understanding verification gate: prove understanding before writing training code")
            return understanding_warning

        # Component isolation plan gate — hard block training script for multi-component models
        component_plan_warning = self._check_component_isolation_gate(tool_name, arguments)
        if component_plan_warning:
            display.warn("Component isolation gate: create component plan before writing training script")
            return component_plan_warning

        # Import verification gate — strong warning before writing training script with unverified imports
        import_warning = self._check_import_verification_gate(tool_name, arguments)

        # Reference comparison strategy gate — strong warning before writing without comparison plan
        reference_warning = self._check_reference_comparison_gate(tool_name, arguments)

        # Checkpoint verification gate — strong warning before training with unverified checkpoint
        ckpt_warning = self._check_checkpoint_verified_gate(tool_name, arguments)

        # Distributed prerequisite gate — strong warning about single-GPU verification
        dist_warning = self._check_distributed_prerequisite_gate(tool_name, arguments)

        # Pre-launch gates — strong warnings before training launch
        failure_mode_warning = self._check_failure_mode_analysis_gate(tool_name, arguments)
        sanity_warning = self._check_sanity_check_gate(tool_name, arguments)
        config_model_warning = self._check_config_model_consistency_gate(tool_name, arguments)
        env_warning = self._check_environment_consistency_gate(tool_name, arguments)
        tp_warning = self._check_tp_compatibility_gate(tool_name, arguments)
        component_int_warning = self._check_component_integration_gate(tool_name, arguments)

        # Progress gate check — hard block prevents execution
        progress_warning, progress_hard_block = self._check_progress_gate(tool_name)
        if progress_hard_block:
            display.warn("Progress gate: HARD BLOCK — tool not executed")
            return progress_warning

        # Plan creation gate — hard block or soft warning
        plan_gate_warning = self._check_plan_creation_gate(tool_name)
        if plan_gate_warning and "TOOL NOT EXECUTED" in plan_gate_warning:
            display.warn("Plan gate: HARD BLOCK — tool not executed")
            return plan_gate_warning

        # Error-escalation gate check
        escalation_warning = self._check_error_escalation(tool_name, arguments)

        # Analysis persistence and verification gates (soft warnings)
        analysis_warning = self._check_analysis_persistence(tool_name, arguments)
        verification_warning = self._check_verification_ladder(tool_name, arguments)
        config_understanding_warning = self._check_config_understanding(tool_name, arguments)

        def _fmt_arg(k, v):
            s = str(v)
            if k in ("content", "new_string", "old_string") and len(s) > 100:
                lines = s.split('\n')
                if len(lines) > 3:
                    s = f"{lines[0][:80]}... ({len(lines)} lines, {len(s)} chars)"
                else:
                    s = s[:100] + f"... ({len(s)} chars)"
            if isinstance(v, str):
                return f'{k}="{s}"'
            return f'{k}={s}'

        t0 = time.time()

        if tool_name == "shell":
            cmd = arguments.get("command", "")
            cmd_display = self._shell_display_summary(cmd)
            display.tool_start(tool_name, cmd_display)
        else:
            args_summary = ", ".join(
                _fmt_arg(k, v) for k, v in list(arguments.items())[:3]
            )
            display.tool_start(tool_name, args_summary)

        try:
            if skip_confirm and tool_name == "shell":
                result = self.tool_registry.execute(
                    tool_name, _skip_confirm=True, **arguments)
            else:
                result = self.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            result = f"ERROR: {e}"
        elapsed = time.time() - t0

        logger.info("Tool %s: %.1fs, result %d chars", tool_name, elapsed, len(result))
        error = "ERROR" in result[:20] if result else False

        # Track files read for reading depth gate
        if tool_name == "read_file" and not error:
            path = arguments.get("path", "")
            if path:
                was_already_read = path in self._files_read_this_session
                self._files_read_this_session.add(path)
                # Track reading categories for quality gate
                for cat, pattern in self._PORTING_READ_CATEGORIES.items():
                    if pattern.search(path):
                        self._reading_categories.add(cat)
                # Cross-compaction re-read enforcement
                if was_already_read:
                    self._rereads_without_save += 1
                    if self._rereads_without_save >= 3:
                        result = (
                            "⛔ [RE-READ BLOCK] You have re-read 3+ files without saving findings. "
                            "You MUST call memory_write NOW to record key findings from your reads "
                            "before reading any more files. This prevents repeated context loss.\n\n"
                            + result
                        )
                    else:
                        result = (
                            "[⚠️ RE-READ DETECTED] You already read this file earlier (lost to compaction). "
                            "After this read, immediately use memory_write to save the key information "
                            "you need from this file. Do NOT continue reading more files without saving.]\n\n"
                            + result
                        )
            threshold = (self._READ_FILE_SUMMARY_THRESHOLD_PORTING
                         if self._porting_mode else self._READ_FILE_SUMMARY_THRESHOLD)
            if len(result) > threshold:
                result = self._summarize_file_content(result, path)

        # Track write operations for auto-continue stagnation detection
        if tool_name in ("write_file", "edit_file") and not error:
            self._last_write_turn = self._turn_count
            path = arguments.get("path", "") or arguments.get("file_path", "")
            if path:
                self._files_written_this_session.add(path)

        # Reset re-read counter when findings are saved
        if tool_name in ("memory_write", "workspace_experiment") and not error:
            self._rereads_without_save = 0

        # Track porting mode activation
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if "model-porter" in skill_name:
                self._porting_mode = True
            if "data-prep" in skill_name:
                self._data_prep_mode = True
            # Strip "SUCCESS: Skill '...' loaded.\n\n" prefix from tool result
            skill_content = result
            prefix_end = result.find("\n\n")
            if prefix_end != -1 and result.startswith("SUCCESS:"):
                skill_content = result[prefix_end + 2:]
            full_content = self._maybe_summarize_skill(skill_name, skill_content)
            # Track skill content for lifecycle management (full content in system prompt)
            self._active_skill_content[skill_name] = full_content
            self._skill_load_iterations[skill_name] = self._total_iterations
            # Replace result with short placeholder — full content is in system prompt
            result = f"[Skill '{skill_name}' loaded — content available in system context]"

        # Track porting path confirmation (plan_create or workspace_experiment with path decision)
        if self._porting_mode and not self._porting_path_confirmed and not error:
            args_lower = str(arguments).lower()
            if tool_name == "plan_create" and any(
                kw in args_lower for kw in ("mode b", "mode c", "native", "huggingface wrapper", "hf wrapper", "fsdp2", "megatron", "tensor parallel")
            ):
                self._porting_path_confirmed = True
                logger.info("Porting path confirmed via plan_create")
            elif tool_name == "workspace_experiment" and any(
                kw in args_lower for kw in ("porting_path", "mode b", "mode c", "native", "megatron")
            ):
                self._porting_path_confirmed = True
                logger.info("Porting path confirmed via workspace_experiment")
            elif tool_name == "memory_write" and any(
                kw in args_lower for kw in ("porting path", "mode b", "mode c", "porting decision")
            ):
                self._porting_path_confirmed = True
                logger.info("Porting path confirmed via memory_write")

        # Track analysis persistence
        if tool_name in ("workspace_experiment", "memory_write") and not error:
            content = arguments.get("content", "")
            if len(content) > 200 or any(
                kw in content.lower()
                for kw in ("mapping", "component", "analysis", "architecture", "diff")
            ):
                self._analysis_persisted = True
                if self._verification_stage == "none":
                    self._verification_stage = "analysis"

        # Detect training started for skill lifecycle
        if not self._training_started and not error:
            if tool_name == "parse_training_metrics" and "step" in result.lower():
                self._training_started = True
            elif tool_name == "find_latest_log" and result and "not found" not in result.lower():
                self._training_started = True

        # Track skill references for lifecycle
        if tool_name == "load_skill":
            skill_name = arguments.get("name", "")
            self._recently_referenced_skills.add(skill_name)

        # Track data pipeline understanding confirmation
        if self._data_prep_mode and not self._data_pipeline_understood and not error:
            if tool_name == "memory_write":
                content = arguments.get("content", "").lower()
                data_kws = ("source format", "data format", "pipeline", "tokeniz",
                            "preprocess", "get_batch", "dataloader", "task_encoder",
                            "webdataset", "energon", "data flow")
                if sum(1 for kw in data_kws if kw in content) >= 2:
                    self._data_pipeline_understood = True
                    logger.info("Data pipeline understanding confirmed via memory_write")
            elif tool_name == "plan_create":
                content = str(arguments).lower()
                if any(kw in content for kw in ("data pipeline", "data format", "preprocessing", "data flow")):
                    self._data_pipeline_understood = True
                    logger.info("Data pipeline understanding confirmed via plan_create")

        # Track new gate unlocking conditions
        if self._porting_mode and not error:
            content = str(arguments.get("content", "")).lower() if tool_name in ("workspace_experiment", "memory_write") else ""
            args_lower = str(arguments).lower()

            # A1: Understanding verification — unlock when understanding documented
            if not self._understanding_verified and tool_name == "workspace_experiment":
                understanding_kws = ("data flow", "model signature", "forward()", "get_batch",
                                     "parallelism", "input format", "output format", "loss function")
                if sum(1 for kw in understanding_kws if kw in content) >= 3:
                    self._understanding_verified = True
                    logger.info("Understanding verification gate unlocked")

            # A3: Component isolation plan — unlock when plan created
            if not self._component_plan_created and tool_name == "plan_create":
                component_kws = ("component", "phase", "isolat", "verify forward", "verify backward",
                                 "integration", "data flow order")
                if sum(1 for kw in component_kws if kw in args_lower) >= 2:
                    self._component_plan_created = True
                    logger.info("Component isolation plan gate unlocked")

            # B1: Failure mode analysis — unlock when analysis documented
            if not self._failure_mode_analyzed and tool_name == "workspace_experiment":
                failure_kws = ("failure mode", "what could go wrong", "how to detect",
                               "what if", "most likely", "risk")
                if sum(1 for kw in failure_kws if kw in content) >= 2:
                    self._failure_mode_analyzed = True
                    logger.info("Failure mode analysis gate unlocked")

            # B2: Sanity checks — unlock when checks documented
            if not self._sanity_checks_passed and tool_name == "workspace_experiment":
                sanity_kws = ("sanity check", "data check", "model init", "config check",
                              "checkpoint check", "all checks passed", "verified")
                if sum(1 for kw in sanity_kws if kw in content) >= 2:
                    self._sanity_checks_passed = True
                    logger.info("Sanity check gate unlocked")

            # B4: Config-model consistency — unlock when verified
            if not self._config_model_verified and tool_name in ("workspace_experiment", "shell"):
                if tool_name == "workspace_experiment":
                    config_kws = ("config", "model __init__", "parameter", "mismatch", "consistent", "match")
                    if sum(1 for kw in config_kws if kw in content) >= 2:
                        self._config_model_verified = True
                        logger.info("Config-model consistency gate unlocked")
                elif tool_name == "shell" and "mismatch" not in (result or "").lower():
                    cmd = arguments.get("command", "")
                    if re.search(r'python.*config.*model|python.*verify.*config', cmd, re.I):
                        self._config_model_verified = True
                        logger.info("Config-model consistency gate unlocked via verification script")

            # C2: Environment consistency — unlock when verified
            if not self._env_verified and tool_name == "shell":
                cmd = arguments.get("command", "")
                if re.search(r'python.*-c.*import.*__file__|pip show|conda list', cmd) and not error:
                    self._env_verified = True
                    logger.info("Environment consistency gate unlocked")

            # C4: Component integration — unlock when documented
            if not self._component_integration_verified and tool_name == "workspace_experiment":
                int_kws = ("component.*verified", "per-component", "individual.*test",
                           "all components", "integration verified")
                if sum(1 for kw in int_kws if re.search(kw, content)) >= 1:
                    self._component_integration_verified = True
                    logger.info("Component integration gate unlocked")

            # A4: Import verification — unlock when imports tested
            if not self._imports_verified and tool_name == "shell":
                cmd = arguments.get("command", "")
                if re.search(r'python.*-c.*"import|python.*-c.*\'import', cmd) and not error:
                    self._imports_verified = True
                    logger.info("Import verification gate unlocked")

            # A2: Reference comparison — unlock when plan created
            if not self._reference_comparison_planned:
                if tool_name in ("plan_create", "workspace_experiment"):
                    ref_kws = ("comparison", "reference", "compare", "baseline", "align")
                    if sum(1 for kw in ref_kws if kw in args_lower) >= 2:
                        self._reference_comparison_planned = True
                        logger.info("Reference comparison strategy gate unlocked")

        # Verification ladder stage advancement
        if self._porting_mode and tool_name == "shell" and not error:
            cmd = arguments.get("command", "")
            # Use output analysis in addition to command regex for robustness
            output_lower = result.lower() if result else ""

            if self._verification_stage == "analysis":
                # Check if model was successfully instantiated (output analysis)
                if any(kw in output_lower for kw in ("model initialized", "model created", "parameters:", "total params")):
                    self._verification_stage = "init_ok"
                    self._record_verification_advance("init_ok", cmd[:100])
                # Fallback to command regex
                elif re.search(r'python.*import.*Model|python.*model.*init|python.*instantiate', cmd, re.I):
                    self._verification_stage = "init_ok"
                    self._record_verification_advance("init_ok", cmd[:100])

            elif self._verification_stage == "init_ok":
                # Check if forward pass succeeded (output analysis)
                if any(kw in output_lower for kw in ("loss:", "logits:", "forward pass", "output shape", "verification passed")):
                    self._verification_stage = "forward_aligned"
                    self._record_verification_advance("forward_aligned", cmd[:100])
                # Fallback to command regex
                elif re.search(r'python.*(forward|compare|reference|align)', cmd, re.I):
                    self._verification_stage = "forward_aligned"
                    self._record_verification_advance("forward_aligned", cmd[:100])

            elif self._verification_stage == "forward_aligned":
                # Check if backward pass succeeded (output analysis)
                if self._TRAIN_LAUNCH_RE.search(cmd) and self._is_quick_test_command(cmd):
                    if any(kw in output_lower for kw in ("iteration", "step", "grad", "backward")):
                        self._verification_stage = "backward_ok"
                        self._record_verification_advance("backward_ok", cmd[:100])

            elif self._verification_stage == "backward_ok":
                # Check if distributed training succeeded (output analysis + command)
                if self._TRAIN_LAUNCH_RE.search(cmd) and self._is_quick_test_command(cmd):
                    has_parallelism = re.search(r'tensor.model.parallel.size.*[2-9]|nproc.per.node.*[2-9]|--tp\s+[2-9]|--pp\s+[2-9]', cmd)
                    has_distributed_output = any(kw in output_lower for kw in ("rank", "world_size", "distributed", "nccl"))
                    if has_parallelism or has_distributed_output:
                        self._verification_stage = "distributed_ok"
                        self._record_verification_advance("distributed_ok", cmd[:100])

        # Manual verification stage override via workspace_experiment
        if self._porting_mode and tool_name == "workspace_experiment" and not error:
            content = arguments.get("content", "")
            for stage in self._VERIFICATION_STAGES:
                if f"verification_stage: {stage}" in content.lower():
                    current_idx = self._VERIFICATION_STAGES.index(self._verification_stage) if self._verification_stage in self._VERIFICATION_STAGES else 0
                    new_idx = self._VERIFICATION_STAGES.index(stage)
                    if new_idx > current_idx:
                        self._verification_stage = stage

        # Track experiment registration — on successful workspace_experiment create
        if (tool_name == "workspace_experiment" and not error):
            action = arguments.get("action", "")
            if action == "create":
                self._experiment_registered = True

        detail = ""
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            # Track dry-run success
            if (self._TRAIN_LAUNCH_RE.search(cmd)
                    and self._is_quick_test_command(cmd)
                    and not error):
                self._dry_run_passed = True

            # Experiment registry gate: warn if training launched without entry
            if (self._TRAIN_LAUNCH_RE.search(cmd)
                    and not self._is_quick_test_command(cmd)):
                if not self._experiment_registered:
                    result = self._EXPERIMENT_GATE_WARNING + result
                    display.warn("Training launched without experiment entry!")
                # Dry-run gate: warn if checkpoint-loading run without prior dry-run
                if (self._CHECKPOINT_LOAD_RE.search(cmd)
                        and not self._dry_run_passed):
                    result = self._DRY_RUN_WARNING + result
                    display.warn("Checkpoint-loading training launched without prior dry-run!")
                # Reset after each launch — next launch needs its own entry/dry-run
                self._experiment_registered = False
                self._dry_run_passed = False
            # Remind to update experiment record when training fails
            if (self._TRAIN_LAUNCH_RE.search(cmd)
                    and not self._is_quick_test_command(cmd)
                    and error):
                result = result + self._EXPERIMENT_UPDATE_REMINDER
                # Checkpoint: auto-record training failure
                ckpt_warn = self._checkpoint_training_failure(cmd, result)
                if ckpt_warn:
                    result = result + ckpt_warn
            # Remind to memorize learnings when training launches successfully
            if (self._TRAIN_LAUNCH_RE.search(cmd)
                    and not self._is_quick_test_command(cmd)
                    and not error):
                result = result + self._TRAINING_MEMORY_HINT
                # Checkpoint: auto-record training launch
                ckpt_warn = self._checkpoint_training_launch(cmd, result)
                if ckpt_warn:
                    result = result + ckpt_warn
            annotations = self._result_judge(cmd, result, elapsed)
            annotations = self._dedup_annotations(annotations)
            if annotations:
                header = "\n".join(f"[{a}]" for a in annotations)
                result = header + "\n" + result
        if error and result:
            raw = result.split('\n')[0].replace("ERROR:", "").strip()
            detail = (raw[:57] + "...") if len(raw) > 60 else raw

        # Workaround detection: same tool, previous call failed, this one succeeded
        if (self._last_tool_call is not None
                and not error
                and self._last_tool_call[0] == tool_name
                and self._last_tool_call[2]):
            result = result + self._WORKAROUND_MEMORY_HINT
            # Checkpoint: auto-record workaround
            prev_cmd = self._last_tool_call[1]
            curr_cmd = arguments.get("command", "") if tool_name == "shell" else str(arguments)[:200]
            ckpt_warn = self._checkpoint_workaround(tool_name, prev_cmd, curr_cmd)
            if ckpt_warn:
                result = result + ckpt_warn

        # Checkpoint: new unique error
        if error and tool_name == "shell":
            error_sig = self._extract_error_summary(result)
            ckpt_warn = self._checkpoint_new_error(error_sig, result)
            if ckpt_warn:
                result = result + ckpt_warn

        # Auto-memory hint after writing model/training code
        if tool_name in ("write_file", "edit_file") and not error and self._porting_mode:
            path = arguments.get("path", "") or arguments.get("file_path", "")
            if re.search(r'train_|pretrain_|model|get_batch|forward_step|dataset', path):
                result = result + (
                    "\n\n[AUTO-MEMORY] You just wrote/edited model code. "
                    "Record what you implemented and why with memory_write "
                    "(key: what_was_implemented, type: finding). "
                    "This ensures cross-session recovery if context is compacted."
                )

        # Auto-update workspace current after verification stage advance
        if not error and self._porting_mode:
            self._auto_update_workspace_on_progress(tool_name, arguments)

        # Auto-persist key events (training, kills, code changes, plan steps)
        self._auto_persist_on_event(tool_name, arguments, result, error)

        # Track error state for error-escalation gate
        self._last_tool_had_error = error
        if error:
            self._root_cause_recorded_since_error = False
        # Track if root cause was recorded (memory_write or workspace_experiment)
        if tool_name in ("memory_write", "workspace_experiment") and not error:
            self._root_cause_recorded_since_error = True

        # Apply dry-run gate for shell commands
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            result = self._check_dry_run_gate(cmd, result)
            # Kill-retry loop detection
            kill_warn = self._check_kill_retry_loop(cmd)
            if kill_warn:
                result = result + kill_warn
            # Training hang detection
            hang_warn = self._check_training_hang(cmd, result, elapsed)
            if hang_warn:
                result = result + hang_warn

        # Phase transition gate — check AFTER execution, only for successful calls
        phase_warning = ""
        if not error:
            phase_warning = self._check_phase_transition(tool_name)

        # Auto-load skill for detected error patterns
        if error and result:
            skill_warning = self._auto_load_skill_for_error(result)
            if skill_warning:
                result = skill_warning + "\n" + result
                display.warn(f"Auto-loaded skill based on error pattern")

        # Inject enforcement warnings (with user notifications) — deduplicated
        all_warnings = [
            (import_warning, "Import verification gate: verify critical imports before writing training script"),
            (reference_warning, "Reference comparison gate: create comparison strategy before implementation"),
            (ckpt_warning, "Checkpoint verification gate: verify converted checkpoint before training"),
            (dist_warning, "Distributed prerequisite gate: consider single-GPU verification first"),
            (failure_mode_warning, "Failure mode analysis gate: analyze failure modes before first launch"),
            (sanity_warning, "Sanity check gate: run 4 critical checks before first real training"),
            (config_model_warning, "Config-model consistency gate: verify config matches model parameters"),
            (env_warning, "Environment consistency gate: verify package paths"),
            (tp_warning, "TP compatibility gate: verify custom layers support tensor parallelism"),
            (component_int_warning, "Component integration gate: verify per-component before full training"),
            (progress_warning, "Progress gate: too many reads without recording findings"),
            (plan_gate_warning, "Plan gate: consider creating a plan to organize your approach"),
            (escalation_warning, "Error escalation: repeated failures require root cause analysis"),
            (analysis_warning, "Analysis persistence: findings must be saved before implementation"),
            (verification_warning, "Verification ladder: must complete verification stages in order"),
            (config_understanding_warning, None),
            (phase_warning, "Phase transition: prerequisites not met for next phase"),
        ]
        for warning_text, display_msg in all_warnings:
            if warning_text and warning_text != self._last_gate_warning:
                result = warning_text + "\n" + result
                if display_msg:
                    display.warn(display_msg)
                self._last_gate_warning = warning_text
            # Skip duplicate warnings silently

        # Post-launch informational gates (D1, D2, D3, D4) — append to training output
        if tool_name == "shell" and not error:
            cmd = arguments.get("command", "")
            if self._is_training_launch(cmd) and result:
                grad_info = self._check_gradient_health(cmd, result)
                if grad_info:
                    result = result + "\n" + grad_info
                loss_info = self._check_loss_sanity(cmd, result)
                if loss_info:
                    result = result + "\n" + loss_info
                component_grad_info = self._check_component_gradient_flow(cmd, result)
                if component_grad_info:
                    result = result + "\n" + component_grad_info
                ckpt_info = self._check_checkpoint_numerical_verification(cmd, result)
                if ckpt_info:
                    result = result + "\n" + ckpt_info
            # GPU zombie detection — applies to any shell command showing GPU issues
            zombie_info = self._check_gpu_zombie_escalation(cmd, result)
            if zombie_info:
                result = result + "\n" + zombie_info

        # Track for next call's workaround detection
        cmd_key = arguments.get("command", "") if tool_name == "shell" else tool_name
        self._last_tool_call = (tool_name, cmd_key, error)

        # Cache tool result for duplicate detection within this turn
        if tool_name == "read_file" and not error:
            path = arguments.get("path", "")
            cache_key = ("read_file", path)
            self._tool_call_cache[cache_key] = result

        # Mid-turn autosave: save state every N tool calls
        self._tool_calls_since_save = getattr(self, '_tool_calls_since_save', 0) + 1
        if self._tool_calls_since_save >= self._AUTOSAVE_INTERVAL:
            self._mid_turn_autosave()
            self._tool_calls_since_save = 0

        display.tool_done(tool_name, elapsed, detail=detail, error=error)
        return result

    @staticmethod
    def _is_quick_test_command(cmd):
        """Check if a training-like command is actually a quick test (dry-run, help, etc.)."""
        cmd_lower = cmd.lower()
        # Exact patterns (no word-boundary issues)
        exact_patterns = ['--dryrun', '--dry-run', '--dry_run', '--help', '-h', '--version',
                          'python -c', 'import ']
        if any(p in cmd_lower for p in exact_patterns):
            return True
        # Numeric patterns — must match the exact value (e.g., "1" not "100")
        numeric_patterns = [
            (r'--total[_-]steps[\s=]+[012]\b', None),
            (r'--max[_-]steps[\s=]+[012]\b', None),
            (r'--num[_-]steps[\s=]+[012]\b', None),
            (r'--train[_-]iters[\s=]+[012]\b', None),
        ]
        for pattern, _ in numeric_patterns:
            if re.search(pattern, cmd_lower):
                return True
        return False

    def _is_training_launch(self, cmd):
        """Check if a command is a training launch command."""
        return bool(self._TRAIN_LAUNCH_RE.search(cmd))

    def _append_tool_results(self, tool_results):
        """Append tool results, merging into one message for Anthropic compatibility."""
        if not tool_results:
            return
        if len(tool_results) == 1:
            self.history.append(tool_results[0])
            return
        first = tool_results[0]
        if first.get("role") == "user" and isinstance(first.get("content"), list):
            merged_content = []
            for tr in tool_results:
                merged_content.extend(tr["content"])
            self.history.append({"role": "user", "content": merged_content})
        else:
            for tr in tool_results:
                self.history.append(tr)

    def _replace_last_tool_result(self, new_result):
        """Replace the last tool result message in history with a new one."""
        for i in range(len(self.history.messages) - 1, -1, -1):
            msg = self.history.messages[i]
            if _is_tool_result_msg(msg):
                self.history.messages[i] = new_result
                return
        self.history.append(new_result)

    # ── Auto skill loading (P3-12) ───────────────────────────────────────

    def _auto_load_skills(self, user_input):
        if not self.config.auto_skill or len(user_input.strip()) < 10:
            return
        candidates = self._skill_judge(user_input)

        for skill_name in candidates:
            try:
                content = self.skill_manager.load(skill_name)
                content = self._maybe_summarize_skill(skill_name, content)
                tool_call_id = f"auto_{uuid.uuid4().hex[:8]}"
                fake_response = {
                    "content": None,
                    "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
                }
                self.history.append(self.provider.format_assistant_message(fake_response))
                # Short placeholder in history — full content lives in system prompt via _active_skill_content
                self.history.append(self.provider.format_tool_result(
                    tool_call_id, f"[Skill '{skill_name}' loaded — content available in system context]"))
                self._loaded_skills.add(skill_name)
                # Track for skill lifecycle management
                self._active_skill_content[skill_name] = content
                self._skill_load_iterations[skill_name] = self._total_iterations
                display.skill_auto_loaded(skill_name)
            except Exception:
                pass

    # ── Commands ─────────────────────────────────────────────────────────

    def _handle_skill_command(self, user_input):
        skill_name = user_input[len("/skill"):].strip()
        if not skill_name:
            print("Usage: /skill <name>")
            return

        try:
            content = self.skill_manager.load(skill_name)
            content = self._maybe_summarize_skill(skill_name, content)
        except FileNotFoundError as e:
            print(f"Skill not found: {e}")
            return

        tool_call_id = f"skill_{uuid.uuid4().hex[:8]}"
        fake_response = {
            "content": None,
            "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
        }
        self.history.append(self.provider.format_assistant_message(fake_response))
        # Short placeholder in history — full content lives in system prompt
        self.history.append(self.provider.format_tool_result(
            tool_call_id, f"[Skill '{skill_name}' loaded — content available in system context]"))
        self._loaded_skills.add(skill_name)
        # Track for skill lifecycle management
        self._active_skill_content[skill_name] = content
        self._skill_load_iterations[skill_name] = self._total_iterations
        self._refresh_system_prompt()

        self.history.append({
            "role": "user",
            "content": f"I've loaded the '{skill_name}' skill. Please acknowledge and tell me how you can help with it.",
        })
        self._react_loop()

    def _handle_file_command(self, user_input):
        path = user_input[len("/file"):].strip()
        if not path:
            print("Usage: /file <path>")
            return
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        display.file_injected(path, len(content))
        self.history.append({
            "role": "user",
            "content": f"[File: {path}]\n```\n{content}\n```",
        })

    def _handle_save_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        sid = parts[1].strip() if len(parts) > 1 else None
        msgs = [m for m in self.history.full_log if m.get("role") != "system"]
        metadata = {
            "provider": self.config.provider,
            "model": self.config.model,
            "turns": self._turn_count,
        }
        path = save_session(msgs, self.config.session_dir, sid, metadata)
        display.session_saved(path)

    def _handle_load_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else None

        if not target:
            sessions = list_sessions(self.config.session_dir)
            display.session_list(sessions)
            return

        path = target
        if not os.path.isfile(path):
            d = self.config.session_dir
            if not d:
                d = os.path.join(Path.home(), ".flagscale", "sessions")
            candidate = os.path.join(d, f"{target}.json")
            if os.path.isfile(candidate):
                path = candidate
            else:
                print(f"Session not found: {target}")
                return

        try:
            data = load_session(path)
        except Exception as e:
            print(f"Error loading session: {e}")
            return

        msgs = data.get("messages", [])
        self.history._messages = [self.history.messages[0]] if self.history.messages and self.history.messages[0].get("role") == "system" else []
        self.history._messages.extend(msgs)
        user_turns = len([m for m in msgs if m.get("role") == "user"])
        display.session_loaded(path, user_turns)

    def _handle_export_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            path = os.path.expanduser(parts[1].strip())
        else:
            d = self.config.session_dir or os.path.join(Path.home(), ".flagscale", "sessions")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"session_{self._session_id}.md")

        lines = [f"# FlagScale Agent Session Export\n"]
        lines.append(f"Provider: {self.config.provider} | Model: {self.config.model}")
        lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

        messages = self.history.full_log
        turn_num = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")

            # Detect new turn: user message that is not a tool result
            if role == "user" and not _is_tool_result_msg(msg):
                turn_num += 1
                lines.append(f"\n---\n\n## Turn {turn_num}\n")

            if isinstance(content, list):
                parts_text = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts_text.append(block.get("text", ""))
                        elif btype == "tool_use":
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            inp_str = json.dumps(inp, ensure_ascii=False, indent=2)
                            parts_text.append(f"[Tool: {name}]\n```json\n{inp_str}\n```")
                        elif btype == "tool_result":
                            inner = block.get("content", "")
                            parts_text.append(f"[Result]\n```\n{inner}\n```")
                    elif isinstance(block, str):
                        parts_text.append(block)
                content = "\n\n".join(parts_text)

            if role == "user":
                if _is_tool_result_msg(msg):
                    lines.append(f"\n**Tool Result:**\n\n{content}\n")
                else:
                    lines.append(f"\n### User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"\n### Assistant\n\n{content}\n")
            elif role == "tool":
                lines.append(f"\n**Tool Result:**\n\n```\n{content}\n```\n")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(display.green(f"✓ Exported to {path} ({len(messages)} messages)"))
        except Exception as e:
            print(f"Error exporting: {e}")


class _PluginShellTool:
    """A tool loaded from a JSON spec that wraps a shell command template."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.parameters = spec.get("parameters", {"type": "object", "properties": {}})
        self.max_result_size = spec.get("max_result_size", 50000)
        self._command_template = spec.get("command", "")
        self._timeout = spec.get("timeout", 120)

    def execute(self, **kwargs):
        import subprocess
        cmd = self._command_template
        for k, v in kwargs.items():
            cmd = cmd.replace(f"{{{k}}}", shlex.quote(str(v)))
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self._timeout)
            output = (result.stdout or "") + (result.stderr or "")
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {self._timeout}s."
        except Exception as e:
            return f"ERROR: {e}"

    def to_openai_schema(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def to_anthropic_schema(self):
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}