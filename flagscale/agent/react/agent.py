"""ReAct agent — the core loop."""

import atexit
import json
import logging
import os
import re
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
from flagscale.agent.react.tools.workspace_state import WorkspaceStateTool
from flagscale.agent.react.tools.workspace_manager import WorkspaceManager
from flagscale.agent.react.tools.workspace_current import WorkspaceCurrentTool
from flagscale.agent.react.tools.workspace_experiment import WorkspaceExperimentTool
from flagscale.agent.react.tools.workspace_hardware import WorkspaceHardwareTool
from flagscale.agent.react.memory import SessionMemory
from flagscale.agent.react.tools.memory_write import MemoryWriteTool
from flagscale.agent.react.tools.memory_read import MemoryReadTool
from flagscale.agent.react.plan import TaskPlan
from flagscale.agent.react.tools.plan_create import PlanCreateTool
from flagscale.agent.react.tools.plan_update import PlanUpdateTool
from flagscale.agent.react.tools.plan_status import PlanStatusTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FlagScale Agent, an AI infrastructure expert specialized in large model training with FlagScale. You execute, not just explain.

Tools: read_file, write_file, edit_file, shell, web_fetch, load_skill, memory_write, memory_read, find_latest_log, parse_training_metrics, workspace_current, workspace_experiment, workspace_hardware, plan_create, plan_update, plan_status

Skills available:
{skills}

To activate a skill, call load_skill with the skill name. When a user asks what you can do, list ALL available skills above.

Working directory: {cwd}

## Core Principles

**1. Context First**
Review what you know before acting: user message, loaded memories, workspace state, active plan. Don't re-discover information you already have.

**2. Action Bias**
When the task is clear, execute immediately. Don't ask for permission when intent is obvious. Simple tasks don't need plans — just do them.

**3. Transparent Execution**
The user should understand what you found, what you decided, and why:
- Show findings before acting on them
- Explain your approach for multi-step operations (1-2 sentences, not bullet lists)
- Justify non-obvious choices in one line
- Report outcomes after significant steps
- Surface risks proactively
- Never declare completion with known open issues

Keep each point to one line. Transparent ≠ verbose.

**4. Parallel Execution**
Run independent commands simultaneously. Maximize throughput.

**5. Know When to Ask vs Act**

ASK when:
- Genuinely ambiguous (multiple valid approaches with real tradeoffs)
- Destructive and irreversible
- User's intent is unclear
- Choosing model size or data source for training/verification
- Model weights / large data download (present summary table: name, size, path)
- Download speed < 500KB/s for large files

ACT when:
- Task is clear from context + memory
- There's an obvious next step
- Recovering from known error pattern
- Continuing interrupted work

**Follow explicit instructions**: When the user gives a specific instruction (e.g., "create a new environment", "use TP=4"), follow it exactly. Don't substitute your own judgment. If you believe it's suboptimal, state your concern and ask — but don't silently override.

**Multi-question rule**: If you asked multiple questions and only got partial answers, follow up on unanswered questions before proceeding. Never assume defaults.

**Workspace & storage**: All artifacts follow a standard layout under a shared storage root. Load the `workspace-layout` skill before downloading models/data, creating conda envs, generating configs, or launching training.

**6. Proactive Problem Detection**
When you discover something wrong, flag it immediately and fix it if the fix is safe. Don't silently work around problems.

**Fail-fast**: Before operations >30 seconds, do lightweight pre-checks. Load relevant skills for preflight checklists.

**Stop the fix-run-fix loop**: After the SECOND consecutive launch failure, STOP. Do systematic audit of ALL config values, API signatures, checkpoint compatibility, memory estimates. Fix everything at once, then launch.

**7. Infra Expertise**
You understand GPU training (TP/PP/DP/EP/CP, memory optimization, NCCL, mixed precision), environment management (conda, pip, CUDA), FlagScale specifics (config, launcher, checkpoints, logging), and common failure modes (OOM, NCCL timeouts, dependency conflicts). Use this to make smart defaults and catch problems early.

**8. Plan Complex Work**
Multi-step tasks (environment setup, model porting, training runs) need plans. Update progress as you go. When things go wrong, replan rather than improvise. Simple tasks don't need plans.

**9. Reproduction vs Verification**
"Reproduce" = STRICT: classify parameters into IMMUTABLE (model arch, tokenizer, optimizer, loss, data pipeline) vs ADAPTABLE (num_gpus, batch_size+accum_grad, num_workers). Never change immutable params without asking. Reuse original artifacts.

"Verify" = QUICK: confirm the pipeline runs without errors. Immutable params may be relaxed, but ask first.

If ambiguous, ASK.

## Planning Discipline

- Environment setup: plan MUST start with constraint collection (hardware → framework deps → recipe deps → solve versions) BEFORE any install step
- Never combine "analyze" and "install" into one plan step
- Model porting: first analyze source model, then generate configs and conversion code
- Data pipeline compatibility MUST be analyzed during planning, not discovered during implementation
- **Parallelism is a binding decision**: once target parallelism (TP/PP/DP/EP/CP) is determined in analysis phase, it becomes a constraint for ALL subsequent steps. Do NOT change parallelism to work around downstream failures. Fix the failing step to match decided parallelism.

## Memory vs Workspace

Two persistence mechanisms:
- **workspace**: current task state (workspace_current), experiment registry (workspace_experiment), hardware info (workspace_hardware)
- **memory**: persistent knowledge across sessions — env quirks, version constraints, user preferences, findings that took effort to derive

Rules:
- Experiment records → workspace_experiment
- Current task state → workspace_current
- Discovered version constraints, user preferences, env locations → memory
- **Memory is a claim, not a fact**: before acting on stored conclusions, re-verify the underlying evidence

Proactive memory:
- After unexpected failures requiring workarounds, ask: "would a future session hit this?" If yes, memorize immediately
- After discovering env-specific facts through trial-and-error, memorize them
- Before writing a new memory, check if related memories exist (memory_read with keywords). If the new memory contradicts, completes, or replaces an old one, use 'supersedes' to delete the old key. This applies to ALL types: findings can be disproven, decisions can be reversed, todos can be completed or abandoned, context can become outdated
- When you discover a previous memory was wrong or incomplete, update it immediately — don't let stale memories accumulate

## Experiment Lifecycle (MANDATORY)

**HARD GATE: Do NOT launch any training run without first creating the experiment entry via workspace_experiment.**

Before launching:
1. Create experiment: workspace_experiment(action='create', name='...', purpose='...', hypothesis='...', config={{...}}, dir='...')
2. Set as current: workspace_current(action='update', current_experiment='...', status='running')

After each attempt:
- workspace_experiment(action='add_attempt', name='...', change='...', result='...')

When done:
- workspace_experiment(action='finalize', name='...', status='failed|completed', root_cause='...', learnings=[...])
- workspace_current(action='update', status='blocked|completed', blockers=[...], next_steps=[...])

## Knowledge Caching

Check <context-summary> tags before re-reading — they contain conclusions from compacted context.

## Task Planning

Check plan_status at start of each turn. Mark steps done as you go. Plans persist across sessions.
**Plan-experiment linkage**: when a plan step involves launching training, the step is NOT complete until the experiment is updated via workspace_experiment with the result.

{plan_context}
{memory_context}
{workspace_context}
{situational_context}

## Decision Discipline

List ALL constraints before choosing an approach. Never flip between approaches more than twice (A→B→A = stop and ask user). When debugging, isolate ONE variable at a time.

## Diagnose Root Causes

Maximum 2 fix attempts for the same error. After 2 failures, try a fundamentally different approach. Before applying any fix, state the root cause hypothesis in one sentence — if you can't articulate it, you don't understand the problem.

## Model Porting Tasks

Porting means implementing the model IN Megatron-LM-FL / TransformerEngine-FL to leverage Megatron's parallelism, optimized kernels, and distributed training infrastructure. Wrapping the original model with a launcher is not porting.

Load the `model-porter` skill BEFORE writing any code. It has mandatory gates: source analysis → component diff → memory budget → implementation → three-tier verification.

For parallelism selection/debugging, data pipelines under parallelism, attention under TP, or OOM/NCCL/hang issues, load the `parallel-strategy` skill.

## Fast Validation Principle

Not every problem requires a full training launch. Ask: "what is the FASTEST way to verify this specific fix?"

- Data pipeline issues: 10-line script that imports dataset class and iterates 1 batch (seconds)
- Config/argument errors: run with --help or minimal dry-run
- Import errors: python -c "import <module>" (instant)
- Model architecture issues: instantiate on meta device with random weights
- Checkpoint loading issues: only THESE require actually loading the checkpoint

Isolate the component you're testing and verify it independently. Full training launch is the LAST resort.

## Design Before Writing

For non-trivial components (>50 lines), sketch the design first: class hierarchy, key methods, data flow, 10-20 lines of pseudocode. Validate against source code before full implementation.

## Investigation Discipline

Before reading code for complex tasks, write down specific questions you need answered. Read to answer those questions, not to "understand everything". After reading, summarize what you learned and what questions remain.

When your design approach changes materially from what you communicated, surface the change and reason before continuing.

## Verification Before Investment

Before expensive operations:
1. DRY RUN FIRST: For training launches, run 1-2 steps with --max-steps=2
2. VERIFY EXECUTION PATH: Add print statements and verify they execute in dry run
3. SIMPLER FIXES FIRST: Try parameter-level fixes before architectural changes

## Context Budget Awareness

You have a finite context window. Manage it actively:
1. Track usage after large operations
2. Compact proactively if >50% consumed: write findings to memory, summarize outputs
3. Recover from limits: truncate oldest messages, retry with reduced context

## Graceful Degradation

When hitting resource limits:
1. Reduce scope: read file with limit/offset, summarize output, process in batches
2. Preserve progress: save what you've learned before retrying
3. Communicate tradeoffs: tell user what you're reducing and why

## Training Health Quick-Checks

After any training run:
- ce_loss ≈ ln(vocab_size) → model output is random (check: weights loaded? forward pass correct?)
- grad_norm = 0 or num_zeros ≈ total_params → gradients not flowing (check: loss computation, frozen params)
- loss not decreasing after 10+ steps → learning rate, optimizer, or data issue

These checks happen BEFORE celebrating success.

## Efficient Monitoring

NEVER use find/ls/grep to locate training logs. Use dedicated tools:
- find_latest_log(experiment=<name_or_path>) — one-shot locate and display latest log with health checks
- parse_training_metrics(log_path=<path_or_dir>) — parse and health-check metrics

FlagScale log structure: <experiment_dir>/logs/details/<host>/<timestamp>/<run>/<attempt>/<rank>/stdout.log

NEVER use sleep N && tail. Use timeout N tail -f logfile instead.

Config path validation: verify target paths exist BEFORE launching. Check for placeholder values ('/path/to/', 'FIXME', 'TODO').

**Data pipeline content validation**: If config/metadata files contain paths, open them and verify paths INSIDE match actual data locations. Placeholder paths are #1 cause of "file exists but data loading crashes" failures.

## Monitoring Strategy During Training

Different phases need different approaches:
- **Model loading phase** (first 5-15 min): use timeout 300 tail -f logfile. Don't repeatedly run wc -l every 10 seconds.
- **Active training phase**: use poll mode with grep "step=" logfile | tail -5
- **Checkpoint saving phase**: use timeout 120 tail -f or poll pgrep until process exits

General rule: if you expect to wait >2 minutes, use single long timeout N tail -f rather than many short queries.

## Language

Match the user's language. If the user writes in Chinese, reply in Chinese. If in English, reply in English. Code, commands, and technical terms can stay in English regardless.

## Identity

You are FlagScale Agent. NEVER call yourself Claude, GPT, or any other AI name. When users ask who you are, say "I'm FlagScale Agent."

## User commands

Users can type these slash commands directly (handled by the client, not by you). When users ask about available commands or modes, tell them about these:
- `/mode confirm` — risky shell commands require user confirmation before execution (default)
- `/mode auto` — all shell commands execute without confirmation
- `/memory list` — show all memory entries
- `/memory clear [type]` — clear memory entries, optionally filtered by type (finding/decision/todo/context)
- `/memory delete <key>` — delete a specific memory entry
- `/skill <name>` — load a skill manually
- `/file <path>` — add a file to context
- `/save [path]` — save conversation to file
- `/load <path>` — load a saved conversation
- `/export` — export conversation
- `/plan` — show current task plan status
- `/plan list` — list all plans (including history)
- `/plan abandon` — abandon the current plan
- `/plan clear` — clear completed/abandoned plans
- `/reload` — reload skills and config
- `/quit` — exit the agent

## Shell Command Essentials

- Use `conda run -n <env> <command>`, never `conda activate`. Never install into base env.
- Never `find /` — scope to working directory.
- When using `find`, exclude conda environments and site-packages: `find <path> -name "*.py" -not -path "*/envs/*" -not -path "*site-packages*" -not -path "*__pycache__*"`
- Use `read_file` to read source code, not `sed -n` or `cat`. Read whole files or complete classes/methods.
- For stable training: prefer `wait <PID>` over repeated sleep-check loops.
- Process lifecycle: after `pkill`, verify process is dead (`pgrep -f <pattern>` returns empty) before proceeding. Sequence: kill → verify dead → clean files → relaunch.
- Use FlagScale Launcher (`flagscale train <model> --config <config>`) to launch training. It handles experiment directory layout, per-rank log separation, multi-node coordination, config resolution, and clean shutdown.
- FlagScale launcher caching: `flagscale train --dryrun` generates scripts with hardcoded config values. If you modify config AFTER dryrun, re-run dryrun to regenerate.
- To stop FlagScale training: `flagscale train <model> --config <config> --stop`.
- Before launching, verify no old training processes are alive (`pgrep`).
- Network errors: STOP and tell user to configure proxy.
- Before `rm -rf`: check with `ls`/`du -sh` first. Prefer `mv` to trash over delete.
- Load `ops-discipline` skill for detailed shell rules, dependency resolution, training launch discipline."""


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

        # Workspace manager — split into current.yaml + per-experiment files + hardware.yaml
        workspace_dir = os.path.join(Path.home(), ".flagscale", "workspace_state")
        self._workspace_manager = WorkspaceManager(workspace_dir)
        # Migrate old workspace_state.md if new structure doesn't exist yet
        old_ws_path = os.path.join(Path.home(), ".flagscale", "workspace_state.md")
        if os.path.isfile(old_ws_path) and not os.path.isfile(self._workspace_manager._current_path):
            self._workspace_manager.migrate_from_markdown(old_ws_path)
        self.tool_registry.register(WorkspaceCurrentTool(self._workspace_manager))
        self.tool_registry.register(WorkspaceExperimentTool(self._workspace_manager))
        self.tool_registry.register(WorkspaceHardwareTool(self._workspace_manager))
        # Keep old tool for backward compatibility (reads only)
        self.tool_registry.register(WorkspaceStateTool())
        self._load_plugin_tools()

        memory_dir = os.path.join(Path.home(), ".flagscale", "agent_memory")
        self._session_id = uuid.uuid4().hex[:8]
        self.session_memory = SessionMemory(memory_dir, config.memory_ttl_days)
        self.tool_registry.register(MemoryWriteTool(self.session_memory, self._session_id, workspace_manager=self._workspace_manager))
        self.tool_registry.register(MemoryReadTool(self.session_memory))

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
        self._loaded_skills = set()
        self._interrupted = False
        self._streaming_in_code_block = False
        self._recent_iters = []
        self._last_result_annotations = []
        self._consecutive_train_failures = 0
        self._last_train_failure_reasons = []
        self._error_pattern_history = []  # Track error patterns for smart escalation
        self._context_pressure_soft_warned = False
        self._context_pressure_hard_warned = False
        self._last_checkpoint_tokens = 0  # For progress checkpoint
        self._experiment_registered = False  # True after workspace_state Experiments section is written
        self._dry_run_passed = False  # True after a quick-test / dry-run training command succeeds
        self._last_tool_call = None  # (tool_name, cmd_or_key, was_error) for workaround detection
        self._seen_errors = set()  # Track unique error signatures for checkpoint_new_error
        # Enforcement mechanism state
        self._consecutive_reads = 0  # Progress gate: track read-only tool calls
        self._context_pressure_warned = False  # Context pressure: track if 75% warning shown
        self._last_tool_had_error = False  # Error-escalation gate: track if last tool errored
        self._root_cause_recorded_since_error = False  # Error-escalation gate: track if root cause recorded
        self._files_read_this_session = set()  # Reading depth gate: track files read
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

        # Extract state block
        try:
            state_block = summary_content.split("[State at compaction:")[1].split("]")[0].strip()
        except (IndexError, AttributeError):
            logger.warning("Failed to parse state block from compaction summary")
            return

        # Parse and restore state
        for line in state_block.split("\n"):
            line = line.strip()
            if not line or line == "(no critical state)":
                continue

            if line.startswith("Error patterns:"):
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

    def _refresh_system_prompt(self, memory_context="", plan_context="", workspace_context=""):
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

        # Format the prompt
        prompt = SYSTEM_PROMPT.format(
            skills=skills_text,
            cwd=os.getcwd(),
            memory_context=memory_context,
            plan_context=plan_context,
            workspace_context=workspace_context,
            situational_context=situational_context,
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
        """Build memory context string from recent memories, with task filtering, staleness warnings, and session review hint."""
        task = self._workspace_manager.get_current_task()
        notes = self.session_memory.recent(max_tokens=4000, task_filter=task)
        if not notes:
            return ""
        lines = []
        stale_keys = []
        stale_threshold = 14 * 86400  # 14 days
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
        """Load current.yaml and hardware.yaml for system prompt injection."""
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

        if not parts:
            # Fall back to old workspace_state.md if new format doesn't exist yet
            state_path = os.path.join(Path.home(), ".flagscale", "workspace_state.md")
            if os.path.isfile(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        content = self._truncate_workspace_state(content, max_chars=5000)
                        return f"<workspace-state>\n{content}\n</workspace-state>"
                except Exception:
                    pass
            return ""

        return "<workspace-state>\n" + "\n\n".join(parts) + "\n</workspace-state>"

    @staticmethod
    def _truncate_workspace_state(content: str, max_chars: int = 5000) -> str:
        """Truncate workspace state preserving high-priority sections.

        Priority: Experiments > Hardware > Session Summary > others.
        Drops lowest-priority sections first. If Experiments alone exceeds
        the budget, keeps only the most recent entries.
        """
        if len(content) <= max_chars:
            return content

        PRIORITY = {"Experiments": 0, "Hardware": 1, "Session Summary": 2}

        lines = content.split("\n")
        current_header = ""
        current_lines: list[str] = []
        preamble_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        for line in lines:
            if line.startswith("## "):
                if current_header:
                    sections.append((current_header, "\n".join(current_lines)))
                else:
                    preamble_lines = current_lines
                current_header = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_header:
            sections.append((current_header, "\n".join(current_lines)))
        else:
            # No sections at all — everything is preamble
            preamble_lines = current_lines

        # If no sections exist, just truncate the raw content
        if not sections:
            return content[:max_chars]

        preamble = "\n".join(preamble_lines).strip()
        if len(preamble) > max_chars // 2:
            preamble = preamble[:max_chars // 2]

        sections.sort(key=lambda s: PRIORITY.get(s[0], 10))

        result_parts = [preamble] if preamble else []
        for header, body in sections:
            section_text = f"## {header}\n{body}"
            candidate = "\n\n".join(result_parts + [section_text])
            if len(candidate) <= max_chars:
                result_parts.append(section_text)
            elif header == "Experiments":
                if "### " in body:
                    # Keep only the most recent experiment entries
                    entries = body.split("### ")[1:]
                    kept: list[str] = []
                    for entry in reversed(entries):
                        test_body = "### " + "### ".join(kept + [entry]) if kept else "### " + entry
                        test_section = f"## Experiments\n{test_body}"
                        full = "\n\n".join(result_parts + [test_section])
                        if len(full) <= max_chars:
                            kept.insert(0, entry)
                        else:
                            break
                    if kept:
                        trimmed = "## Experiments\n### " + "### ".join(kept)
                        result_parts.append(trimmed)
                else:
                    # No sub-headers — truncate the body to fit
                    budget = max_chars - len("\n\n".join(result_parts)) - len("## Experiments\n") - 4
                    if budget > 100:
                        result_parts.append(f"## Experiments\n{body[:budget]}")

        return "\n\n".join(result_parts)

    def _inject_context(self, user_input):
        """Auto-inject session memory, plan, and workspace context into the system prompt."""
        memory_context = self._build_memory_context()

        plan_context = self.task_plan.context_for_prompt()

        workspace_context = self._build_workspace_context()

        complexity_hint = ""
        if not plan_context and self.config.auto_plan:
            judge_result = self._complexity_judge(user_input)
            if judge_result.get("needs_plan"):
                complexity_hint = (
                    "\n<system-hint>This task appears complex. "
                    "Consider creating a plan with plan_create before starting execution.</system-hint>\n"
                )
                display.complexity_hint()

        plan_context = plan_context + complexity_hint if complexity_hint else plan_context

        # Session resume gate: remind agent to load workspace state on new session or post-compaction
        resume_hint = ""
        compaction_count = getattr(self.history, 'compaction_count', 0)
        if self._turn_count <= 1 or compaction_count > self._last_compaction_count:
            self._last_compaction_count = compaction_count
            try:
                current = self._workspace_manager.read_current()
            except Exception:
                current = None
            if current:
                task = current.get("task", "")
                status = current.get("status", "")
                if isinstance(task, dict):
                    status = task.get("status", status)
                    task_name = task.get("name", str(task))
                else:
                    task_name = str(task) if task else "unknown"
                if status and status not in ("completed", "abandoned", ""):
                    experiment = current.get("experiment", current.get("current_experiment", ""))
                    if isinstance(experiment, dict):
                        exp_name = experiment.get("name", "")
                    else:
                        exp_name = str(experiment) if experiment else ""
                    resume_hint = (
                        f"\n<system-hint>[SESSION RESUME] Previous work detected:\n"
                        f"  Task: {task_name} (status: {status})\n"
                        f"  Experiment: {exp_name}\n"
                        f"Run workspace_current to load full state. "
                        f"Check memory for findings. Do NOT re-read already-analyzed files.</system-hint>\n"
                    )
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
            print("  auto    — all commands execute without confirmation")
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
            print("Mode: auto — all commands will execute without confirmation.")
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

            self._inject_context(user_input)
            self.history.append({"role": "user", "content": user_input})
            self._react_loop()

    def _run_single_shot(self, query):
        if self.config.auto_skill:
            self._auto_load_skills(query)
        self._inject_context(query)
        self.history.append({"role": "user", "content": query})
        self._react_loop()

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
        Also writes metadata to old workspace_state.md for backward compatibility.
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

        # Also update old workspace_state.md for backward compatibility
        try:
            ws_tool = self.tool_registry.get("workspace_state")
            if ws_tool:
                ws_tool.execute(action="write", content=full_summary, section="Session Summary")
        except Exception as e:
            logger.debug("Old workspace state update skipped: %s", e)

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
        """Record iteration metadata for poll pattern detection."""
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
        return len(set(commands)) == 1 and commands[0]

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
        "writing an experiment entry in workspace_state first.\n"
        "This is a HARD REQUIREMENT. You MUST now:\n"
        "1. Call workspace_state(action='write', section='Experiments', content='### <exp_name> (running)\\n"
        "- **Purpose**: ...\\n- **Config**: ...\\n- **Dir**: ...')\n"
        "   NOTE: Use section='Experiments' — the tool adds the ## header. "
        "Your content should start with ### entries directly.\n"
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
        """Checkpoint: training launched successfully. Auto-record to memory and experiment."""
        current_exp = self._workspace_manager.get_current_experiment()
        if not current_exp:
            return ""

        # Extract key info from command
        cmd_summary = cmd[:200] if len(cmd) <= 200 else cmd[:197] + "..."
        content = f"Training launched: {cmd_summary}"

        # Save to memory
        task = self._workspace_manager.get_current_task()
        warning = ""
        try:
            self.session_memory.put(
                key=f"launch_{int(time.time())}",
                mem_type="finding",
                content=content,
                session_id=self._session_id,
                task=task,
            )
        except Exception as e:
            logger.warning("Failed to save training launch to memory: %s", e)
            warning = f"\n⚠️ Memory write failed: {e}. Training launch not recorded.\n"

        # Auto-add attempt to experiment
        try:
            self._workspace_manager.add_attempt(current_exp, "Training launched", "Running...")
        except Exception as e:
            logger.warning("Failed to add experiment attempt: %s", e)
            warning += f"\n⚠️ Experiment update failed: {e}. Attempt not recorded.\n"

        return warning

    def _checkpoint_training_failure(self, cmd: str, result: str):
        """Checkpoint: training failed. Auto-record to memory and experiment."""
        current_exp = self._workspace_manager.get_current_experiment()
        if not current_exp:
            return ""

        # Extract error summary (first meaningful error line)
        error_summary = self._extract_error_summary(result)
        content = f"Training failed: {error_summary}"

        # Save to memory
        task = self._workspace_manager.get_current_task()
        warning = ""
        try:
            self.session_memory.put(
                key=f"fail_{int(time.time())}",
                mem_type="finding",
                content=content,
                session_id=self._session_id,
                task=task,
            )
        except Exception as e:
            logger.warning("Failed to save training failure to memory: %s", e)
            warning = f"\n⚠️ Memory write failed: {e}. Training failure not recorded.\n"

        # Auto-update experiment attempt
        try:
            self._workspace_manager.update_last_attempt(current_exp, f"FAILED: {error_summary}")
        except Exception as e:
            logger.warning("Failed to update experiment attempt: %s", e)
            warning += f"\n⚠️ Experiment update failed: {e}. Failure not recorded.\n"

        return warning

    def _checkpoint_workaround(self, tool_name: str, prev_error: str, curr_cmd: str):
        """Checkpoint: workaround found (same tool, prev failed, now succeeded)."""
        prev_summary = prev_error[:100] if len(prev_error) <= 100 else prev_error[:97] + "..."
        cmd_summary = curr_cmd[:100] if len(curr_cmd) <= 100 else curr_cmd[:97] + "..."
        content = f"Workaround: {tool_name} failed with [{prev_summary}], fixed by [{cmd_summary}]"

        task = self._workspace_manager.get_current_task()
        warning = ""
        try:
            self.session_memory.put(
                key=f"workaround_{int(time.time())}",
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
        """Checkpoint: new unique error encountered."""
        if not hasattr(self, "_seen_errors"):
            self._seen_errors = set()

        if error_signature in self._seen_errors:
            return ""
        self._seen_errors.add(error_signature)

        content = f"New error: {error_signature}"
        task = self._workspace_manager.get_current_task()
        warning = ""
        try:
            self.session_memory.put(
                key=f"error_{error_signature[:40].replace(' ', '_')}",
                mem_type="finding",
                content=content,
                session_id=self._session_id,
                task=task,
            )
        except Exception as e:
            logger.warning("Failed to save error to memory: %s", e)
            warning = f"\n⚠️ Memory write failed: {e}. Error not recorded.\n"

        return warning

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
            }
            state_path = os.path.join(self._workspace_manager._dir, ".agent_state.json")
            import json
            with open(state_path, "w") as f:
                json.dump(state, f)
            logger.info("Mid-turn autosave: %d files, phase=%s, verification=%s",
                        len(self._files_read_this_session), self._current_phase, self._verification_stage)
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

    _PROGRESS_GATE_THRESHOLD = 8

    _PRODUCTIVE_TOOLS = frozenset({
        "memory_write", "write_file", "edit_file",
        "workspace_experiment", "workspace_current",
        "plan_update", "plan_create",
    })

    _READ_ONLY_TOOLS = frozenset({
        "read_file", "shell", "web_fetch", "find_latest_log",
        "parse_training_metrics", "memory_read",
    })

    def _check_progress_gate(self, tool_name):
        """Track consecutive read-only calls. Force checkpoint when stuck.

        Relaxed during debugging (consecutive failures) to allow deep investigation.
        """
        if tool_name in self._PRODUCTIVE_TOOLS:
            self._consecutive_reads = 0
            return ""
        if tool_name in self._READ_ONLY_TOOLS:
            self._consecutive_reads += 1

        # Relax threshold during debugging (consecutive failures)
        threshold = self._PROGRESS_GATE_THRESHOLD
        if self._consecutive_train_failures >= 2:
            threshold = 15  # Allow deeper investigation during debugging

        if self._consecutive_reads >= threshold:
            self._consecutive_reads = 0
            return (
                "\n\n[PROGRESS CHECK] You've made many consecutive read/shell calls "
                "without recording any findings. Before continuing:\n"
                "1. What specific question are you trying to answer?\n"
                "2. Write what you've learned so far to memory_write.\n"
                "3. Then continue with a focused goal."
            )
        return ""

    def _check_dry_run_gate(self, cmd, result):
        """Enforce dry-run before full training."""
        if not self._TRAIN_LAUNCH_RE.search(cmd):
            return result
        if self._is_quick_test_command(cmd):
            self._dry_run_passed = True
            return result + (
                "\n\n[DRY RUN COMPLETE] Verify: model loaded? data flowing? "
                "no crashes? If OK, proceed to full run."
            )
        if not self._dry_run_passed:
            return result + (
                "\n\n[WARNING: NO DRY RUN] This is a full training run without "
                "prior dry-run verification. Issues like unloaded checkpoints, "
                "broken data pipelines, or config errors will waste GPU hours. "
                "Consider stopping and running with --max-steps=2 first."
            )
        return result

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
        total = self._session_input_tokens + self._session_output_tokens
        max_ctx = self.config.max_context_tokens
        if not max_ctx or max_ctx <= 0:
            return ""
        ratio = total / max_ctx

        if ratio > 0.85:
            self.history.force_compact(target_ratio=0.60)
            self._context_pressure_soft_warned = False
            self._context_pressure_hard_warned = False
            return ""

        if ratio > 0.75 and not self._context_pressure_hard_warned:
            self._context_pressure_hard_warned = True
            return (
                "\n\n[CONTEXT PRESSURE] You've used ~75% of context budget. "
                "Write ALL key findings to memory NOW: what you've learned, "
                "what files you've read, what's left to do. "
                "After that, history will be auto-compacted to free space."
            )

        if ratio > 0.60 and not self._context_pressure_soft_warned:
            self._context_pressure_soft_warned = True
            return (
                "\n\n[CONTEXT BUDGET] You've used ~60% of context. "
                "Start writing key findings to memory. "
                "Avoid re-reading files you've already analyzed. "
                "Use read_file with offset/limit for targeted reads."
            )

        return ""

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

    # ── Infinite loop / duplicate tool call detection ──────────────────

    _LOOP_DETECTION_WINDOW = 10
    _LOOP_DETECTION_THRESHOLD = 3
    _AUTOSAVE_INTERVAL = 10  # Save state every N tool calls within a turn

    def _get_tool_call_key(self, tool_name, arguments):
        """Generate a hashable key for a tool call.

        For read_file and edit_file, include offset/line range to distinguish
        different parts of the same file.
        """
        if tool_name == "shell":
            return (tool_name, arguments.get("command", ""))
        elif tool_name == "read_file":
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 0)
            return (tool_name, path, offset, limit)
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
            display.warn(f"Loop detected: {tool_name} called {count}x with same args")
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
        """Detect duplicate tool calls within one turn (cache hit)."""
        if tool_name == "read_file":
            path = arguments.get("path", "")
            if not path:
                return None
            key = ("read_file", path)
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
                self._tool_call_cache[("read_file", path)] = result
        elif tool_name == "memory_write" and "ERROR" not in result[:20]:
            mem_key = arguments.get("key", "")
            if mem_key:
                self._tool_call_cache[("memory_write", mem_key)] = result

    # ── Error-to-skill auto-loading ────────────────────────────────────

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
            self.history.append(self.provider.format_tool_result(tool_call_id, content))
            self._loaded_skills.add(skill_name)
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
        schemas = self.tool_registry.to_schemas(self.provider.schema_format)
        self._turn_count += 1
        self._interrupted = False
        turn_start = time.time()
        turn_input_tokens = 0
        turn_output_tokens = 0
        max_iter = self.config.max_iterations
        iteration = 0

        while iteration < max_iter:
            if self._interrupted:
                break

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

            self._record_iteration(response["tool_calls"], results, output_tok, tool_elapsed_list)

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
            results[i] = {"tool": tool_calls[i]["name"],
                          "result": "DENIED: User declined to execute this command."}

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
            if tool_name == "shell":
                annotations = self._result_judge(arguments.get("command", ""), result, elapsed)
                annotations = self._dedup_annotations(annotations)
                if annotations:
                    header = "\n".join(f"[{a}]" for a in annotations)
                    result = header + "\n" + result
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

        # Progress gate check
        progress_warning = self._check_progress_gate(tool_name)

        # Error-escalation gate check
        escalation_warning = self._check_error_escalation(tool_name, arguments)

        # Analysis persistence and verification gates (soft warnings)
        analysis_warning = self._check_analysis_persistence(tool_name, arguments)
        verification_warning = self._check_verification_ladder(tool_name, arguments)

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
                self._files_read_this_session.add(path)
                # Track reading categories for quality gate
                for cat, pattern in self._PORTING_READ_CATEGORIES.items():
                    if pattern.search(path):
                        self._reading_categories.add(cat)
            if len(result) > self._READ_FILE_SUMMARY_THRESHOLD:
                result = self._summarize_file_content(result, path)

        # Track porting mode activation
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if "model-porter" in skill_name:
                self._porting_mode = True
            if "data-prep" in skill_name:
                self._data_prep_mode = True
            result = self._maybe_summarize_skill(skill_name, result)

        # Track porting path confirmation (plan_create or workspace_experiment with path decision)
        if self._porting_mode and not self._porting_path_confirmed and not error:
            args_lower = str(arguments).lower()
            if tool_name == "plan_create" and any(
                kw in args_lower for kw in ("mode b", "mode c", "native", "huggingface wrapper", "hf wrapper", "fsdp2")
            ):
                self._porting_path_confirmed = True
                logger.info("Porting path confirmed via plan_create")
            elif tool_name == "workspace_experiment" and "porting_path" in args_lower:
                self._porting_path_confirmed = True
                logger.info("Porting path confirmed via workspace_experiment")

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
        # Also track old workspace_state tool for backward compatibility
        if (tool_name == "workspace_state" and not error):
            section = arguments.get("section", "")
            if section and "experiment" in section.lower():
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
                    display.warn("Training launched without experiment entry in workspace_state!")
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

        # Inject enforcement warnings (with user notifications)
        if progress_warning:
            result = progress_warning + "\n" + result
            display.warn("Progress gate: too many tool calls without output")
        if escalation_warning:
            result = escalation_warning + "\n" + result
            display.warn("Error escalation: repeated failures require root cause analysis")
        if analysis_warning:
            result = analysis_warning + "\n" + result
            display.warn("Analysis persistence: findings must be saved before implementation")
        if verification_warning:
            result = verification_warning + "\n" + result
            display.warn("Verification ladder: must complete verification stages in order")
        if phase_warning:
            result = phase_warning + "\n" + result
            display.warn("Phase transition: prerequisites not met for next phase")

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
                self.history.append(self.provider.format_tool_result(tool_call_id, content))
                self._loaded_skills.add(skill_name)
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
        self.history.append(self.provider.format_tool_result(tool_call_id, content))
        self._loaded_skills.add(skill_name)

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