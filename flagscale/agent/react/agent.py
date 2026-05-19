"""FlagScale Agent — ReAct loop with composable Guard/Checklist/Judge architecture.

No Mixin inheritance. State is owned by Guard instances.
Scene + Profile parameterize behavior without subclassing.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle

from flagscale.agent.react import display
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.history import HistoryManager
from flagscale.agent.react.logger import setup_logging
from flagscale.agent.react.providers import get_provider
from flagscale.agent.react.retry import retry_with_backoff, _is_context_limit_error
from flagscale.agent.react.session import (
    save_conversation, load_conversation, mark_completed,
    find_resumable_sessions, list_sessions, get_session_dir,
    append_session_index, get_recent_sessions,
)
from flagscale.agent.react.experiment_manager import ExperimentManager
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
from flagscale.agent.react.tools.workspace_experiment import WorkspaceExperimentTool
from flagscale.agent.react.memory import SessionMemory
from flagscale.agent.react.tools.memory_write import MemoryWriteTool
from flagscale.agent.react.tools.memory_read import MemoryReadTool
from flagscale.agent.react.tools.memory_list import MemoryListTool
from flagscale.agent.react.plan import TaskPlan
from flagscale.agent.react.tools.monitor import MonitorTool
from flagscale.agent.react.tools.plan_create import PlanCreateTool
from flagscale.agent.react.tools.plan_update import PlanUpdateTool
from flagscale.agent.react.tools.plan_status import PlanStatusTool
from flagscale.agent.react.tools.validate_config import ValidateConfigTool
from flagscale.agent.react.tools.inspect_checkpoint import InspectCheckpointTool

from flagscale.agent.react.guard.safety import SafetyGuard
from flagscale.agent.react.guard.loop_detect import LoopDetectGuard
from flagscale.agent.react.guard.progress import ProgressGuard
from flagscale.agent.react.guard.context_pressure import ContextPressureGuard
from flagscale.agent.react.guard.plan import PlanGuard
from flagscale.agent.react.guard.training_runtime import TrainingRuntimeGuard
from flagscale.agent.react.guard.constraint import ConstraintGuard
from flagscale.agent.react.guard.warning import WarningGuard
from flagscale.agent.react.guard.error_classifier import ErrorClassifierGuard
from flagscale.agent.react.guard.circuit_breaker import CircuitBreakerGuard
from flagscale.agent.react.guard.budget import BudgetGuard
from flagscale.agent.react.constraint.cache import ConstraintCache
from flagscale.agent.react.prompt_builder import PromptBuilder
from flagscale.agent.react.tool_executor import ToolExecutor, tool_display_summary

from flagscale.agent.react.checklist.base import ChecklistEngine, ChecklistItem, Checklist
from flagscale.agent.react.judge import Judge, JudgeBudget
from flagscale.agent.react.scene import ScenePreset, PRESETS
from flagscale.agent.react.profile import WorkerProfile, PROFILES
from flagscale.agent.react.constants import (
    READ_ONLY_TOOLS,
    CORE_TOOLS,
    PHASE_TOOL_SETS,
    GATE_OVERRIDE_RE,
    CHECKLIST_OVERRIDE_RE,
    READ_FILE_SUMMARY_THRESHOLD,
    READ_FILE_SUMMARY_THRESHOLD_PORTING,
)
from flagscale.agent.react.commands import CommandHandler

logger = logging.getLogger(__name__)


# ── WorkerResult ───────────────────────────────────────────────────────────────

@dataclass
class WorkerResult:
    """Structured result from WorkerAgent.execute().

    Used by Orchestrator to compose multi-stage pipeline results.
    status: "success" | "failed" | "partial"
    """

    status: str  # "success", "failed", "partial"
    summary: str = ""
    artifacts: dict = field(default_factory=dict)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    turn_count: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    elapsed_seconds: float = 0.0


# ── _ModeFlags ─────────────────────────────────────────────────────────────────

@dataclass
class _ModeFlags:
    """Dynamic mode flags — set by skill effects, not hardcoded.

    _active_modes: arbitrary mode strings set by skill frontmatter effects.
    Workflow state fields are scenario-agnostic (work for training and inference).
    """

    _active_modes: set = field(default_factory=set)

    # Scenario-agnostic workflow state
    runtime_started: bool = False       # training started / inference serving started
    env_compat_analyzed: bool = False
    path_confirmed: bool = False        # user confirmed approach (porting path, deploy path, etc.)
    confirmed_path: str | None = None   # which approach was confirmed

    def has(self, mode: str) -> bool:
        """Check if a mode is active."""
        return mode in self._active_modes

    def set(self, mode: str):
        """Activate a mode."""
        self._active_modes.add(mode)


# ── WorkerAgent ──────────────────────────────────────────────────────────────

class WorkerAgent:
    """Single agent class with composable Guard/Checklist/Judge architecture.

    No Mixin inheritance. State that belongs to Guards is owned
    by Guard instances. All infrastructure is composed via __init__.
    """

    def __init__(self, config: AgentConfig, scene: ScenePreset | None = None,
                 # ── Shared infrastructure (for Orchestrator injection) ──
                 _provider=None, _tool_registry=None, _skill_manager=None,
                 _session_memory=None, _task_plan=None, _experiment_manager=None):
        setup_logging()
        self.config = config
        self.scene = scene

        # ── Infrastructure ──
        self.skill_manager = _skill_manager or SkillManager(config.skill_dirs)
        self.tool_registry = _tool_registry or ToolRegistry()

        self._session_id = uuid.uuid4().hex[:8]
        sessions_root = config.session_dir or os.path.join(Path.home(), ".flagscale", "sessions")
        session_dir = os.path.join(sessions_root, self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        self._sessions_root = sessions_root

        experiments_dir = os.path.join(session_dir, "experiments")
        self._experiment_manager = _experiment_manager or ExperimentManager(experiments_dir)

        memory_dir = os.path.join(Path.home(), ".flagscale", "agent_memory")
        self.session_memory = _session_memory or SessionMemory(memory_dir, config.memory_ttl_days)

        plan_dir = os.path.join(session_dir, "plans")
        self.task_plan = _task_plan or TaskPlan(plan_dir)

        if not _tool_registry:
            self._register_tools()
        if not _experiment_manager:
            self._load_plugin_tools()
        self.tool_registry.register(MemoryWriteTool(self.session_memory, self._session_id))
        self.tool_registry.register(MemoryReadTool(self.session_memory))
        self.tool_registry.register(MemoryListTool(self.session_memory))
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        self.tool_registry.register(PlanUpdateTool(self.task_plan))
        self.tool_registry.register(PlanStatusTool(self.task_plan))

        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = _provider or get_provider(
            config.provider, config.model, config.api_key,
            config.base_url, config.max_output_tokens,
        )

        self.session_memory._llm_fn = lambda prompt: self.provider.chat(
            [{"role": "user", "content": prompt}], tools=[]
        ).get("content", "")

        self.history = HistoryManager(max_context_tokens=config.max_context_tokens)
        self.history.set_summarizer(self._summarize_for_compaction)
        self.history.set_scorer(self._score_messages_for_compaction)

        # ── Orchestrator (set by run_agent_orchestrated) ──
        self._orchestrator = None

        # ── Command handler ──
        self._command_handler = CommandHandler(self)

        # ── Prompt builder ──
        self._prompt_builder = PromptBuilder(self.skill_manager, self.scene)

        # ── Tool executor ──
        self._tool_executor = ToolExecutor(self)

        # ── Composed components ──
        self.judge = Judge(self.provider, budget=JudgeBudget(max_calls_per_turn=16))
        self._loaded_skills: set[str] = set()
        self._constraint_cache = ConstraintCache(self._sessions_root)
        self.checklist: Checklist | None = self._build_checklist()

        self._init_runtime_state()
        atexit.register(self._atexit_hook)

    def _init_runtime_state(self):
        """Initialize mutable per-session state. Called from __init__.

        Extracted to keep __init__ focused on dependency wiring.
        Can be re-called for tests or worker resets.
        """
        self._phase_override: str | None = None  # Only set for testing
        self.turn_count: int = 0
        self._interrupted: bool = False
        self._last_tool_calls_deque = deque(maxlen=5)
        self._extra_tools_next_iter: set[str] = set()
        self._turn_iteration_count: int = 0
        self._consecutive_single_tool_calls: int = 0
        self._active_skill_content: dict[str, str] = {}
        self._skill_load_iterations: dict[str, int] = {}
        self._total_iterations: int = 0
        self._recently_referenced_skills: set[str] = set()
        self.modes = _ModeFlags()
        self._mode_phase_map: dict[str, str] = {}  # mode → initial phase, built from skill effects
        self._original_user_task: str = ""
        self._session_start: float = time.time()
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._auto_turn_count: int = 0
        self._last_write_turn: int = 0
        self._code_written: bool = False
        self._files_read_this_session: set[str] = set()
        self._files_written_this_session: set[str] = set()
        self._last_checkpoint_tokens: int = 0
        self._last_tool_call: tuple | None = None
        self._tool_call_cache: dict[tuple, str] = {}
        self._streaming_in_code_block: bool = False
        self._last_compaction_count: int = 0
        self._recent_iters: list[dict] = []
        self._current_stage_id: str | None = None  # For focused context injection

        self._refresh_system_prompt()

        # ── Initialize Kernel ──
        self._kernel = self._build_kernel()

    def _build_kernel(self):
        """Build AgentKernel with injected dependencies."""
        from flagscale.agent.react.kernel import AgentKernel, KernelDeps
        from flagscale.agent.react.guard import GuardRegistry

        guard_registry = GuardRegistry()
        # Register native guards
        constraints = self.scene.constraints if self.scene else set()
        guard_registry.register(SafetyGuard())

        # Reliability guards (P7)
        self._budget_guard = BudgetGuard(
            max_tokens=self.config.budget_max_tokens,
            max_tool_calls=self.config.budget_max_tool_calls,
        )
        guard_registry.register(self._budget_guard)
        guard_registry.register(CircuitBreakerGuard(
            trip_threshold=self.config.circuit_breaker_threshold,
            cooldown_iters=self.config.circuit_breaker_cooldown,
        ))
        guard_registry.register(LoopDetectGuard())
        guard_registry.register(ErrorClassifierGuard())

        # Create ConstraintGuard (will be populated with Skill constraints later)
        self._constraint_guard = ConstraintGuard()
        guard_registry.register(self._constraint_guard)

        guard_registry.register(ProgressGuard())
        guard_registry.register(ContextPressureGuard())
        guard_registry.register(PlanGuard(task_plan=self.task_plan))
        if "is_training" in constraints or "always" in constraints or not constraints:
            guard_registry.register(TrainingRuntimeGuard())

        deps = KernelDeps(
            provider=self.provider,
            history=self.history,
            tool_registry=self.tool_registry,
            judge=self.judge,
            guard_registry=guard_registry,
            config=self.config,
            display=display,
            get_schemas_fn=lambda: self._get_filtered_schemas(self.phase),
            inject_message_fn=self._inject_message,
            append_tool_results_fn=self._append_tool_results,
            format_tool_result_fn=self.provider.format_tool_result,
            execute_tools_fn=self._execute_tools,
            is_context_limit_error_fn=self._is_context_limit_error,
            call_llm_fn=self._call_llm_stream,
            task_plan=self.task_plan,
            on_response_fn=self._on_kernel_response,
            on_tool_results_fn=self._on_kernel_tool_results,
        )
        return AgentKernel(deps)

    # ── Initialization helpers ───────────────────────────────────────────────

    def _register_tools(self):
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(EditFileTool())
        self.tool_registry.register(
            ShellTool(
                remind_interval=self.config.shell_remind_interval,
                check_dangerous=self.config.dangerous_commands_check,
                require_confirm=self.config.confirm_commands,
                env=self.config.shell_env,
                health_judge_fn=self._health_judge,
            )
        )
        self.tool_registry.register(LoadSkillTool(self.skill_manager))
        self.tool_registry.register(WebFetchTool(proxies=self._build_proxies()))
        self.tool_registry.register(FindLatestLogTool())
        self.tool_registry.register(ParseTrainingMetricsTool())
        self.tool_registry.register(MonitorTool(classify_fn=self._judge_confirm))
        self.tool_registry.register(WorkspaceExperimentTool(self._experiment_manager, task_plan=self.task_plan))
        self.tool_registry.register(ValidateConfigTool())
        self.tool_registry.register(InspectCheckpointTool())

    def _build_checklist(self) -> Checklist | None:
        engine = ChecklistEngine()
        items = []

        # ── Auto-detect runtime facts for checklist evaluation ──
        shared_paths = self._detect_shared_storage()
        self._shared_storage_paths = shared_paths

        # Inject facts into the engine so every classify() call sees them
        engine.facts = {}
        if shared_paths:
            engine.facts["shared_storage"] = shared_paths

        # ── Skill-based constraints (LLM-extracted from skill content) ──
        for skill_name in self._loaded_skills:
            constraint_specs = self._constraint_cache.items.get(skill_name)
            if not constraint_specs:
                continue
            for c in constraint_specs:
                try:
                    items.append(ChecklistItem(
                        id=c.get("id", f"{skill_name}_constraint"),
                        description=c.get("description", ""),
                        phases=set(c.get("phases", ["*"])),
                        trigger_on=c.get("trigger_on"),
                        prompt=c.get("prompt", ""),
                        reminder=c.get("reminder", ""),
                        severity=c.get("severity", "warning"),
                        max_reminders=c.get("max_reminders", 3),
                    ))
                except Exception:
                    pass

        # ── Shared storage constraint (dynamic, based on actual mounts) ──
        if shared_paths:
            path_list = ", ".join(shared_paths)
            items.append(ChecklistItem(
                id="env_shared_storage",
                description=f"Shared storage detected at {path_list}. Conda envs must use --prefix on shared storage.",
                phases={"*"},
                trigger_on={"tool": "shell"},
                prompt=(
                    f"DETECT if this shell command creates a conda environment using -n/--name "
                    f"(local node storage) instead of --prefix on shared storage.\n"
                    f"CONTEXT: shared storage IS available at: {path_list}. "
                    f"The _facts confirm this.\n"
                    f"MATCH: 'conda create' with -n or --name flag, WITHOUT a --prefix "
                    f"pointing to the shared storage.\n"
                    f"DO NOT MATCH: conda create with --prefix, or any non-conda-create command."
                ),
                reminder=(
                    f"⚠ [ENV-SETUP] Shared storage detected at: {path_list}. "
                    f"Creating conda env with -n/--name stores it on local node only. "
                    f"Use --prefix on shared storage instead (e.g. --prefix {shared_paths[0]}/envs/<name>). "
                    f"This ensures multi-node training can access the same environment."
                ),
                severity="warning",
                max_reminders=3,
            ))

        if not items:
            return None
        return Checklist(engine=engine, items=items)

    def _rebuild_checklist(self):
        """Rebuild the checklist to include newly loaded skill constraints."""
        self.checklist = self._build_checklist()
        if self.checklist and self.checklist._items:
            print(display.dim(
                f"  📋 Checklist now has {len(self.checklist._items)} active constraint(s)"
            ))

    def _extract_skill_constraints(self, skill_name: str, skill_content: str):
        """Extract checklist constraints from skill content via LLM judge."""
        self._constraint_cache.get_or_extract(
            skill_name, skill_content, self.judge.extract_constraints
        )

    def _on_skill_loaded(self, skill_name: str, skill_content: str,
                          skip_extract: bool = False):
        """Centralized handler after any skill is loaded.

        If skip_extract is True, constraint extraction is deferred (used
        when batching multiple skill loads with concurrent extraction).
        """
        # Register frontmatter-defined guards (structured constraints/warnings)
        self._register_skill_guards(skill_name)
        if not skip_extract:
            self._extract_skill_constraints(skill_name, skill_content)
        self._rebuild_checklist()
        self._refresh_system_prompt()

    def _batch_extract_and_rebuild(self, skill_map: dict[str, str]):
        """Extract constraints from multiple skills concurrently, then rebuild once."""
        self._constraint_cache.batch_extract(skill_map, self.judge.extract_constraints)
        self._rebuild_checklist()
        self._refresh_system_prompt()

    @staticmethod
    def _detect_shared_storage() -> list[str]:
        """Detect shared/network filesystem mount points.

        Checks common mount points and the current working directory.
        Returns a (possibly empty) list of shared filesystem paths.
        """
        candidates = [
            "/share", "/mnt/share", "/mnt/cfs", "/mnt/dfs",
            "/mnt/nfs", "/mnt/lustre", "/data/shared", "/shared",
        ]
        shared = []
        for path in candidates:
            if os.path.ismount(path):
                shared.append(path)

        # Also check if cwd is under a shared FS
        cwd = os.getcwd()
        # Look for FUSE, NFS, CIFS, Lustre, GPFS in mount info
        try:
            with open("/proc/mounts") as f:
                mounts = f.read()
            for line in mounts.splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                mount_point = parts[0]
                mount_path = parts[1]
                fs_type = parts[2] if len(parts) > 2 else ""
                # Network/shared filesystem types
                if fs_type in ("fuse", "nfs", "nfs4", "cifs", "lustre", "gpfs",
                               "glusterfs", "ceph", "pvfs2", "afs", "beegfs"):
                    if mount_path not in shared:
                        shared.append(mount_path)
                # Check if cwd or its parents match a mount
                if mount_path != "/" and cwd.startswith(mount_path + "/"):
                    if mount_path not in shared:
                        shared.append(mount_path)
        except Exception:
            pass

        # Sort by path length (shorter = more general) so the most general
        # shared path comes first for --prefix suggestions
        shared.sort(key=len)
        return shared

    # Keys from session memory that are relevant to checklist evaluation.
    # Anything not in this list is NOT injected into engine.facts to
    # avoid polluting the LLM's context with irrelevant memory entries.
    _FACT_KEYS_WHITELIST = frozenset({
        "env_path", "env_name", "conda_env_path", "conda_env",
        "cuda_version", "driver_version", "gpu_type", "gpu_count",
        "model_path", "model_name", "checkpoint_path",
        "data_path", "dataset_path",
        "shared_storage", "workspace_root", "deps_dir",
        "torch_version",
    })

    def _inject_runtime_facts(self, engine):
        """Refresh ChecklistEngine.facts with latest session memory context.

        Called before each checklist evaluation so the LLM has up-to-date
        runtime knowledge (env path, cuda version, model path, etc.).
        Only injects whitelisted keys to avoid context pollution.
        """
        base_facts = dict(engine.facts)
        if self.session_memory:
            for entry in self.session_memory.list_entries():
                key = entry.get("key", "")
                content = str(entry.get("content", ""))
                if not key or not content:
                    continue
                if key not in self._FACT_KEYS_WHITELIST:
                    continue
                base_facts[key] = content[:300]
        engine.facts = base_facts

    def _load_plugin_tools(self):
        for tool_dir in self.config.plugin_tool_dirs:
            if not os.path.isdir(tool_dir):
                continue
            for entry in os.listdir(tool_dir):
                if not entry.endswith(".py") or entry.startswith("_"):
                    continue
                path = os.path.join(tool_dir, entry)
                try:
                    with open(path) as f:
                        exec(f.read(), {"__file__": path})
                    logger.info("Loaded plugin tool: %s", entry)
                except Exception:
                    logger.warning("Failed to load plugin tool %s: %s", entry, sys.exc_info()[1])

    def _build_proxies(self) -> dict[str, str]:
        proxies = {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            val = os.environ.get(var)
            if val:
                proxies[var.lower()] = val
        return proxies

    # ── System prompt ────────────────────────────────────────────────────────

    def _refresh_system_prompt(self, memory_context: str = "", plan_context: str = ""):
        tool_names = [t.name for t in self.tool_registry.all_tools()]
        self._prompt_builder.refresh(
            history=self.history,
            active_skill_content=self._active_skill_content,
            current_stage_id=self._current_stage_id,
            shared_storage_paths=getattr(self, "_shared_storage_paths", []),
            memory_context=memory_context,
            plan_context=plan_context,
            tool_names=tool_names,
        )

    # ── GuardContext builder ────────────────────────────────────────────────

    def _build_obs(
        self,
        tool_name: str = "",
        tool_args: dict | None = None,
        tool_result: str | None = None,
    ) -> "GuardContext":
        from flagscale.agent.react.guard import GuardContext
        from flagscale.agent.react.tools.base import ToolEffect
        tool_effects = ToolEffect()
        try:
            tool = self.tool_registry.get(tool_name)
            tool_effects = tool.effects
        except (KeyError, AttributeError):
            pass
        return GuardContext(
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_result=tool_result,
            tool_effects=tool_effects,
            turn_count=self.turn_count,
            recent_tool_names=[t[0] for t in list(self._last_tool_calls_deque)[-10:]],
            context_pressure=self.history.get_context_pressure() if self.history else 0.0,
            current_state=self._kernel.fsm.current_state,
            transitions_count=len(self._kernel.fsm.history),
            classify_fn=self.judge.classify,
            experiment_compare_fn=self._experiment_manager.compare if self._experiment_manager else None,
            experiment_diff_fn=self._experiment_manager.diff_last_attempts if self._experiment_manager else None,
            current_experiment_name=self._experiment_manager.get_current_experiment() if self._experiment_manager else "",
        )

    # ── Health judge (delegates to unified Judge) ───────────────────────────

    def _health_judge(self, command: str, recent_output: str, elapsed: str,
                      output_changed: bool = True, stall_count: int = 0) -> dict:
        return self.judge.health(command, recent_output, elapsed, output_changed, stall_count)

    def _judge_confirm(self, category: str, matched_text: str, context: str = "") -> bool:
        return self.judge.classify(category, {"text": matched_text, "context": context}, default=True)

    # ── Atexit ──────────────────────────────────────────────────────────────

    def _atexit_hook(self):
        try:
            self._save_conversation(completed=False)
        except Exception:
            pass

    def _save_conversation(self, completed: bool = False):
        if not self.history.messages:
            return
        save_conversation(
            self._session_dir, self._session_id,
            self.history.messages,
            loaded_skills=list(self._loaded_skills),
            completed=completed,
        )

    def _exit(self):
        # ── Checklist session summary ──
        if self.checklist and self.checklist.total_checks > 0:
            c = self.checklist
            print(display.dim(
                f"\n📋 Checklist: {c.total_checks} checks, "
                f"{c.total_violations} violation(s) across "
                f"{len(c.violation_by_id)} constraint(s)"
            ))
            if c.violation_by_id:
                for cid, count in sorted(c.violation_by_id.items(), key=lambda x: -x[1]):
                    print(display.dim(f"   [{cid}] violated {count}x"))

        display.goodbye()
        self._save_conversation(completed=True)
        mark_completed(self._session_dir)
        sys.exit(0)

    # ── Main entry ──────────────────────────────────────────────────────────

    def run(self, single_shot_query: str | None = None):
        if single_shot_query:
            self._run_single_shot(single_shot_query)
            return

        extra = self._startup_hints()
        display.banner(self.config.provider, self.config.model, mode=self.config.mode, extra_lines=extra)
        self._check_proxy()
        self._check_resume()

        history_file = os.path.join(os.path.expanduser("~"), ".flagscale", "input_history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        completer = WordCompleter(
            ["/quit", "/reload", "/skill", "/file", "/save", "/load",
             "/export", "/memory", "/mode", "/plan", "/resume", "/compact"],
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

            if self._command_handler.handle_slash_command(user_input):
                continue

            # ── Orchestrator routing ──
            if self._orchestrator is not None:
                self._run_orchestrated(user_input)
                continue

            # Detect scene
            if self.scene is None:
                self.scene = ScenePreset.auto_detect(user_input=user_input)

            if self.config.auto_skill:
                self._auto_load_skills(user_input)

            self._auto_turn_count = 0
            self._inject_context(user_input)
            self._check_user_porting_confirmation(user_input)
            self.history.append({"role": "user", "content": user_input})
            try:
                self._react_loop()
            except KeyboardInterrupt:
                display.interrupted()
                continue

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

    def _run_orchestrated(self, user_input: str):
        """Route user input via Orchestrator and dispatch to execution mode.

        Called from run() when self._orchestrator is set.
        Displays stage progress for subtask mode, handles Ctrl+C for cancellation.
        """
        o = self._orchestrator
        route = o.route(user_input)

        mode = route.get("mode", "single")
        template = route.get("template", "")
        dynamic_stages = route.get("dynamic_stages", [])

        # ── Routing display ──
        source_parts = []
        if template:
            source_parts.append(f"template={template}")
        if dynamic_stages:
            source_parts.append("stages=dynamic")
        if not source_parts:
            source_parts.append("default")
        source_str = ", ".join(source_parts)
        print(display.dim(f"\n[Orchestrator] mode={mode}, {source_str}"))

        # ── Subtask mode: show stage overview, then serial execution ──
        if mode == "subtask":
            subtasks = o._build_subtask_definitions(route, user_input)
            if not subtasks:
                print(display.red("\nNo stages to execute for this task."))
                return

            total = len(subtasks)
            print(f"\n[Orchestrator] Task will be split into {total} stage{'s' if total > 1 else ''}:")
            for i, sub in enumerate(subtasks, 1):
                print(f"  Stage {i}/{total}: {sub.id} — {sub.description}")
            print()

            upstream: dict[str, str] = {}
            batches = o.subtask_runner._topological_batches(subtasks)
            stage_idx = 0

            try:
                for batch in batches:
                    for sub in batch:
                        stage_idx += 1
                        self._current_stage_id = sub.id
                        self._refresh_system_prompt()
                        print(f"[Stage {stage_idx}/{total}] Running: {sub.id}...")
                        context = o.subtask_runner._build_upstream_summary(
                            sub.upstream_keys, upstream
                        )

                        worker = o._create_worker(sub.profile_name)
                        task = o.subtask_runner._build_task(
                            sub.description, user_input, context
                        )

                        result = worker.execute(task)
                        if result.status == "failed":
                            print(display.red(
                                f"  ✗ {sub.id} failed: {result.summary[:200]}"
                            ))
                            self._inject_subtask_result_to_history(
                                f"[Stage {stage_idx}/{total}] {sub.id}: FAILED — {result.summary[:300]}"
                            )
                            upstream.update(result.artifacts)
                            upstream[sub.id] = result.summary
                            return

                        upstream.update(result.artifacts)
                        upstream[sub.id] = result.summary
                        art_str = ", ".join(
                            f"{k}={str(v)[:60]}" for k, v in result.artifacts.items()
                        ) if result.artifacts else "none"
                        print(f"  ✓ {sub.id} complete. Artifacts: {art_str}")

                        # Inject stage summary into main agent's history
                        self._inject_subtask_result_to_history(
                            f"[Stage {stage_idx}/{total}] {sub.id}: OK — {result.summary[:300]}"
                        )

            except KeyboardInterrupt:
                interrupted_name = subtasks[stage_idx - 1].id if 0 < stage_idx <= len(subtasks) else "?"
                print(display.yellow(
                    f"\n  ⚠ Stage {stage_idx}/{total} ({interrupted_name}) interrupted by user."
                ))
                print(display.yellow("  Progress saved. Continue later with /plan resume."))
                self._current_stage_id = None
                return

            # Final summary
            final_summary = f"All {total} stages completed."
            self._inject_subtask_result_to_history(
                f"[Orchestrator] {final_summary} Artifacts: {json.dumps({k: str(v)[:100] for k, v in upstream.items()}, ensure_ascii=False)}"
            )
            self._current_stage_id = None
            return

        # ── Batch mode: parallel execution ──
        if mode == "batch":
            batch_tasks = route.get("batch_tasks", [])
            if len(batch_tasks) < 2:
                print(display.red("\n[Orchestrator] Batch mode requires at least 2 tasks."))
                return

            # Keep user input in main agent's history for context continuity
            self.history.append({"role": "user", "content": user_input})

            print(f"\n[Orchestrator] Running {len(batch_tasks)} experiments in parallel:")
            for i, t in enumerate(batch_tasks, 1):
                print(f"  Run {i}: {t[:80]}")

            results = o.run_batch_interactive(route, user_input)

            print()
            for i, r in enumerate(results, 1):
                icon = "✓" if r.status == "success" else "✗"
                print(f"[Run {i}] {icon} {r.status}: {r.summary[:150]}")

            # Inject summary into main agent's history
            summary_lines = ["[Batch comparison results]"]
            for i, r in enumerate(results, 1):
                summary_lines.append(
                    f"  Run {i}: {r.status} — {r.summary[:200]}"
                )
            self._inject_subtask_result_to_history("\n".join(summary_lines))
            return

        # ── Single mode: use existing ReAct loop ──
        if self.scene is None:
            self.scene = ScenePreset.auto_detect(user_input=user_input)

        if self.config.auto_skill:
            self._auto_load_skills(user_input)

        self._auto_turn_count = 0
        self._inject_context(user_input)
        self._check_user_porting_confirmation(user_input)
        self.history.append({"role": "user", "content": user_input})
        try:
            self._react_loop()
        except KeyboardInterrupt:
            display.interrupted()

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

    def _inject_subtask_result_to_history(self, summary: str):
        """Inject a structured subtask/batch result into the main agent's history.

        Injected as a user message prefixed with [system: task stage result] marker
        so the LLM sees upstream results while keeping turn semantics correct.
        """
        self.history.append({
            "role": "user",
            "content": f"[system: task stage result]\n{summary}"
        })

    def _run_single_shot(self, query: str):
        if self.scene is None:
            self.scene = ScenePreset.auto_detect(user_input=query)
        if self.config.auto_skill:
            self._auto_load_skills(query)
        self._inject_context(query)
        self.history.append({"role": "user", "content": query})
        try:
            self._react_loop()
        except Exception:
            logger.exception("WorkerAgent._run_single_shot() react loop failed")

    def execute(self, task: str) -> WorkerResult:
        """Non-interactive entry point for programmatic (Orchestrator) use.

        Runs the full ReAct loop for a single task and returns structured results.
        No PromptSession, no CLI interaction, no sys.exit() — safe for Embedder
        or BatchRunner to call.

        Returns WorkerResult with status, summary, artifacts, and token stats.
        """
        t0 = time.time()

        if self.scene is None:
            self.scene = ScenePreset.auto_detect(user_input=task)
        if self.config.auto_skill:
            self._auto_load_skills(task)

        self._inject_context(task)
        self._original_user_task = task
        self.history.append({"role": "user", "content": task})

        # ── Run loop with error guard ──
        loop_error: str | None = None
        try:
            self._react_loop()
        except Exception as e:
            logger.exception("WorkerAgent.execute() react loop failed: %s", e)
            loop_error = str(e)

        # ── Determine outcome ──
        last_text = self._get_last_assistant_text()
        task_complete = "[TASK_COMPLETE]" in last_text
        needs_user = "[NEED_USER_INPUT]" in last_text

        # Collect artifacts from session
        artifacts: dict[str, str] = {}
        active_plan = self.task_plan.get_active()
        if active_plan:
            done = sum(1 for s in active_plan.get("steps", [])
                       if s.get("status") in ("done", "skipped"))
            total = len(active_plan.get("steps", []))
            artifacts["plan_progress"] = f"{done}/{total}"
            artifacts["plan_id"] = active_plan.get("id", "")
            artifacts["plan_title"] = active_plan.get("title", "")

        experiments = self._experiment_manager.list()
        if experiments:
            artifacts["experiments"] = json.dumps(
                [{"name": e["name"], "status": e["status"]} for e in experiments[:5]]
            )

        # Determine status
        if loop_error:
            status = "failed"
            summary = f"ReAct loop crashed: {loop_error[:200]}"
        elif task_complete:
            status = "success"
            summary = last_text.replace("[TASK_COMPLETE]", "").strip()[:500] or "Task completed."
        elif needs_user:
            status = "partial"
            summary = last_text.replace("[NEED_USER_INPUT]", "").strip()[:500] or "Waiting for user input."
        elif not self.history.messages:
            status = "failed"
            summary = "No messages in history — provider or config issue."
        else:
            status = "partial"
            summary = last_text[:500] if last_text else "No final response."

        elapsed = time.time() - t0

        return WorkerResult(
            status=status,
            summary=summary,
            artifacts=artifacts,
            files_read=list(self._files_read_this_session),
            files_written=list(self._files_written_this_session),
            turn_count=self.turn_count,
            session_input_tokens=self._session_input_tokens,
            session_output_tokens=self._session_output_tokens,
            elapsed_seconds=elapsed,
        )

    # ── Session management ─────────────────────────────────────────────────

    def _restore_session(self, data: dict):
        messages = data.get("messages", [])
        # Skip the old system prompt — we already have a fresh one from __init__
        for msg in messages:
            if msg.get("role") == "system":
                continue
            self.history.append(msg)
        # Restore turn count from message history
        self.turn_count = sum(
            1 for m in messages
            if m.get("role") == "user" and isinstance(m.get("content", ""), str)
            and not m.get("content", "").startswith("[") and not m.get("content", "").startswith("<")
        )
        loaded = data.get("loaded_skills", [])
        skill_map = {}
        for skill_name in loaded:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    self._loaded_skills.add(skill_name)
                    self._active_skill_content[skill_name] = content
                    skill_map[skill_name] = content
            except Exception:
                pass
        if skill_map:
            self._batch_extract_and_rebuild(skill_map)
        # Refresh system prompt with restored context
        self._refresh_system_prompt()

    def _check_resume(self):
        sessions = find_resumable_sessions(self._sessions_root)
        if not sessions:
            return
        latest = sessions[0]
        if latest.get("user_turns", 0) >= 1:
            msg = latest.get("last_user_msg", "")
            print(display.yellow(f"\n[resume] Last session: {msg[:60]}..."))
            print(display.dim(f"Use /resume to restore ({len(sessions)} session(s) available)"))

    # ── Context injection ───────────────────────────────────────────────────

    def _build_memory_context(self) -> str:
        entries = self.session_memory.list_entries()
        if not entries:
            return ""
        lines = ["<context-memory>"]
        for e in entries[-10:]:
            key = e.get("key", "")
            mem_type = e.get("type", "")
            content = e.get("content", "")
            lines.append(f"<entry key=\"{key}\" type=\"{mem_type}\">{content[:300]}</entry>")
        lines.append("</context-memory>")
        return "\n".join(lines)

    def _inject_context(self, user_input: str):
        memory_context = self._build_memory_context()
        plan_context = self._build_plan_context()
        self._refresh_system_prompt(memory_context=memory_context, plan_context=plan_context)

    def _build_plan_context(self) -> str:
        active = self.task_plan.get_active()
        if not active:
            return ""
        steps = active.get("steps", [])
        icons = {"pending": "⬜", "doing": "🔄", "done": "✅", "skipped": "⏭", "blocked": "🚫"}
        lines = [f'<active-plan title="{active.get("title", "")}">']
        for s in steps:
            icon = icons.get(s.get("status", "pending"), "?")
            title = s.get("title", "") or s.get("description", "")
            notes = s.get("notes", "")
            line = f"  [{icon}] Step {s.get('id', '?')}: {title[:120]}"
            if notes:
                line += f" — {notes[:80]}"
            lines.append(line)
        lines.append("</active-plan>")
        return "\n".join(lines)

    # ── Startup ─────────────────────────────────────────────────────────────

    def _startup_hints(self) -> list[str]:
        hints = []
        sessions = find_resumable_sessions(self._sessions_root)
        if sessions:
            hints.append(f"{len(sessions)} resumable session(s) — use /resume to restore")
        return hints

    def _check_proxy(self):
        proxy_vars = [v for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY") if os.environ.get(v)]
        if proxy_vars:
            print(display.dim(f"Proxy detected: {', '.join(proxy_vars)}"))

    # ── Auto-continue ──────────────────────────────────────────────────────

    def _should_auto_continue(self) -> bool:
        if self._auto_turn_count >= self.config.max_auto_turns:
            return False
        if self._interrupted:
            return False
        last_text = self._get_last_assistant_text()
        if "[TASK_COMPLETE]" in last_text or "[NEED_USER_INPUT]" in last_text:
            return False
        active_plan = self.task_plan.get_active()
        result = self.judge.complexity(last_text[:500], has_plan=active_plan is not None)
        if result.get("needs_plan"):
            return False
        return True

    def _get_last_assistant_text(self) -> str:
        for msg in reversed(self.history.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    def _generate_continuation_prompt(self) -> str:
        plan_context = self._build_plan_context()
        base = (
            "[SYSTEM] Continue working on the task. If you've completed the task, "
            "respond with [TASK_COMPLETE]. If you need user input, respond with [NEED_USER_INPUT]."
        )
        if plan_context:
            return base + f"\n\nCurrent plan:\n{plan_context}"
        return base

    # ── Auto-skill loading ─────────────────────────────────────────────────

    def _register_skill_guards(self, skill_name: str):
        """Auto-register constraints and warnings from a Skill's frontmatter.

        Called after a skill is loaded. Extracts structured constraints/warnings
        from the Skill's YAML frontmatter and registers them with the Guard system.
        This is separate from the LLM-based constraint extraction (_extract_skill_constraints)
        which analyzes prose content. Idempotent — skips if already registered.
        """
        if not hasattr(self, '_skill_guards_registered'):
            self._skill_guards_registered = set()
        if skill_name in self._skill_guards_registered:
            return
        self._skill_guards_registered.add(skill_name)

        # Register hard constraints from frontmatter
        try:
            constraints = self.skill_manager.get_constraints(skill_name)
            if constraints:
                self._constraint_guard.add_constraints(constraints)
                logger.info("Registered %d frontmatter constraints from skill '%s'",
                            len(constraints), skill_name)
        except Exception as e:
            logger.debug("No frontmatter constraints for skill '%s': %s", skill_name, e)

        # Register soft warnings from frontmatter
        try:
            warnings = self.skill_manager.get_warnings(skill_name)
            if warnings:
                # Check if we already have a WarningGuard registered
                guard_registry = self._kernel.deps.guard_registry
                existing_warning_guard = None
                for g in guard_registry.guards:
                    if isinstance(g, WarningGuard):
                        existing_warning_guard = g
                        break
                if existing_warning_guard:
                    existing_warning_guard.add_warnings(warnings)
                else:
                    warning_guard = WarningGuard(warnings=warnings)
                    guard_registry.register(warning_guard)
                logger.info("Registered %d frontmatter warnings from skill '%s'",
                            len(warnings), skill_name)
        except Exception as e:
            logger.debug("No frontmatter warnings for skill '%s': %s", skill_name, e)

    def _auto_load_skills(self, user_input: str):
        """Load skills based on semantic judgment (primary) or keyword fallback.

        Uses Judge.suggest_skills() for semantic matching. Falls back to keyword
        matching only when Judge is unavailable.
        """
        skills = self.skill_manager.list_skills()
        available = [s for s in skills if s.get("name", "") not in self._loaded_skills]
        if not available:
            return

        loaded = set()

        # Primary: semantic suggestion via Judge
        if self.judge and self.judge.provider is not None:
            suggested = self.judge.suggest_skills(user_input, available)
            loaded = set(suggested)
        else:
            # Fallback: keyword matching (Judge unavailable)
            for s in available:
                keywords = s.get("keywords", [])
                name = s.get("name", "")
                if any(kw.lower() in user_input.lower() for kw in keywords):
                    loaded.add(name)

        for name in loaded:
            try:
                content = self.skill_manager.load(name)
                if content:
                    self._loaded_skills.add(name)
                    self._active_skill_content[name] = content
                    self._apply_skill_effects(name)
                    display.skill_auto_loaded(name)
                    self._register_skill_guards(name)
            except Exception:
                pass
        if loaded:
            skill_map = {n: self._active_skill_content.get(n, "") for n in loaded}
            self._batch_extract_and_rebuild(skill_map)

    def _apply_skill_effects(self, skill_name: str):
        """Apply effects declared in skill frontmatter — no hardcoded skill names."""
        effects = self.skill_manager.get_effects(skill_name)
        if not effects:
            return
        # Set mode flag
        mode = effects.get("mode")
        if mode:
            self.modes.set(mode)
        # Record initial_phase mapping
        initial_phase = effects.get("initial_phase")
        if mode and initial_phase:
            self._mode_phase_map[mode] = initial_phase
        # Auto-load companion skills
        companions = effects.get("companion_skills")
        if companions and isinstance(companions, list):
            self._auto_load_companion_skills(companions)

    def _auto_load_companion_skills(self, skill_names: list[str]):
        needs_refresh = False
        for name in skill_names:
            if name in self._loaded_skills:
                continue
            try:
                content = self.skill_manager.load(name)
                if content:
                    self._loaded_skills.add(name)
                    self._active_skill_content[name] = content
                    self._skill_load_iterations[name] = self._total_iterations
                    display.skill_auto_loaded(name)
                    self._register_skill_guards(name)
                    needs_refresh = True
            except Exception:
                pass
        if needs_refresh:
            skill_map = {
                n: self._active_skill_content.get(n, "")
                for n in skill_names if n in self._loaded_skills
            }
            self._batch_extract_and_rebuild(skill_map)

    # ── User path confirmation ────────────────────────────────────────────

    def _check_user_porting_confirmation(self, user_input: str):
        if self.modes.path_confirmed:
            return
        result = self.judge.classify("is_user_porting_confirm", {"user_input": user_input}, default="")
        if result == "mode_b":
            self.modes.path_confirmed = True
            self.modes.confirmed_path = "mode_b"
        elif result == "mode_c":
            self.modes.path_confirmed = True
            self.modes.confirmed_path = "mode_c"

    # ── React loop ──────────────────────────────────────────────────────────

    def _react_loop(self):
        """Kernel-based react loop."""
        self.turn_count += 1
        self._interrupted = False
        self._turn_iteration_count = 0
        self.judge.budget._exhausted_warned = False

        result = self._kernel.run_turn()

        self._interrupted = result.interrupted
        self._session_input_tokens += result.input_tokens
        self._session_output_tokens += result.output_tokens
        self._budget_guard.report_tokens(result.input_tokens, result.output_tokens)
        self._turn_iteration_count = result.iterations
        display.turn_summary(self.turn_count, result.iterations, result.input_tokens, result.output_tokens)

    def _on_kernel_response(self, response: dict):
        """Called by Kernel after LLM response is appended to history."""
        # Gate override detection
        if response.get("content"):
            for m in GATE_OVERRIDE_RE.finditer(response["content"]):
                gate_name = m.group(1).upper()
                reason = m.group(2).strip()
                if gate_name in (
                    "MODE_B_DESIGN_INTEGRITY", "MEGATRON_NATIVE_INTEGRITY",
                    "COMPONENT_MAPPING", "MIGRATION_BLUEPRINT",
                ) and self.judge.classify("is_frozen_excuse", {"reason": reason, "gate_name": gate_name}, default=False):
                    logger.warning("Gate override REJECTED for %s — 'frozen' is not a valid reason", gate_name)
                    continue

        # Checklist override detection
        if response.get("content") and self.checklist:
            for m in CHECKLIST_OVERRIDE_RE.finditer(response["content"]):
                cid = m.group(1)
                removed = self.checklist.override(cid)
                if removed:
                    print(display.green(f"  🛡 [{cid}] dismissed"))
                else:
                    logger.debug("Checklist override for unknown id: %s", cid)

    def _on_kernel_tool_results(self, tool_calls: list, results: list):
        """Called by Kernel after tool execution and guard checks."""
        # Track tool calls for phase management
        for tc in tool_calls:
            self._last_tool_calls_deque.append(tc["name"])
            phase_tools = PHASE_TOOL_SETS.get(self.phase)
            if phase_tools is not None and tc["name"] not in (phase_tools | CORE_TOOLS):
                self._extra_tools_next_iter.add(tc["name"])
        self._total_iterations += 1

        # Refresh system prompt if skill/plan tools were used
        if any(tc["name"] in ("load_skill", "plan_create", "plan_update", "plan_status")
               for tc in tool_calls):
            self._refresh_system_prompt()

        # Checklist checks (per tool)
        if self.checklist:
            for tc, result in zip(tool_calls, results):
                obs = self._build_obs(
                    tool_name=tc.get("name", ""),
                    tool_args=tc.get("arguments", {}),
                    tool_result=result,
                )
                self._inject_runtime_facts(self.checklist.engine)
                for alert in self.checklist.check(obs):
                    if alert.severity == "error":
                        print(display.red(f"  🛡 {alert.message}"))
                    elif alert.severity == "info":
                        print(display.dim(f"  🛡 {alert.message}"))
                    else:
                        print(display.yellow(f"  🛡 {alert.message}"))
                    self._inject_message(alert.message)

        # Judge budget exhaustion warning
        if self.judge.budget.exhausted and self.judge.budget.skipped_detail:
            if not self.judge.budget._exhausted_warned:
                self.judge.budget._exhausted_warned = True
                print(display.yellow(
                    f"\n[⚠ JUDGE BUDGET EXHAUSTED] {self.judge.budget.calls_this_turn}/"
                    f"{self.judge.budget.max_calls_per_turn} calls used. "
                    f"Skipped: {self.judge.budget.skipped_detail}"
                ))

        # Context pressure warning
        pressure = self.history.get_context_pressure()
        if pressure >= 0.75:
            warning = (
                f"⚠ Context at {int(pressure*100)}%. Save key findings with memory_write. "
                "Batch independent tool calls to reduce turn count."
            )
            msgs = self.history.messages
            if msgs and msgs[-1].get("role") == "user":
                last = msgs[-1]
                content = last.get("content", "")
                if isinstance(content, list):
                    content.append({"type": "text", "text": warning})
                elif isinstance(content, str):
                    last["content"] = content + "\n\n" + warning
            else:
                self.history.append({"role": "user", "content": warning})

        self._tool_call_cache = {}
        print()

    def _inject_message(self, msg: str):
        self.history.append({"role": "user", "content": msg})

    # ── Phase tracking ─────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        """Derive tool-availability phase from runtime context.

        Replaces the old mutable self.phase string. Phase determines which
        tools are available via PHASE_TOOL_SETS.
        """
        if self._phase_override:
            return self._phase_override
        # If runtime is active (training running / inference serving), verification mode
        runtime_active = any(
            isinstance(g, TrainingRuntimeGuard) and g._training_started
            for g in self._kernel.deps.guard_registry.guards
        )
        if runtime_active:
            return "verification"
        if self._code_written:
            return "implementation"
        # Check mode→phase mapping from loaded skill effects
        for mode, phase in self._mode_phase_map.items():
            if self.modes.has(mode):
                return phase
        return "idle"

    @phase.setter
    def phase(self, value: str):
        """Allow explicit phase override (for tests and backward compat)."""
        self._phase_override = value if value != "idle" else None

    def _get_filtered_schemas(self, phase: str) -> list[dict]:
        phase_tools = PHASE_TOOL_SETS.get(phase, set())
        tool_names = CORE_TOOLS | phase_tools | self._extra_tools_next_iter
        return self.tool_registry.to_schemas_filtered(
            self.provider.schema_format, tool_names
        )

    # ── LLM streaming ──────────────────────────────────────────────────────

    def _call_llm_stream(self, messages, schemas):
        content_parts = []
        tool_calls = []
        tool_calls_by_id = {}
        current_tool = None
        stream_truncated = False
        usage = {}
        self._streaming_in_code_block = False

        pressure = self.history.get_context_pressure()
        if pressure >= 0.85:
            logger.warning("Context pressure %.0f%%, forcing compaction before LLM call", pressure * 100)
            self.history.force_compact(target_ratio=0.50)
            if self.history._last_compacted_from:
                display.context_compacted(
                    self.history._last_compacted_from,
                    self.history._last_compacted_to,
                    compaction_num=self.history.compaction_count,
                    ratio=self.history.last_compaction_ratio,
                )
                self.history._last_compacted_from = None
                self.history._last_compacted_to = None
            messages = self.history.get_messages()

        _overflow_attempts = [0]
        _OVERFLOW_RATIOS = [0.50, 0.35, 0.25]

        def _handle_context_overflow():
            attempt = _overflow_attempts[0]
            if attempt >= len(_OVERFLOW_RATIOS):
                logger.error("Context overflow: all compaction ratios exhausted")
                return False
            ratio = _OVERFLOW_RATIOS[attempt]
            _overflow_attempts[0] += 1
            overflow_limit = self.history._actual_input_tokens or self.config.max_context_tokens
            logger.warning("Context overflow recovery (attempt %d): compacting to ratio %.2f of %d",
                           attempt + 1, ratio, overflow_limit)
            compacted = self.history.force_compact(target_ratio=ratio, base_limit=overflow_limit)
            if compacted:
                if self.history._last_compacted_from:
                    display.context_compacted(
                        self.history._last_compacted_from,
                        self.history._last_compacted_to,
                        compaction_num=self.history.compaction_count,
                        ratio=self.history.last_compaction_ratio,
                    )
                    self.history._last_compacted_from = None
                    self.history._last_compacted_to = None
                messages[:] = self.history.get_messages()
            return compacted

        stream = retry_with_backoff(
            lambda: self.provider.chat_stream(messages, schemas),
            max_retries=3,
            on_context_overflow=_handle_context_overflow,
        )

        thinking_cleared = False
        streaming_trailing_newlines = 0
        streaming_started = False

        def compress_newlines(text, trailing_from_prev, is_first):
            if not text:
                return text, trailing_from_prev
            if is_first:
                text = text.lstrip('\n')
                if not text:
                    return text, 0
            if trailing_from_prev > 0:
                leading = 0
                for ch in text:
                    if ch == '\n':
                        leading += 1
                    else:
                        break
                total_trailing = trailing_from_prev + leading
                if total_trailing > 2:
                    text = '\n\n' + text[leading:]
            new_trailing = 0
            for ch in reversed(text):
                if ch == '\n':
                    new_trailing += 1
                else:
                    break
            if new_trailing > 2:
                text = text[:len(text) - new_trailing + 2]
                new_trailing = 2
            return text, new_trailing

        max_stream_retries = 2
        for _stream_attempt in range(1 + max_stream_retries):
            try:
                for event in stream:
                    if not thinking_cleared:
                        display.thinking_done()
                        thinking_cleared = True
                    if event["type"] == "text":
                        text = event["content"]
                        text, streaming_trailing_newlines = compress_newlines(
                            text, streaming_trailing_newlines, not streaming_started)
                        if text:
                            streaming_started = True
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
                        # Clear thinking spinner on first tool call
                        if not thinking_cleared:
                            display.thinking_clear()
                            thinking_cleared = True
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
                break
            except KeyboardInterrupt:
                if not thinking_cleared:
                    display.thinking_clear()
                raise
            except Exception as e:
                if not thinking_cleared:
                    display.thinking_clear()
                    thinking_cleared = True
                if content_parts or tool_calls:
                    logger.warning("Stream interrupted after partial content: %s", e)
                    stream_truncated = True
                    break
                if _is_context_limit_error(e):
                    logger.warning("Context limit error during stream, compacting: %s", e)
                    display.warn("Context too large, compacting...")
                    if _handle_context_overflow():
                        stream = retry_with_backoff(
                            lambda: self.provider.chat_stream(messages, schemas),
                            max_retries=3,
                            on_context_overflow=_handle_context_overflow,
                        )
                        continue
                    else:
                        raise
                if _stream_attempt < max_stream_retries:
                    wait = 2 ** _stream_attempt
                    logger.warning("Stream interrupted (attempt %d/%d), retrying in %ds: %s",
                                   _stream_attempt + 1, max_stream_retries + 1, wait, e)
                    display.warn(f"Stream interrupted, retrying in {wait}s...")
                    time.sleep(wait)
                    stream = retry_with_backoff(
                        lambda: self.provider.chat_stream(messages, schemas),
                        max_retries=3,
                        on_context_overflow=_handle_context_overflow,
                    )
                    continue
                raise

        if content_parts:
            if streaming_trailing_newlines > 1 and display._use_color():
                up = streaming_trailing_newlines - 1
                display._write(f"\033[{up}A\033[J")
                print()
            elif streaming_trailing_newlines == 0:
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

        return {"content": "".join(content_parts) or None, "tool_calls": parsed_tool_calls, "truncated": stream_truncated}, usage

    # ── Tool execution (delegated to ToolExecutor) ──────────────────────────

    def _execute_tools(self, tool_calls):
        return self._tool_executor.execute_batch(tool_calls)

    def _execute_tool(self, tool_call, skip_confirm=False):
        return self._tool_executor.execute_single(tool_call, skip_confirm=skip_confirm)

    def _append_tool_results(self, tool_results: list[dict]):
        for tr in tool_results:
            self.history.append(tr)

    @staticmethod
    def _tool_display_summary(tool_name: str, arguments: dict) -> str:
        return tool_display_summary(tool_name, arguments)

    @staticmethod
    def _is_context_limit_error(e) -> bool:
        return _is_context_limit_error(e)

    # ── Compaction helpers ─────────────────────────────────────────────────

    def _summarize_for_compaction(self, text: str) -> str:
        response = self.provider.chat(
            [{"role": "user", "content": f"Summarize this conversation segment for an AI agent that will continue working on the same task. Keep under 1500 tokens. Include: file paths, error messages, decisions, current approach.\n\n{text}"}],
            tools=[]
        )
        return response.get("content", "")

    def _score_messages_for_compaction(self, messages: list[dict]) -> list[int]:
        return [5] * len(messages)

    def _summarize_file_content(self, content: str, path: str) -> str:
        lines = content.splitlines()
        if len(lines) <= 100:
            return content
        head = "\n".join(lines[:30])
        mid = "\n".join(lines[len(lines)//2 - 10:len(lines)//2 + 20])
        tail = "\n".join(lines[-30:])
        return f"{head}\n\n[... {len(lines) - 60} lines omitted from {path} ...]\n\n{mid}\n\n[...]\n\n{tail}"


    @staticmethod
    def _is_quick_test_command(cmd: str) -> bool:
        return bool(re.search(r'--train-iters\s+', cmd))
