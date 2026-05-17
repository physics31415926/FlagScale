"""FlagScale Agent — ReAct loop with composable Interrupt/Checklist/Judge architecture.

No Mixin inheritance. State is owned by Interrupt instances.
Scene + Profile parameterize behavior without subclassing.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shlex
import signal
import sys
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle

from flagscale.agent.react import display
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.history import HistoryManager, COMPACTION_NOTICE
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

from flagscale.agent.react.prompt import (
    SYSTEM_PROMPT_CORE, SYSTEM_PROMPT_OPTIONAL, SYSTEM_PROMPT,
)

from flagscale.agent.react.interrupt.base import Interrupt, Observation, Intervention
from flagscale.agent.react.interrupt.safety import SafetyInterrupt
from flagscale.agent.react.interrupt.loop_detect import LoopDetectInterrupt
from flagscale.agent.react.interrupt.progress import ProgressInterrupt
from flagscale.agent.react.interrupt.context_pressure import ContextPressureInterrupt
from flagscale.agent.react.interrupt.plan import PlanInterrupt
from flagscale.agent.react.interrupt.training_runtime import TrainingRuntimeInterrupt

from flagscale.agent.react.checklist.base import ChecklistEngine, ChecklistItem, Checklist
from flagscale.agent.react.judge import Judge, JudgeBudget
from flagscale.agent.react.scene import ScenePreset, PRESETS
from flagscale.agent.react.profile import WorkerProfile, PROFILES

logger = logging.getLogger(__name__)

# ── Registered interrupts ────────────────────────────────────────────────────

_REGISTERED_INTERRUPTS: list[type[Interrupt]] = [
    SafetyInterrupt,
    LoopDetectInterrupt,
    ProgressInterrupt,
    ContextPressureInterrupt,
    PlanInterrupt,
    TrainingRuntimeInterrupt,
]

# ── Phase tool sets (from v1, preserved) ────────────────────────────────────

_READ_ONLY_TOOLS = {
    "read_file", "grep", "find", "ls", "list_files",
    "memory_read", "memory_list", "plan_status", "web_fetch",
}

_PRODUCTIVE_TOOLS = {
    "write_file", "edit_file", "shell", "memory_write",
    "plan_create", "plan_update", "workspace_experiment",
}

_CORE_TOOLS = {
    "read_file", "write_file", "edit_file", "shell",
    "load_skill", "web_fetch", "memory_write", "memory_read",
    "memory_list", "monitor", "workspace_experiment",
    "plan_create", "plan_update", "plan_status",
}

_PHASE_TOOL_SETS = {
    "idle": {
        "read_file", "shell", "load_skill", "memory_read", "memory_list",
        "web_fetch", "workspace_experiment", "find_latest_log",
        "plan_create", "plan_status", "memory_write", "write_file",
        "edit_file", "monitor", "validate_config",
    },
    "analysis": {
        "read_file", "shell", "memory_read", "memory_list",
        "web_fetch", "load_skill", "workspace_experiment",
        "find_latest_log", "memory_write",
        "plan_create", "plan_update", "plan_status",
        "write_file", "edit_file", "inspect_checkpoint",
        "validate_config",
    },
    "implementation": {
        "read_file", "write_file", "edit_file", "shell",
        "load_skill", "memory_write", "memory_read",
        "plan_update", "plan_status", "workspace_experiment",
        "find_latest_log", "monitor", "validate_config",
        "inspect_checkpoint", "parse_training_metrics",
    },
    "verification": {
        "read_file", "shell", "write_file", "edit_file",
        "monitor", "find_latest_log", "parse_training_metrics",
        "memory_write", "memory_read", "workspace_experiment",
        "plan_update", "plan_status", "load_skill",
        "inspect_checkpoint", "validate_config",
    },
}

_SHELL_READ_RE = re.compile(
    r'\s*(grep|find|cat|ls|head|tail|wc|file|stat|which|type|'
    r'echo|pwd|env|printenv|hostname|uname|date|id|whoami|ps|pgrep)\b'
)

_TRAIN_LAUNCH_RE = re.compile(
    r'flagscale\s+train|torchrun|deepspeed|python.*pretrain|'
    r'python.*train(?:ing)?_', re.IGNORECASE
)

_PIPELINE_KNOWLEDGE_KEYWORDS = [
    "model architecture", "component mapping", "pipeline",
    "forward pass", "backward pass", "loss function",
    "data flow", "checkpoint", "Megatron", "FlagScale",
]

_MIN_PIPELINE_KEYWORDS_IN_MEMORY = 3

_VERIFICATION_STAGES = [
    "none", "analysis", "init_ok", "forward_aligned",
    "backward_ok", "distributed_ok", "full_training",
]

_FROZEN_EXCUSE_PATTERNS = re.compile(
    r'frozen|no.?grad|requires_grad.*False|'
    r'feature.extractor|no.trainable|'
    r'zero.trainable|no.gradient|'
    r'not.trained|inference.only|'
    r'no.TP.benefit.*frozen|'
    r'frozen.*no.need|'
    r'frozen.*skip|'
    r'doesn.t.need.*native.*frozen',
    re.IGNORECASE
)

_PORTING_COMPANION_SKILLS = [
    "parallel-strategy",
    "precision-alignment",
    "train-config",
    "train-run",
    "env-setup",
    "workspace-layout",
    "data-prep",
]

_KNOWLEDGE_CONFIRM_RE = re.compile(
    r'\[PIPELINE_KNOWLEDGE_CONFIRMED:\s*(YES|NO)\]', re.IGNORECASE
)

_GATE_OVERRIDE_RE = re.compile(
    r'\[GATE_OVERRIDE:\s*([A-Z_]+)\]\s*Reason:\s*(.+?)(?:\n|$)',
    re.IGNORECASE
)

_READ_FILE_SUMMARY_THRESHOLD = 8000
_READ_FILE_SUMMARY_THRESHOLD_PORTING = 15000


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
    """Consolidated boolean/string flags that parameterize agent behavior.

    Replaces 8 scattered self._xxx_mode flags in WorkerAgent.
    """

    porting: bool = False
    data_prep: bool = False
    training_started: bool = False
    env_setup_loaded: bool = False
    env_compat_analyzed: bool = False
    workspace_layout_loaded: bool = False
    porting_path_confirmed: bool = False
    confirmed_porting_path: str | None = None


# ── WorkerAgent ──────────────────────────────────────────────────────────────

class WorkerAgent:
    """Single agent class with composable Interrupt/Checklist/Judge architecture.

    No Mixin inheritance. State that belongs to Interrupts is owned
    by Interrupt instances. All infrastructure is composed via __init__.
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

        # ── Composed components (v3 architecture) ──
        self.judge = Judge(self.provider, budget=JudgeBudget(max_calls_per_turn=3))
        self.interrupts: list[Interrupt] = self._build_interrupts()
        self.checklist: Checklist | None = self._build_checklist()

        self._init_runtime_state()
        atexit.register(self._atexit_hook)

    def _init_runtime_state(self):
        """Initialize mutable per-session state. Called from __init__.

        Extracted to keep __init__ focused on dependency wiring.
        Can be re-called for tests or worker resets.
        """
        self.phase: str = "idle"
        self.turn_count: int = 0
        self._interrupted: bool = False
        self._last_tool_calls_deque = deque(maxlen=5)
        self._extra_tools_next_iter: set[str] = set()
        self._turn_iteration_count: int = 0
        self._consecutive_single_tool_calls: int = 0
        self._loaded_skills: set[str] = set()
        self._active_skill_content: dict[str, str] = {}
        self._skill_load_iterations: dict[str, int] = {}
        self._total_iterations: int = 0
        self._recently_referenced_skills: set[str] = set()
        self.modes = _ModeFlags()
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

        if self.session_memory:
            for entry in self.session_memory.list_entries():
                content = entry.get("content", "").lower()
                kw_hits = sum(1 for kw in _PIPELINE_KNOWLEDGE_KEYWORDS if kw.lower() in content)
                if kw_hits >= _MIN_PIPELINE_KEYWORDS_IN_MEMORY:
                    break

        self._refresh_system_prompt()

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
        self.tool_registry.register(MonitorTool(regex_judge_fn=self._regex_judge_confirm))
        self.tool_registry.register(WorkspaceExperimentTool(self._experiment_manager, task_plan=self.task_plan))
        self.tool_registry.register(ValidateConfigTool())
        self.tool_registry.register(InspectCheckpointTool())

    def _build_interrupts(self) -> list[Interrupt]:
        constraints = self.scene.constraints if self.scene else set()
        interrupts = []
        for cls in _REGISTERED_INTERRUPTS:
            if "always" in cls.activate_on or cls.activate_on & constraints:
                interrupts.append(cls())
        return sorted(interrupts, key=lambda i: i.priority)

    def _build_checklist(self) -> Checklist | None:
        engine = ChecklistEngine()
        items = []

        constraints = self.scene.constraints if self.scene else set()

        # ── Scene-default rules (always loaded) ──

        # Training: verify launch prerequisites
        if "is_training" in constraints:
            items.append(ChecklistItem(
                id="train_config_exists",
                description="Verify training config path exists before launch",
                phases={"implementation", "verification"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "No such file or directory", "mode": "contains"},
                    {"match": "file not found", "mode": "contains"},
                ],
                reminder="Config file not found. Verify the path with read_file or install the model-porter skill for config generation.",
                severity="error",
            ))
            items.append(ChecklistItem(
                id="train_gpu_visible",
                description="Verify GPU devices are visible before launching training",
                phases={"implementation", "verification"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "CUDA_VISIBLE_DEVICES", "mode": "not_contains"},
                ],
                reminder="GPU visibility not confirmed. Run nvidia-smi or set CUDA_VISIBLE_DEVICES before launch.",
                severity="warning",
            ))
            items.append(ChecklistItem(
                id="train_output_dir_writable",
                description="Verify output directory is writable before launching training",
                phases={"implementation", "verification"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "Permission denied", "mode": "contains"},
                    {"match": "Read-only file system", "mode": "contains"},
                ],
                reminder="Output directory is not writable. Check permissions or choose a different --save path.",
                severity="error",
            ))

        # Migration: verify weight mapping and precision
        if "is_migration" in constraints:
            items.append(ChecklistItem(
                id="migration_weight_map_complete",
                description="Verify weight mapping covers all keys",
                phases={"implementation", "verification"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "missed", "mode": "contains"},
                    {"match": "unexpected key", "mode": "contains"},
                    {"match": "skipped", "mode": "contains"},
                ],
                reminder="Weight mapping has missed/unexpected keys. Review the FULL list — not just the count of differences. Copy the complete output, audit each key, identify the source layer for each mismatch.",
                severity="error",
                max_reminders=5,
            ))
            items.append(ChecklistItem(
                id="migration_precision_align",
                description="Verify precision alignment between source and target",
                phases={"implementation"},
                trigger_on={"tool": "write_file"},
                content_rules=[
                    {"match": "fp16", "mode": "regex"},
                ],
                reminder="Source model uses fp16 but target defaults to bf16. Check precision alignment: loss scale, gradient scaler, mixed precision config.",
                severity="warning",
            ))
            items.append(ChecklistItem(
                id="migration_forward_verify_first",
                description="Verify forward pass before opening distributed gates",
                phases={"implementation", "verification"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "nccl error", "mode": "contains"},
                    {"match": "nccl", "mode": "contains"},
                    {"match": "process group", "mode": "contains"},
                ],
                reminder="Distributed error detected. Run single-GPU forward verification first: set tp=1, dp=1, pp=1 and verify loss matches reference before opening distributed gates.",
                severity="error",
            ))

        # Inference: verify serving setup
        if "is_inference" in constraints:
            items.append(ChecklistItem(
                id="inference_model_exists",
                description="Verify model checkpoint exists before starting serving",
                phases={"implementation"},
                trigger_on={"tool": "shell"},
                result_rules=[
                    {"match": "No such file or directory", "mode": "contains"},
                ],
                reminder="Model checkpoint path does not exist. Verify the path exists and contains weight files.",
                severity="error",
            ))

        # Domestic chip migration
        if "is_chip_migration" in constraints:
            items.append(ChecklistItem(
                id="chip_cuda_only_op",
                description="Check for CUDA-only operators in migrated code",
                phases={"analysis", "implementation"},
                trigger_on={"tool": "read_file", "path_match": "*.py"},
                content_rules=[
                    {"match": "flash_attn", "mode": "contains"},
                    {"match": "cuda_ext", "mode": "contains"},
                    {"match": "torch.cuda", "mode": "contains"},
                ],
                reminder="CUDA-specific code found. Check if alternative exists for target chip (e.g. ascend flash_attn replacement, torch_npu equivalents).",
                severity="warning",
            ))
            items.append(ChecklistItem(
                id="chip_precision_bf16_fp16",
                description="Check bf16 usage on non-NVIDIA chips",
                phases={"analysis", "implementation"},
                trigger_on={"tool": "write_file"},
                content_rules=[
                    {"match": "bf16", "mode": "regex"},
                ],
                reminder="bf16 may not be natively supported on domestic chips — many fall back to fp16 internally. Verify actual precision behavior on target chip.",
                severity="warning",
            ))

        # DDP training: auto-detect and enforce process group cleanup
        items.append(ChecklistItem(
            id="ddp_process_group_cleanup",
            description="Verify destroy_process_group after training crash",
            phases={"implementation", "verification"},
            trigger_on={"tool": "shell"},
            result_rules=[
                {"match": "process group.*already", "mode": "regex"},
                {"match": "tcpStore.*already", "mode": "regex"},
            ],
            reminder="Process group leak detected. Kill zombie processes with pgrep -a python | grep pretrain | awk '{print $1}' | xargs kill -9, then destroy process group in code.",
            severity="error",
        ))

        # ── Skill-based constraints (opt-in, supplements defaults) ──
        if self.skill_manager:
            try:
                meta = self.skill_manager.get_meta("model-porter")
                if meta and meta.get("constraints"):
                    for c in meta["constraints"]:
                        items.append(ChecklistItem(
                            id=c["id"],
                            description=c.get("description", c["id"]),
                            phases=set(c.get("phases", ["*"])),
                            trigger_on=c.get("trigger_on"),
                            content_rules=c.get("content_rules", []),
                            result_rules=c.get("result_rules", []),
                            reminder=c.get("reminder", ""),
                            severity=c.get("severity", "warning"),
                            max_reminders=c.get("max_reminders", 3),
                        ))
            except Exception:
                pass

        if not items:
            return None
        return Checklist(engine=engine, items=items)

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
        skills_summary = self._build_skills_summary()
        cwd = os.getcwd()
        core = SYSTEM_PROMPT_CORE.format(skills=skills_summary, cwd=cwd)
        optional = SYSTEM_PROMPT_OPTIONAL.format(memory_context=memory_context, plan_context=plan_context)
        full = core + "\n" + optional

        if self._active_skill_content:
            skill_bodies = []
            for name, content in self._active_skill_content.items():
                skill_bodies.append(content)
            full = full.replace("</system-prompt>", "")
            full += "\n\n" + "\n\n".join(skill_bodies)
            full += "\n</system-prompt>"

        # Apply info-density booster if available
        try:
            full = SYSTEM_PROMPT.boost(full)
        except Exception:
            pass

        self.history.set_system_prompt(full)

    def _build_skills_summary(self) -> str:
        try:
            available = self.skill_manager.list_skills()
            lines = []
            for s in available:
                name = s.get("name", "")
                desc = s.get("description", "")
                kws = s.get("keywords", [])
                if kws:
                    lines.append(f"- {name}: {desc} (keywords: {', '.join(kws[:5])})")
                else:
                    lines.append(f"- {name}: {desc}")
            return "\n".join(lines)
        except Exception:
            return "(skills not available)"

    # ── Observation builder ─────────────────────────────────────────────────

    def _build_obs(
        self,
        tool_name: str = "",
        tool_args: dict | None = None,
        tool_result: str | None = None,
    ) -> Observation:
        return Observation(
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_result=tool_result,
            turn_count=self.turn_count,
            phase_name=self.phase,
            recent_tool_names=[t[0] for t in getattr(self, '_recent_tool_calls', [])[-10:]],
            context_pressure=self.history.get_context_pressure() if self.history else 0.0,
            experiment_compare_fn=self._experiment_manager.compare if self._experiment_manager else None,
            experiment_diff_fn=self._experiment_manager.diff_last_attempts if self._experiment_manager else None,
            current_experiment_name=self._experiment_manager.get_current_experiment() if self._experiment_manager else "",
        )

    # ── Health judge (delegates to unified Judge) ───────────────────────────

    def _health_judge(self, command: str, recent_output: str, elapsed: str,
                      output_changed: bool = True, stall_count: int = 0) -> dict:
        return self.judge.health(command, recent_output, elapsed, output_changed, stall_count)

    def _regex_judge_confirm(self, category: str, matched_text: str, context: str = "") -> bool:
        return self.judge.regex_confirm(category, matched_text, context)

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

            if self._handle_slash_command(user_input):
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

    def _run_single_shot(self, query: str):
        if self.scene is None:
            self.scene = ScenePreset.auto_detect(user_input=query)
        if self.config.auto_skill:
            self._auto_load_skills(query)
        self._inject_context(query)
        self.history.append({"role": "user", "content": query})
        self._react_loop()

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

    # ── Slash commands ──────────────────────────────────────────────────────

    def _handle_slash_command(self, user_input: str) -> bool:
        cmd = user_input.split()[0] if user_input.startswith("/") else None
        if not cmd:
            return False

        if cmd == "/quit":
            self._exit()
            return True
        elif cmd == "/reload":
            self.config.reload()
            self._refresh_system_prompt()
            print("Config and skills reloaded.")
            return True
        elif cmd == "/skill":
            self._handle_skill_command(user_input)
            return True
        elif cmd == "/file":
            self._handle_file_command(user_input)
            return True
        elif cmd == "/save":
            self._save_conversation(completed=False)
            print("Conversation saved.")
            return True
        elif cmd == "/load":
            self._handle_load_command(user_input)
            return True
        elif cmd == "/export":
            self._handle_export_command(user_input)
            return True
        elif cmd == "/memory":
            self._handle_memory_command(user_input)
            return True
        elif cmd == "/mode":
            self._handle_mode_command(user_input)
            return True
        elif cmd == "/plan":
            self._handle_plan_command(user_input)
            return True
        elif cmd == "/resume":
            self._handle_resume_command(user_input)
            return True
        elif cmd == "/compact":
            self.history.force_compact(target_ratio=0.50)
            print("History compacted.")
            return True

        return False

    def _handle_skill_command(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            skills = self.skill_manager.list_skills()
            print("Available skills:")
            for s in skills:
                print(f"  {s['name']}: {s['description'][:60]}")
            return
        name = parts[1]
        try:
            self.skill_manager.load(name)
            print(f"Skill '{name}' loaded.")
        except FileNotFoundError:
            print(f"Skill '{name}' not found.")

    def _handle_file_command(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /file <path>")
            return
        path = parts[1]
        if os.path.isfile(path):
            result = self.tool_registry.execute("read_file", path=path)
            print(result[:2000])
        else:
            print(f"File not found: {path}")

    def _handle_load_command(self, user_input: str):
        parts = user_input.split()
        sessions = find_resumable_sessions(self._sessions_root)
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(sessions):
                s = sessions[idx]
                data = load_conversation(s["session_dir"])
                if data:
                    self._restore_session(data)
                    print(f"Loaded session: {s.get('last_user_msg', '')[:60]}")
                    return
        if not sessions:
            print("No resumable sessions found.")
            return
        print("Resumable sessions:")
        for i, s in enumerate(sessions[:10], 1):
            print(f"  {i}. [{time.strftime('%m-%d %H:%M', time.localtime(s['timestamp']))}] {s.get('last_user_msg', '')[:60]}")
        print("Usage: /load <number>")

    def _handle_export_command(self, user_input: str):
        path = os.path.join(self._session_dir, "conversation.json")
        print(f"Conversation exported to: {path}")

    def _handle_memory_command(self, user_input: str):
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
                print(f"  [{mem_type}] {key}: {content[:80]}")
        else:
            print(f"Unknown /memory subcommand: {sub}")

    def _handle_mode_command(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            print(f"Current mode: {self.config.mode}")
            print("Available modes: confirm, auto")
            return
        mode = parts[1]
        if mode in ("confirm", "auto"):
            self.config.mode = mode
            if mode == "auto":
                self.config.confirm_commands = False
                self.config.max_iterations = 2**31 - 1
                # Re-register shell tool without confirm
                self.tool_registry._tools.pop("shell", None)
                self.tool_registry.register(
                    ShellTool(
                        remind_interval=self.config.shell_remind_interval,
                        check_dangerous=self.config.dangerous_commands_check,
                        require_confirm=False,
                        env=self.config.shell_env,
                        health_judge_fn=self._health_judge,
                    )
                )
            print(f"Mode set to: {mode}")
        else:
            print(f"Unknown mode: {mode}")

    def _handle_plan_command(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            active = self.task_plan.get_active()
            if active:
                print(f"Active plan: {active.get('id', '?')}")
                for step in active.get("steps", []):
                    icon = {"pending": " ", "doing": "→", "done": "✓", "skipped": "-", "blocked": "!"}.get(step.get("status", "pending"), " ")
                    print(f"  [{icon}] {step.get('description', '?')[:80]}")
            else:
                print("No active plan.")
            return
        print(f"Unknown /plan subcommand: {' '.join(parts[1:])}")

    def _handle_resume_command(self, user_input: str):
        sessions = find_resumable_sessions(self._sessions_root)
        if not sessions:
            print("No resumable sessions found.")
            return
        parts = user_input.split()
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(sessions):
                s = sessions[idx]
                data = load_conversation(s["session_dir"])
                if data:
                    self._restore_session(data)
                    print(f"Resumed session: {s.get('last_user_msg', '')[:60]}")
                    return
        for i, s in enumerate(sessions[:10], 1):
            print(f"  {i}. [{time.strftime('%m-%d %H:%M', time.localtime(s['timestamp']))}] {s.get('last_user_msg', '')[:60]}")
        print("Usage: /resume <number>")

    def _restore_session(self, data: dict):
        messages = data.get("messages", [])
        for msg in messages:
            self.history.append(msg)
        loaded = data.get("loaded_skills", [])
        for skill_name in loaded:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    self._loaded_skills.add(skill_name)
                    self._active_skill_content[skill_name] = content
            except Exception:
                pass
        self._refresh_system_prompt()

    def _check_resume(self):
        sessions = find_resumable_sessions()
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
        lines = ["<active-plan>"]
        for s in steps:
            status = s.get("status", "pending")
            desc = s.get("description", "")[:100]
            lines.append(f"  [{status}] {desc}")
        lines.append("</active-plan>")
        return "\n".join(lines)

    # ── Startup ─────────────────────────────────────────────────────────────

    def _startup_hints(self) -> list[str]:
        hints = []
        sessions = find_resumable_sessions()
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
        return (
            "[SYSTEM] Continue working on the task. If you've completed the task, "
            "respond with [TASK_COMPLETE]. If you need user input, respond with [NEED_USER_INPUT]."
        )

    # ── Auto-skill loading ─────────────────────────────────────────────────

    def _auto_load_skills(self, user_input: str):
        skills = self.skill_manager.list_skills()
        loaded = set()
        for s in skills:
            keywords = s.get("keywords", [])
            name = s.get("name", "")
            if name in self._loaded_skills:
                continue
            if any(kw.lower() in user_input.lower() for kw in keywords):
                loaded.add(name)
        for name in loaded:
            try:
                content = self.skill_manager.load(name)
                if content:
                    self._loaded_skills.add(name)
                    self._active_skill_content[name] = content
                    if "model-porter" in name:
                        self.modes.porting = True
                    if "data-prep" in name:
                        self.modes.data_prep = True
                    if "env-setup" in name:
                        self.modes.env_setup_loaded = True
                    if "workspace-layout" in name:
                        self.modes.workspace_layout_loaded = True
                    display.skill_auto_loaded(name)
            except Exception:
                pass
        if loaded:
            self._refresh_system_prompt()

    def _auto_load_companion_skills(self, skill_names: list[str]):
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
            except Exception:
                pass

    # ── User porting confirmation ───────────────────────────────────────────

    def _check_user_porting_confirmation(self, user_input: str):
        if self.modes.porting_path_confirmed:
            return
        user_lower = user_input.lower()
        if re.search(r'mode.?b|模式.?b|megatron.?native|原生', user_lower):
            self.modes.porting_path_confirmed = True
            self.modes.confirmed_porting_path = "mode_b"
        elif re.search(r'mode.?c|模式.?c|wrapper|包装', user_lower):
            self.modes.porting_path_confirmed = True
            self.modes.confirmed_porting_path = "mode_c"

    # ── React loop ──────────────────────────────────────────────────────────

    def _react_loop(self):
        self.turn_count += 1
        self._interrupted = False
        self._turn_iteration_count = 0
        turn_input_tokens = 0
        turn_output_tokens = 0
        max_iter = self.config.max_iterations
        iteration = 0

        _prev_handler = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            if self._interrupted:
                signal.signal(signal.SIGINT, _prev_handler)
                raise KeyboardInterrupt
            self._interrupted = True
            display.interrupted()

        signal.signal(signal.SIGINT, _sigint_handler)

        while iteration < max_iter:
            if self._interrupted:
                break

            # Reset per-turn interrupt + judge state
            for intr in self.interrupts:
                intr.reset_turn()
            self.judge.reset_turn()

            # Phase-based schema filtering
            self._check_phase()
            schemas = self._get_filtered_schemas(self.phase)
            self._extra_tools_next_iter = set()

            t0 = time.time()
            messages = self.history.get_messages()

            # Show compaction notice if needed
            if self.history._last_compacted_from:
                display.context_compacted(
                    self.history._last_compacted_from,
                    self.history._last_compacted_to,
                    compaction_num=self.history.compaction_count,
                    ratio=self.history.last_compaction_ratio,
                )
                self.history._last_compacted_from = None
                self.history._last_compacted_to = None

            # ── Pre-turn: Interrupt checks ──
            obs = self._build_obs()
            blocked = False
            for intr in self.interrupts:
                intervention = intr.check_pre(obs)
                if intervention and intervention.action == "block":
                    self._inject_message(intervention.message)
                    blocked = True
                    break
            if blocked:
                continue

            # ── Checklist reminders ──
            if self.checklist:
                reminders = self.checklist.check(obs)
                for msg in reminders:
                    self._inject_message(msg)

            # ── LLM call ──
            display.thinking()

            try:
                response, usage = self._call_llm_stream(messages, schemas)
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break
            except Exception as e:
                if self._is_context_limit_error(e):
                    display.thinking_clear()
                    logger.warning("Context limit hit, forcing compact and retry: %s", e)
                    print(display.yellow("⚠ Context limit hit — compacting and retrying..."))
                    recovered = False
                    for _ratio in [0.50, 0.35, 0.25]:
                        overflow_limit = self.history._actual_input_tokens or self.config.max_context_tokens
                        self.history.force_compact(target_ratio=_ratio, base_limit=overflow_limit)
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
                        try:
                            display.thinking()
                            response, usage = self._call_llm_stream(messages, schemas)
                            recovered = True
                            break
                        except Exception as e2:
                            display.thinking_clear()
                            if self._is_context_limit_error(e2):
                                continue
                            print(display.red(f"✖ LLM error after compact: {e2}"))
                            break
                    if not recovered:
                        print(display.red("✖ Context still too large after aggressive compaction"))
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

            if self._interrupted:
                break

            logger.info("LLM call #%d: %.1fs", iteration + 1, elapsed)
            self.history.append(self.provider.format_assistant_message(response))

            # Gate override detection
            if response.get("content"):
                for m in _GATE_OVERRIDE_RE.finditer(response["content"]):
                    gate_name = m.group(1).upper()
                    reason = m.group(2).strip()
                    if gate_name in (
                        "MODE_B_DESIGN_INTEGRITY", "MEGATRON_NATIVE_INTEGRITY",
                        "COMPONENT_MAPPING", "MIGRATION_BLUEPRINT",
                    ) and _FROZEN_EXCUSE_PATTERNS.search(reason):
                        logger.warning("Gate override REJECTED for %s — 'frozen' is not a valid reason", gate_name)
                        continue

            if not response["tool_calls"]:
                break

            print()

            # ── Execute tools ──
            tool_t0 = time.time()
            try:
                results = self._execute_tools(response["tool_calls"])
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break

            tool_results = [
                self.provider.format_tool_result(tc["id"], result)
                for tc, result in zip(response["tool_calls"], results)
            ]
            self._append_tool_results(tool_results)

            # Track tool calls
            for tc in response["tool_calls"]:
                self._last_tool_calls_deque.append(tc["name"])
                phase_tools = _PHASE_TOOL_SETS.get(self.phase)
                if phase_tools is not None and tc["name"] not in (phase_tools | _CORE_TOOLS):
                    self._extra_tools_next_iter.add(tc["name"])
            self._total_iterations += 1
            self._turn_iteration_count += 1

            if any(tc["name"] == "load_skill" for tc in response["tool_calls"]):
                self._refresh_system_prompt()

            # ── Post-exec: Interrupt + Checklist checks ──
            for tc, result in zip(response["tool_calls"], results):
                obs = self._build_obs(
                    tool_name=tc.get("name", ""),
                    tool_args=tc.get("arguments", {}),
                    tool_result=result,
                )
                for intr in self.interrupts:
                    intervention = intr.check_post(obs)
                    if intervention:
                        if intervention.action == "inject_msg":
                            self._inject_message(intervention.message)
                        elif intervention.action == "force_compact":
                            self.history.force_compact(target_ratio=0.50)
                        elif intervention.action == "escalate":
                            print(display.red(f"\n[ESCALATED] {intervention.message}\n"))
                            break

                if self.checklist:
                    for msg in self.checklist.check(obs):
                        self._inject_message(msg)

            print()

            # Context pressure check
            pressure = self.history.get_context_pressure()
            if pressure >= 0.95:
                self.history.force_compact(target_ratio=0.50)
            elif pressure >= 0.75:
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
            iteration += 1

        signal.signal(signal.SIGINT, _prev_handler)

    def _inject_message(self, msg: str):
        self.history.append({"role": "user", "content": msg})

    # ── Phase tracking ─────────────────────────────────────────────────────

    def _check_phase(self):
        if self.modes.training_started:
            if self.phase != "verification":
                self.phase = "verification"
        elif self._code_written:
            if self.phase != "implementation":
                self.phase = "implementation"
        elif self.modes.porting:
            if self.phase == "idle":
                self.phase = "analysis"

    def _get_filtered_schemas(self, phase: str) -> list[dict]:
        phase_tools = _PHASE_TOOL_SETS.get(phase, set())
        tool_names = _CORE_TOOLS | phase_tools | self._extra_tools_next_iter
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

    # ── Tool execution ─────────────────────────────────────────────────────

    def _execute_tools(self, tool_calls):
        if len(tool_calls) == 1:
            result = self._execute_tool(tool_calls[0])
            tool_name = tool_calls[0]["name"]
            if tool_name in _READ_ONLY_TOOLS:
                self._consecutive_single_tool_calls += 1
                if 2 <= self._consecutive_single_tool_calls <= 4:
                    result += (
                        "\n\n[EFFICIENCY REMINDER: You have made "
                        f"{self._consecutive_single_tool_calls} consecutive single-tool responses. "
                        "Batch independent tool calls in ONE response to reduce round-trips.]"
                    )
            else:
                self._consecutive_single_tool_calls = 0
            return [result]

        self._consecutive_single_tool_calls = 0

        # Dedup
        seen_calls = {}
        dedup_indices = set()
        for i, tc in enumerate(tool_calls):
            key = (tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True))
            if key in seen_calls:
                dedup_indices.add(i)
            else:
                seen_calls[key] = i

        _MAX_BATCH = 20
        capped_indices = set()
        if len(tool_calls) > _MAX_BATCH:
            logger.warning("Batch has %d tool calls, capping to %d", len(tool_calls), _MAX_BATCH)
            for i in range(_MAX_BATCH, len(tool_calls)):
                capped_indices.add(i)

        # Pre-exec: Interrupt checks
        skip_indices: set[int] = set()
        for i, tc in enumerate(tool_calls):
            obs = self._build_obs(
                tool_name=tc["name"],
                tool_args=tc.get("arguments", {}),
            )
            for intr in self.interrupts:
                intervention = intr.check_pre(obs)
                if intervention and intervention.action == "block":
                    skip_indices.add(i)
                    results_tmp = [""] * len(tool_calls)
                    results_tmp[i] = f"⛔ TOOL NOT EXECUTED — blocked by {intr.name}: {intervention.message}"
                    break

        # Pre-confirm shell commands
        shell_tool = self.tool_registry.get("shell")
        denied = set()
        if shell_tool:
            for i, tc in enumerate(tool_calls):
                if i in skip_indices:
                    continue
                if tc["name"] == "shell":
                    cmd = tc["arguments"].get("command", "")
                    if shell_tool.needs_confirm(cmd):
                        if not shell_tool.pre_confirm(cmd):
                            denied.add(i)

        results = [None] * len(tool_calls)
        skip_indices |= denied | dedup_indices | capped_indices

        # Serialize non-read shell commands
        write_shell_indices = []
        for i, tc in enumerate(tool_calls):
            if i in skip_indices:
                continue
            if tc["name"] == "shell":
                cmd = tc["arguments"].get("command", "")
                if not bool(_SHELL_READ_RE.match(cmd)):
                    write_shell_indices.append(i)
        if len(write_shell_indices) > 1:
            for idx in write_shell_indices[1:]:
                skip_indices.add(idx)
                results[idx] = (
                    "[PARALLEL WRITE BLOCK — COMMAND NOT EXECUTED]\n\n"
                    "Non-read shell commands cannot run in parallel. "
                    "Issue them sequentially in separate responses.\n"
                )

        for i in denied:
            results[i] = "DENIED: User declined to execute this command."
        for i in dedup_indices:
            orig = seen_calls[(tool_calls[i]["name"], json.dumps(tool_calls[i].get("arguments", {}), sort_keys=True))]
            results[i] = f"[DEDUP: identical to call #{orig + 1} in this batch, skipped]"
        for i in capped_indices:
            results[i] = f"[BATCH CAPPED — TOOL NOT EXECUTED] Only {_MAX_BATCH} tool calls allowed per response."

        to_run = [(i, tc) for i, tc in enumerate(tool_calls) if i not in skip_indices]

        idx_to_line = {orig_i: line_i for line_i, (orig_i, _) in enumerate(to_run)}

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

        if not to_run:
            display.parallel_tools_finish()
            return results

        with ThreadPoolExecutor(max_workers=min(len(to_run), 4)) as pool:
            futures = {pool.submit(_run_quiet, i, tc): i for i, tc in to_run}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        display.parallel_tools_finish()
        return results

    def _execute_tool(self, tool_call, skip_confirm=False):
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # Read cache
        cached_key = (tool_name, json.dumps(arguments, sort_keys=True))
        if cached_key in self._tool_call_cache:
            logger.info("Cache hit for %s, skipping execution", tool_name)
            return self._tool_call_cache[cached_key] + "\n[Cached result from earlier in this turn]"

        t0 = time.time()
        try:
            if skip_confirm and tool_name == "shell":
                result = self.tool_registry.execute(tool_name, _skip_confirm=True, **arguments)
            else:
                result = self.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            result = f"ERROR: {e}"
        elapsed = time.time() - t0

        logger.info("Tool %s: %.1fs, result %d chars", tool_name, elapsed, len(result))
        error = "ERROR" in result[:20] if result else False

        # Track file reads
        if tool_name == "read_file" and not error:
            path = arguments.get("path", "")
            if path:
                self._files_read_this_session.add(path)
                if len(path) > (_READ_FILE_SUMMARY_THRESHOLD_PORTING if self.modes.porting else _READ_FILE_SUMMARY_THRESHOLD):
                    result = self._summarize_file_content(result, path)

        # Track writes
        if tool_name in ("write_file", "edit_file") and not error:
            self._last_write_turn = self.turn_count
            self._code_written = True
            path = arguments.get("path", "") or arguments.get("file_path", "")
            if path:
                self._files_written_this_session.add(path)

        # Track load_skill side effects
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if skill_name not in self._loaded_skills:
                self._loaded_skills.add(skill_name)
                # Extract content from result
                skill_content = result
                prefix_end = result.find("\n\n")
                if prefix_end != -1 and result.startswith("SUCCESS:"):
                    skill_content = result[prefix_end + 2:]
                self._active_skill_content[skill_name] = skill_content
                self._skill_load_iterations[skill_name] = self._total_iterations
                if "model-porter" in skill_name:
                    self.modes.porting = True
                    self._auto_load_companion_skills(_PORTING_COMPANION_SKILLS)
                if "data-prep" in skill_name:
                    self.modes.data_prep = True
                if "env-setup" in skill_name:
                    self.modes.env_setup_loaded = True
                if "workspace-layout" in skill_name:
                    self.modes.workspace_layout_loaded = True
                result = f"[Skill '{skill_name}' loaded — content available in system context]"

        # Cache for this turn
        self._tool_call_cache[cached_key] = result
        return result

    def _append_tool_results(self, tool_results: list[dict]):
        for tr in tool_results:
            self.history.append(tr)

    @staticmethod
    def _shell_display_summary(cmd: str, max_len: int = 90) -> str:
        s = cmd.replace("\n", " ").replace("\r", "").strip()
        if len(s) > max_len:
            s = s[:max_len - 3] + "..."
        return s

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

    # ── FlagScale helpers ──────────────────────────────────────────────────

    @staticmethod
    def _is_flagscale_dryrun(cmd: str) -> bool:
        return bool(re.search(r'flagscale.*--dryrun', cmd, re.I))

    @staticmethod
    def _is_training_launch(cmd: str) -> bool:
        return bool(_TRAIN_LAUNCH_RE.search(cmd))

    @staticmethod
    def _is_quick_test_command(cmd: str) -> bool:
        return bool(re.search(r'--train-iters\s+', cmd))
