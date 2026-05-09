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
    PromptMixin, SYSTEM_PROMPT_CORE, SYSTEM_PROMPT_OPTIONAL, SYSTEM_PROMPT,
    _is_tool_result_msg,
)
from flagscale.agent.react.compact import CompactionMixin
from flagscale.agent.react.gates import GatesMixin
from flagscale.agent.react.judges import JudgesMixin
from flagscale.agent.react.commands import CommandsMixin
from flagscale.agent.react.loop_detect import LoopDetectMixin
from flagscale.agent.react.checkpoint import CheckpointMixin
from flagscale.agent.react.poll import PollMixin
from flagscale.agent.react.skill_lifecycle import SkillLifecycleMixin

logger = logging.getLogger(__name__)


class ReactAgent(PromptMixin, CompactionMixin, GatesMixin, JudgesMixin, CommandsMixin,
                 LoopDetectMixin, CheckpointMixin, PollMixin, SkillLifecycleMixin):
    """A ReAct agent with streaming, history management, and parallel tool execution."""

    _READ_FILE_SUMMARY_THRESHOLD = 8000
    _READ_FILE_SUMMARY_THRESHOLD_PORTING = 15000

    _KNOWLEDGE_CONFIRM_RE = re.compile(
        r'\[PIPELINE_KNOWLEDGE_CONFIRMED:\s*(YES|NO)\]', re.IGNORECASE
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
        self.tool_registry.register(MonitorTool(regex_judge_fn=self._regex_judge_confirm))

        # Session-centric workspace: each session gets its own directory
        self._session_id = uuid.uuid4().hex[:8]
        sessions_root = config.session_dir or os.path.join(Path.home(), ".flagscale", "sessions")
        session_dir = os.path.join(sessions_root, self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        self._sessions_root = sessions_root

        # Experiment manager (per-session)
        experiments_dir = os.path.join(session_dir, "experiments")
        self._experiment_manager = ExperimentManager(experiments_dir)
        self._load_plugin_tools()

        memory_dir = os.path.join(Path.home(), ".flagscale", "agent_memory")
        self.session_memory = SessionMemory(memory_dir, config.memory_ttl_days)
        self.tool_registry.register(MemoryWriteTool(self.session_memory, self._session_id))
        self.tool_registry.register(MemoryReadTool(self.session_memory))
        self.tool_registry.register(MemoryListTool(self.session_memory))

        plan_dir = os.path.join(session_dir, "plans")
        self.task_plan = TaskPlan(plan_dir)
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        self.tool_registry.register(PlanUpdateTool(self.task_plan))
        self.tool_registry.register(PlanStatusTool(self.task_plan))

        # Register after task_plan exists so experiment can auto-link
        self.tool_registry.register(WorkspaceExperimentTool(self._experiment_manager, task_plan=self.task_plan))
        self.tool_registry.register(ValidateConfigTool())
        self.tool_registry.register(InspectCheckpointTool())

        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = get_provider(config.provider, config.model, config.api_key, config.base_url, config.max_output_tokens)

        # Inject LLM capability into session memory for dedup and semantic expansion
        self.session_memory._llm_fn = lambda prompt: self.provider.chat(
            [{"role": "user", "content": prompt}], tools=[]
        ).get("content", "")

        self.history = HistoryManager(max_context_tokens=config.max_context_tokens)
        self.history.set_summarizer(self._summarize_for_compaction)
        self.history.set_scorer(self._score_messages_for_compaction)

        self._turn_count = 0
        self._original_user_task = ""
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
        self._recovery_from_failures = 0  # Set when training succeeds after failures
        self._kill_retry_timestamps = []  # Track kill+relaunch cycles
        self._training_launch_timestamps = []  # Track all training launches for hang detection
        self._awaiting_monitor = False  # Rule: must call monitor after real training launch (not dryrun)
        self._monitor_gate_block_count = 0
        self._context_pressure_soft_warned = False
        self._context_pressure_hard_warned = False
        self._last_checkpoint_tokens = 0  # For progress checkpoint
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
        self._source_reads_since_last_failure = 0  # Source-reading gate: framework reads after failure
        self._porting_mode = False  # Reading depth gate: True after model-porter skill loaded
        self._init_regex_judge()  # LLM-assisted regex confirmation
        self._porting_path_confirmed = False  # Porting path gate: True after user confirms Mode B/C
        self._confirmed_porting_path = None  # Which mode user confirmed: "mode_b" or "mode_c"
        self._data_prep_mode = False  # Data pipeline gate: True after data-prep skill loaded
        self._data_pipeline_understood = False  # Data pipeline gate: True after pipeline comprehension persisted
        self._analysis_persisted = False  # Analysis persistence gate: True after analysis written
        self._pipeline_knowledge_persisted = False  # Pipeline comprehension gate: True after knowledge persisted
        self._pipeline_knowledge_confirmed = False  # Pipeline comprehension gate: True after LLM confirms
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
        self._code_written = False  # Track if agent has written code (for phase transitions)
        # New gate state (Phase 1-4)
        self._understanding_verified = False  # A1: True after verification questions answered
        self._component_plan_created = False  # A3: True after component isolation plan created
        self._structure_completeness_verified = False  # U1: True after model structure enumeration and verification
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
        self._consecutive_single_tool_calls = 0  # Track single-call responses for batch nudge

        # Token optimization: skill lifecycle management
        self._active_skill_content = {}  # {skill_name: content_text}
        self._skill_load_iterations = {}  # {skill_name: iteration_when_loaded}
        self._total_iterations = 0
        self._training_started = False
        self._recently_referenced_skills = set()

        # Pre-check: if pipeline knowledge already exists in memory, skip gate phases
        if self.session_memory:
            entry = self.session_memory.get(self._PIPELINE_KNOWLEDGE_MEMORY_KEY)
            if entry:
                content = entry.get("content", "").lower()
                kw_hits = sum(1 for kw in self._PIPELINE_KNOWLEDGE_KEYWORDS if kw in content)
                if kw_hits >= self._MIN_PIPELINE_KEYWORDS_IN_MEMORY:
                    self._pipeline_knowledge_persisted = True
                    self._pipeline_knowledge_confirmed = True

        # Now refresh system prompt after all state is initialized
        self._refresh_system_prompt()
        atexit.register(self._atexit_hook)

    # ── Atexit safety net ───────────────────────────────────────────────

    def _atexit_hook(self):
        """Save conversation on any exit path (safety net for abnormal exits)."""
        try:
            if self._session_output_tokens:
                self._save_conversation()
        except Exception:
            pass

    def _save_conversation(self, completed=False):
        """Save conversation state to conversation.json (periodic + exit)."""
        try:
            msgs = [m for m in self.history.messages if m.get("role") != "system"]
            save_conversation(
                self._session_dir, self._session_id, msgs,
                loaded_skills=list(self._active_skill_content.keys()),
                metadata={"provider": self.config.provider, "model": self.config.model},
                completed=completed,
            )
        except Exception:
            pass

    def _check_resume(self):
        """Check for resumable sessions on startup."""
        sessions = find_resumable_sessions(self._sessions_root)
        if not sessions:
            return
        latest = sessions[0]
        last_user = latest.get("last_user_msg", "")
        timestamp = latest.get("timestamp", 0)
        turn_count = latest.get("user_turns", 0)
        display.resume_found(latest["session_id"], last_user, timestamp)
        try:
            answer = input("Resume? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("", "y", "yes"):
            old_dir = latest["session_dir"]
            conv = load_conversation(old_dir)
            if conv:
                msgs = conv.get("messages", [])
                loaded_skills = conv.get("loaded_skills", [])
                self.history._messages = [self.history.messages[0]] if self.history.messages and self.history.messages[0].get("role") == "system" else []
                self.history._messages.extend(msgs)
                self.history._full_log = list(msgs)
                for skill_name in loaded_skills:
                    try:
                        content = self.skill_manager.load(skill_name)
                        if content:
                            self._active_skill_content[skill_name] = content
                            self._loaded_skills.add(skill_name)
                    except Exception:
                        pass
                mark_completed(old_dir)
                old_startup_dir = self._session_dir
                self._session_dir = old_dir
                self._session_id = latest["session_id"]
                if old_startup_dir != old_dir and os.path.isdir(old_startup_dir):
                    try:
                        if not os.listdir(old_startup_dir):
                            os.rmdir(old_startup_dir)
                    except OSError:
                        pass
                # Repoint all session-dir-dependent state
                self.task_plan._dir = os.path.join(old_dir, "plans")
                self._experiment_manager._dir = os.path.join(old_dir, "experiments")
                plan_create_tool = self.tool_registry.get("plan_create")
                if plan_create_tool:
                    plan_create_tool._session_id = latest["session_id"]
                memory_write_tool = self.tool_registry.get("memory_write")
                if memory_write_tool:
                    memory_write_tool._session_id = latest["session_id"]
                display.session_resumed(latest["session_id"])

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
        # Dynamic budget: reduce memory injection when context is tight
        pressure = self.history.get_context_pressure()
        if pressure > 0.7:
            budget = 1000
        elif pressure > 0.5:
            budget = 2000
        else:
            budget = 4000
        task = getattr(self, '_original_user_task', '')
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


    def _inject_context(self, user_input):
        """Auto-inject session memory, plan, and workspace context into the system prompt."""
        if not self._original_user_task:
            self._original_user_task = user_input[:300]

        memory_context = self._build_memory_context()

        plan_context = self.task_plan.context_for_prompt()

        # Plan consistency check (every 5 turns when plan exists)
        if plan_context and self._turn_count > 0 and self._turn_count % 5 == 0:
            consistency_msg = self.task_plan.check_consistency(self._turn_count)
            if consistency_msg:
                plan_context += consistency_msg

        # Plan rebuild suggestion on consecutive failures
        if self._consecutive_train_failures >= 3 and self.task_plan.should_rebuild(self._consecutive_train_failures):
            plan_context += (
                "\n\n⚠️ [PLAN REBUILD SUGGESTED] "
                f"{self._consecutive_train_failures} consecutive training failures on the current step. "
                "Your current plan may be based on wrong assumptions. Consider:\n"
                "1. plan_update(action='abandon') to discard the failing plan\n"
                "2. Analyze the root cause of repeated failures\n"
                "3. plan_create with a fundamentally different approach"
            )

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

        self._refresh_system_prompt(memory_context=memory_context, plan_context=plan_context)

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
        self._check_resume()

        history_file = os.path.join(os.path.expanduser("~"), ".flagscale", "input_history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        completer = WordCompleter(
            ["/quit", "/reload", "/skill", "/file", "/save", "/load", "/export", "/memory", "/mode", "/plan", "/resume", "/compact"],
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
            elif cmd == "/resume":
                self._handle_resume_command(user_input)
                continue
            elif cmd == "/compact":
                self._handle_compact_command(user_input)
                continue

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
                parts.append(f"Current step: {doing[0].get('title', '')[:80]}")
            elif pending:
                parts.append(f"Next step: {pending[0].get('title', '')[:80]}")
                # Check if multiple pending steps are independent (heuristic: first 3 pending)
                if len(pending) >= 2:
                    parts.append("BATCH: multiple pending steps — issue independent tool calls together")

        # Include current experiment status if relevant
        exp_name = self._experiment_manager.get_current_experiment()
        if exp_name:
            exp = self._experiment_manager.read(exp_name)
            if exp and exp.get("status") == "running":
                attempts = exp.get("attempts", [])
                failed_attempts = [a for a in attempts if "fail" in str(a.get("result", "")).lower()
                                   or "error" in str(a.get("result", "")).lower()]
                if failed_attempts:
                    # Inject full failure history so LLM can avoid repeating
                    fail_lines = []
                    for i, a in enumerate(failed_attempts[-5:], 1):
                        change = a.get("change", "")[:60]
                        result = a.get("result", "")[:60]
                        fail_lines.append(f"  #{i} [FAILED] {change} → {result}")
                    parts.append("FAILED attempts (DO NOT REPEAT):\n" + "\n".join(fail_lines))
                elif attempts:
                    parts.append(f"Last attempt: {attempts[-1].get('result', '')[:80]}")

        # Include original task when no plan context available
        if not parts and self._original_user_task:
            parts.append(f"Task: {self._original_user_task[:100]}")

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
            return "继续。按照你的计划推进任务。Remember: batch independent tool calls in one response."
        return "Continue. Proceed with your plan. Remember: batch independent tool calls in one response."

    def _exit(self):
        atexit.unregister(self._atexit_hook)
        self._ensure_memory_written()
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



    # ── Autosave / Resume ───────────────────────────────────────────────






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
            if self._interrupted:
                # Second Ctrl+C — force raise to break out of blocking calls
                signal.signal(signal.SIGINT, _prev_handler)
                raise KeyboardInterrupt
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
                self.history._last_compacted_from = None
                self.history._last_compacted_to = None
                # Restore enforcement state from compaction summary

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

            # Phase 3: detect pipeline knowledge confirmation in LLM response text
            if not self._pipeline_knowledge_confirmed and response.get("content"):
                match = self._KNOWLEDGE_CONFIRM_RE.search(response["content"])
                if match and match.group(1).upper() == "YES":
                    self._pipeline_knowledge_confirmed = True
                    logger.info("Pipeline knowledge confirmed by LLM")

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

            # Intra-turn compaction: only when context pressure is genuinely high
            if iteration > 0 and self.history.get_context_pressure() > 0.70:
                if self.history.compact_intra_turn(keep_last=6):
                    logger.info("Intra-turn compaction at iteration %d (pressure=%.2f)",
                                iteration, self.history.get_context_pressure())

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
        self._save_conversation()

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
        elif pressure >= 0.75:
            logger.info("Context pressure %.0f%%, approaching limit — consider saving key findings", pressure * 100)

        def _handle_context_overflow():
            logger.warning("Context overflow recovery: forcing aggressive compaction")
            compacted = self.history.force_compact(target_ratio=0.50)
            if compacted:
                # Display compaction info before get_messages() resets it
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

        max_stream_retries = 2
        for _stream_attempt in range(1 + max_stream_retries):
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
                # Stream completed successfully
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
                    # Partial content received — return what we have
                    logger.warning("Stream interrupted after partial content: %s", e)
                    break
                # Context overflow — compact before retrying
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
                        logger.error("Context compaction failed, cannot retry")
                        raise
                # Nothing received — retry
                if _stream_attempt < max_stream_retries:
                    wait = 2 ** _stream_attempt
                    logger.warning(
                        "Stream interrupted with no content (attempt %d/%d), retrying in %ds: %s",
                        _stream_attempt + 1, max_stream_retries + 1, wait, e,
                    )
                    display.warn(f"Stream interrupted, retrying in {wait}s...")
                    time.sleep(wait)
                    stream = retry_with_backoff(
                        lambda: self.provider.chat_stream(messages, schemas),
                        max_retries=3,
                        on_context_overflow=_handle_context_overflow,
                    )
                    continue
                # All retries exhausted
                logger.error("Stream interrupted, all retries exhausted: %s", e)
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

    _SHELL_READ_RE = re.compile(
        r'(?:cat|head|tail|sed\s+-n|awk)\s+.*?(/\S+\.(?:py|yaml|yml|json|toml|cfg|md|txt|sh))'
        r'|(?:grep\s+.*?)\s(/\S+\.(?:py|yaml|yml|json|toml|cfg|md|txt|sh))'
    )

    def _track_shell_file_reads(self, cmd: str):
        """Extract file paths from shell read commands and count them for reading depth gate."""
        for match in self._SHELL_READ_RE.finditer(cmd):
            path = match.group(1) or match.group(2)
            if path and os.path.isabs(path):
                self._files_read_this_session.add(path)
                for cat, pattern in self._PORTING_READ_CATEGORIES.items():
                    if pattern.search(path):
                        self._reading_categories.add(cat)

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
            result = self._execute_tool(tool_calls[0])
            tool_name = tool_calls[0]["name"]
            if tool_name in self._READ_ONLY_TOOLS:
                self._consecutive_single_tool_calls += 1
                if 2 <= self._consecutive_single_tool_calls <= 4:
                    result += (
                        "\n\n[EFFICIENCY REMINDER: You have made "
                        f"{self._consecutive_single_tool_calls} consecutive single-tool responses. "
                        "Batch independent tool calls (read_file, shell, workspace_*) in ONE response "
                        "to reduce round-trips. Each round-trip re-sends ~12K tokens of context.]"
                    )
            else:
                self._consecutive_single_tool_calls = 0
            return [result]

        self._consecutive_single_tool_calls = 0

        # Dedup identical tool calls within batch
        seen_calls = {}
        dedup_indices = set()
        for i, tc in enumerate(tool_calls):
            key = (tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True))
            if key in seen_calls:
                dedup_indices.add(i)
            else:
                seen_calls[key] = i
        if dedup_indices:
            logger.info("Deduped %d identical tool calls in batch", len(dedup_indices))

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
                elif name == "shell":
                    # Shell commands are exploratory — only count toward staleness
                    # if the command looks like a repeat (same prefix as recent commands)
                    pass
                else:
                    self._reads_since_last_new_file += 1
            elif name in self._PRODUCTIVE_TOOLS:
                self._consecutive_reads = 0
                self._progress_gate_triggers = 0
                self._reads_since_last_new_file = 0
            # Count toward plan gate (complexity-judge mode)
            if self._complex_task_no_plan and name not in (
                "plan_create", "memory_write", "workspace_experiment"
            ):
                self._pre_plan_tool_calls += 1

        # Check plan gate after counting
        if not any(tc["name"] in self._PRODUCTIVE_TOOLS for tc in tool_calls):
            plan_block = self._check_plan_creation_gate(tool_calls[0]["name"])
            if plan_block and "TOOL NOT EXECUTED" in plan_block:
                display.warn("Plan gate: HARD BLOCK — parallel tools not executed")
                return [plan_block] * len(tool_calls)

        # Check progress gate (staleness-based) for parallel path
        stale_threshold = 30
        if self._porting_mode:
            stale_threshold = 60
        elif self._consecutive_train_failures >= 2:
            stale_threshold = 40
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
        skip_indices = denied | dedup_indices

        # Rule: non-read shell commands cannot run in parallel — serialize them
        write_shell_indices = []
        for i, tc in enumerate(tool_calls):
            if i in skip_indices:
                continue
            if tc["name"] == "shell":
                cmd = tc["arguments"].get("command", "")
                is_read_only = bool(re.match(
                    r'\s*(grep|find|cat|ls|head|tail|wc|file|stat|which|type|echo|pwd|env|printenv|hostname|uname|date|id|whoami|ps|pgrep)\b',
                    cmd
                ))
                if not is_read_only:
                    write_shell_indices.append(i)
        if len(write_shell_indices) > 1:
            for idx in write_shell_indices[1:]:
                skip_indices.add(idx)
                results[idx] = (
                    "[PARALLEL WRITE BLOCK — COMMAND NOT EXECUTED]\n\n"
                    "Non-read shell commands cannot run in parallel. "
                    "Issue them sequentially in separate responses.\n"
                    f"Blocked command: {tool_calls[idx]['arguments'].get('command', '')[:200]}\n"
                )

        # Run pre-execution gates on each tool call in the batch
        for i, tc in enumerate(tool_calls):
            if i in skip_indices:
                continue
            hard_block, soft_warnings = self._run_pre_execution_gates(tc["name"], tc.get("arguments", {}))
            if hard_block:
                skip_indices.add(i)
                results[i] = hard_block
                display.warn(f"Gate blocked parallel tool: {tc['name']}")

        to_run = [
            (i, tc) for i, tc in enumerate(tool_calls) if i not in skip_indices
        ]
        for i in denied:
            results[i] = "DENIED: User declined to execute this command."
        for i in dedup_indices:
            orig = seen_calls[(tool_calls[i]["name"], json.dumps(tool_calls[i].get("arguments", {}), sort_keys=True))]
            results[i] = f"[DEDUP: identical to call #{orig + 1} in this batch, skipped]"

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

        if not to_run:
            display.parallel_tools_finish()
            return results

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

        # Pre-execution gates — hard block if triggered
        hard_block, soft_warnings = self._run_pre_execution_gates(tool_name, arguments)
        if hard_block:
            # Extract gate name from message for display
            import re as _re
            gate_match = _re.search(r'\[([A-Z][A-Z_ ]+?)(?:\s*[—\-\]])', hard_block)
            gate_label = gate_match.group(1).strip() if gate_match else "GATE"
            display.warn(f"Gate blocked: {gate_label}")
            return hard_block

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
        # Monitor tool: detect failure from result reason
        if not error and tool_name == "monitor" and result:
            error = any(kw in result[:60] for kw in ("stderr_error", "process_dead", "error_detected"))

        # Track files read for reading depth gate
        if tool_name == "read_file" and not error:
            path = arguments.get("path", "")
            if path:
                was_already_read = path in self._files_read_this_session
                self._files_read_this_session.add(path)
                # Track framework source reads for source-reading gate
                if self._consecutive_train_failures >= 2:
                    _FRAMEWORK_INDICATORS = (
                        "megatron", "transformer_engine", "flagscale",
                        "Megatron-LM", "TransformerEngine",
                    )
                    if any(ind in path for ind in _FRAMEWORK_INDICATORS):
                        self._source_reads_since_last_failure += 1
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

        # Track shell file reads for reading depth gate
        if tool_name == "shell" and not error:
            cmd = arguments.get("command", "")
            self._track_shell_file_reads(cmd)

        # Track write operations for auto-continue stagnation detection
        if tool_name in ("write_file", "edit_file") and not error:
            self._last_write_turn = self._turn_count
            path = arguments.get("path", "") or arguments.get("file_path", "")
            if path:
                self._files_written_this_session.add(path)

        # Auto-sync plan step based on tool execution result
        if tool_name in self._PRODUCTIVE_TOOLS:
            summary = ""
            if tool_name in ("write_file", "edit_file"):
                summary = arguments.get("path", "") or arguments.get("file_path", "")
            self.task_plan.auto_sync_step(
                tool_name, success=not error,
                result_summary=summary[:100],
                turn=self._turn_count,
            )
        elif error and tool_name == "shell":
            cmd = arguments.get("command", "")[:60]
            self.task_plan.auto_sync_step(
                tool_name, success=False,
                result_summary=f"cmd failed: {cmd}",
                turn=self._turn_count,
            )

        # Reset re-read counter when findings are saved
        if tool_name in ("memory_write", "workspace_experiment") and not error:
            self._rereads_without_save = 0

        # Track porting mode activation
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if "model-porter" in skill_name:
                self._porting_mode = True
                self._auto_load_companion_skills(self._PORTING_COMPANION_SKILLS)
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

        # Track pipeline knowledge persistence (fixed key only)
        if tool_name == "memory_write" and not error:
            key = arguments.get("key", "")
            if key == self._PIPELINE_KNOWLEDGE_MEMORY_KEY:
                content = arguments.get("content", "").lower()
                kw_hits = sum(1 for kw in self._PIPELINE_KNOWLEDGE_KEYWORDS if kw in content)
                if kw_hits >= self._MIN_PIPELINE_KEYWORDS_IN_MEMORY:
                    self._pipeline_knowledge_persisted = True
                    logger.info("Pipeline knowledge persisted (%d/%d keywords)",
                                kw_hits, len(self._PIPELINE_KNOWLEDGE_KEYWORDS))

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

            # U1: Structure completeness — unlock when enumeration persisted
            if not self._structure_completeness_verified and tool_name in ("workspace_experiment", "memory_write"):
                struct_kws = ("component checklist", "structure enumeration", "all components",
                              "module tree", "total parameters", "porting checklist", "named_modules",
                              "param count", "parameter count")
                if sum(1 for kw in struct_kws if kw in content) >= 1:
                    self._structure_completeness_verified = True
                    logger.info("Structure completeness gate unlocked")

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

        detail = ""
        if tool_name == "shell":
            cmd = arguments.get("command", "")
            # Single LLM-confirmed training launch check (regex + judge)
            is_train_launch = bool(self._TRAIN_LAUNCH_RE.search(cmd))
            if is_train_launch:
                is_train_launch = self._regex_judge_confirm(
                    "is_training_launch", cmd,
                    result[:300] if result else "")

            # Track validation run success (NOT FlagScale dryrun — that only generates scripts)
            if (is_train_launch
                    and self._is_quick_test_command(cmd)
                    and not self._is_flagscale_dryrun(cmd)
                    and not error):
                self._dry_run_passed = True

            # Post-launch gates (command already executed at this point)
            if (is_train_launch
                    and not self._is_quick_test_command(cmd)):
                # Dry-run gate: warn if checkpoint-loading run without prior dry-run
                if (self._CHECKPOINT_LOAD_RE.search(cmd)
                        and not self._dry_run_passed):
                    result = self._DRY_RUN_WARNING + result
                    display.warn("Checkpoint-loading training launched without prior dry-run!")
                # Hydra cache warning: if config was edited, old hydra output may be stale
                hydra_warn = self._check_hydra_cache_stale(cmd)
                if hydra_warn:
                    result = hydra_warn + result
                self._dry_run_passed = False
                # Mark pending attempt as "running"
                exp_name = self._experiment_manager.get_current_experiment()
                if exp_name:
                    self._experiment_manager.update_last_attempt(exp_name, "running")
            # Remind to update experiment record when training fails
            if (is_train_launch
                    and not self._is_quick_test_command(cmd)
                    and error):
                result = result + self._EXPERIMENT_UPDATE_REMINDER
                # Checkpoint: auto-record training failure
                ckpt_warn = self._checkpoint_training_failure(cmd, result)
                if ckpt_warn:
                    result = result + ckpt_warn
                # Proactive memory recall: surface past fixes for similar errors
                recall = self._proactive_memory_recall(result)
                if recall:
                    result = result + recall
            # Remind to memorize learnings when training launches successfully
            if (is_train_launch
                    and not self._is_quick_test_command(cmd)
                    and not error):
                self._awaiting_monitor = True
                result = result + self._TRAINING_MEMORY_HINT
                # Inject stderr check hint for FlagScale-style launches
                result = result + self._POST_LAUNCH_STDERR_HINT
                # Checkpoint: auto-record training launch
                ckpt_warn = self._checkpoint_training_launch(cmd, result)
                if ckpt_warn:
                    result = result + ckpt_warn
            annotations = self._result_judge(cmd, result, elapsed)
            annotations = self._dedup_annotations(annotations)
            if annotations:
                header = "\n".join(f"[{a}]" for a in annotations)
                result = header + "\n" + result

        # Knowledge capture: training recovered after failures
        if self._recovery_from_failures > 0:
            result = result + self._KNOWLEDGE_CAPTURE_HINT.format(n=self._recovery_from_failures)
            self._recovery_from_failures = 0

        if error and result:
            raw = result.split('\n')[0].replace("ERROR:", "").strip()
            detail = (raw[:57] + "...") if len(raw) > 60 else raw

        # Workaround detection: same tool, previous call failed, this one succeeded
        if (self._last_tool_call is not None
                and not error
                and self._last_tool_call[0] == tool_name
                and self._last_tool_call[2]):
            result = result + self._WORKAROUND_MEMORY_HINT

        # Checkpoint: new unique error
        if error and tool_name == "shell":
            error_sig = self._extract_error_summary(result)
            ckpt_warn = self._checkpoint_new_error(error_sig, result)
            if ckpt_warn:
                result = result + ckpt_warn
            # Proactive recall for non-training shell errors too
            if not self._is_training_launch(arguments.get("command", "")):
                recall = self._proactive_memory_recall(result)
                if recall:
                    result = result + recall

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

        # Monitor detected training failure — auto-update experiment attempt
        if tool_name == "monitor" and error:
            error_summary = self._extract_error_summary(result)
            if error_summary:
                exp_name = self._experiment_manager.get_current_experiment()
                if exp_name:
                    try:
                        exp = self._experiment_manager.read(exp_name)
                        attempts = exp.get("attempts", []) if exp else []
                        last_result = str(attempts[-1].get("result", "")) if attempts else ""
                        # Don't overwrite a more specific FAILED message with a generic one
                        if "FAILED:" in last_result and "process_dead" in error_summary:
                            pass
                        else:
                            self._experiment_manager.update_last_attempt(
                                exp_name, f"FAILED: {error_summary}")
                    except Exception:
                        pass

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
            # Auto-clear monitor gate if diagnostic shows training process is gone
            if self._awaiting_monitor and re.match(
                r'\s*(pgrep|ps)\b', cmd
            ):
                result_lower = result.lower() if result else ""
                if not result.strip() or "no process" in result_lower:
                    self._awaiting_monitor = False
                    self._monitor_gate_block_count = 0
                    logger.info("Monitor gate auto-cleared: training process not found")

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

        # Inject soft warnings from pre-execution gates — deduplicated
        all_warning_parts = [soft_warnings, phase_warning]
        for warning_text in all_warning_parts:
            if warning_text and warning_text != self._last_gate_warning:
                result = warning_text + "\n" + result
                self._last_gate_warning = warning_text

        # Post-launch informational gates — append to training output
        if tool_name == "shell" and not error:
            cmd = arguments.get("command", "")
            post_info = self._run_post_execution_gates(cmd, result)
            if post_info:
                result = result + "\n" + post_info

        # Track for next call's workaround detection
        cmd_key = arguments.get("command", "") if tool_name == "shell" else tool_name
        self._last_tool_call = (tool_name, cmd_key, error)

        # Cache tool result for duplicate detection within this turn
        self._cache_tool_result(tool_name, arguments, result if not error else f"ERROR: {result}")

        # Periodic save: save conversation every N tool calls
        self._tool_calls_since_save = getattr(self, '_tool_calls_since_save', 0) + 1
        if self._tool_calls_since_save >= self._AUTOSAVE_INTERVAL:
            self._tool_calls_since_save = 0
            self._save_conversation()

        display.tool_done(tool_name, elapsed, detail=detail, error=error)
        return result

    @staticmethod
    def _is_flagscale_dryrun(cmd):
        """FlagScale --dryrun only generates launch scripts — it NEVER launches training.

        No GPU is used, no training process starts, no logs are produced.
        This is fundamentally different from a validation run (--train-iters 20)
        which actually launches real training.
        """
        cmd_lower = cmd.lower()
        return any(p in cmd_lower for p in [
            '--dryrun', '--dry-run', '--dry_run',
            'action=dryrun', 'action=dry_run', 'action=dry-run',
        ])

    _PORTING_CONFIRM_KEYWORDS = re.compile(
        r'\b(mode\s*[bc]|megatron|native|fsdp|huggingface|hf\s*wrapper)\b',
        re.IGNORECASE,
    )
    _PORTING_MODE_B_KEYWORDS = re.compile(
        r'\b(mode\s*b|megatron\s*native|megatron|native|原生)\b', re.IGNORECASE,
    )
    _PORTING_MODE_C_KEYWORDS = re.compile(
        r'\b(mode\s*c|fsdp|huggingface|hf\s*wrapper|wrapper)\b', re.IGNORECASE,
    )

    _PORTING_MODE_CLASSIFY_PROMPT = (
        "The user was asked to choose between two porting approaches:\n"
        "- Mode B (Megatron Native): rewrite model using Megatron parallel layers\n"
        "- Mode C (HuggingFace Wrapper): wrap existing HF model with FSDP\n\n"
        "The user replied: \"{user_input}\"\n\n"
        "Which mode did the user choose? Reply ONLY with one of: "
        '{"mode": "mode_b"} or {"mode": "mode_c"} or {"mode": "none"}\n'
        "Reply \"none\" if the message is not choosing a mode."
    )

    def _check_user_porting_confirmation(self, user_input):
        """Check if user's message confirms a porting path choice. Regex first, LLM fallback."""
        if not self._porting_mode or self._porting_path_confirmed:
            return

        # Fast path: regex matches a clear keyword
        if self._PORTING_CONFIRM_KEYWORDS.search(user_input):
            self._porting_path_confirmed = True
            if self._PORTING_MODE_B_KEYWORDS.search(user_input):
                self._confirmed_porting_path = "mode_b"
            elif self._PORTING_MODE_C_KEYWORDS.search(user_input):
                self._confirmed_porting_path = "mode_c"
            else:
                self._confirmed_porting_path = self._llm_classify_porting_mode(user_input)
            logger.info("Porting path confirmed (regex): %s (input: %s)",
                        self._confirmed_porting_path, user_input[:100])
            return

        # LLM fallback: short messages that might be implicit confirmation
        # e.g. "B，加油", "就用第一个方案吧", "go with native"
        if len(user_input.strip()) < 200:
            mode = self._llm_classify_porting_mode(user_input)
            if mode in ("mode_b", "mode_c"):
                self._porting_path_confirmed = True
                self._confirmed_porting_path = mode
                logger.info("Porting path confirmed (LLM): %s (input: %s)",
                            self._confirmed_porting_path, user_input[:100])

    def _llm_classify_porting_mode(self, user_input):
        """Use LLM to classify which porting mode the user chose."""
        try:
            prompt = self._PORTING_MODE_CLASSIFY_PROMPT.format(user_input=user_input[:300])
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                mode = data.get("mode", "none")
                if mode in ("mode_b", "mode_c"):
                    return mode
        except Exception as e:
            logger.debug("LLM porting mode classification failed: %s", e)
        return "unknown"

    @staticmethod
    def _is_quick_test_command(cmd):
        """Check if a training-like command is not a real training launch.

        Returns True for:
        - FlagScale dryrun (script generation only)
        - Help/version queries
        - Short validation runs (--train-iters 20)
        """
        cmd_lower = cmd.lower()
        # FlagScale dryrun — script generation, no training
        if any(p in cmd_lower for p in [
            '--dryrun', '--dry-run', '--dry_run',
            'action=dryrun', 'action=dry_run', 'action=dry-run',
        ]):
            return True
        # Help/version/imports — not training
        if any(p in cmd_lower for p in ['--help', '-h', '--version', 'python -c', 'import ']):
            return True
        # Short validation runs — real training but only a few iterations (up to 50)
        numeric_patterns = [
            r'--total[_-]steps[\s=]+([1-9]|[1-4]\d|50)\b',
            r'--max[_-]steps[\s=]+([1-9]|[1-4]\d|50)\b',
            r'--num[_-]steps[\s=]+([1-9]|[1-4]\d|50)\b',
            r'--train[_-]iters[\s=]+([1-9]|[1-4]\d|50)\b',
        ]
        for pattern in numeric_patterns:
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

    def _auto_load_companion_skills(self, skill_names):
        """Auto-load a list of companion skills that haven't been loaded yet."""
        for name in skill_names:
            if name in self._loaded_skills:
                continue
            try:
                content = self.skill_manager.load(name)
                content = self._maybe_summarize_skill(name, content)
                tool_call_id = f"auto_{uuid.uuid4().hex[:8]}"
                fake_response = {
                    "content": None,
                    "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": name}}],
                }
                self.history.append(self.provider.format_assistant_message(fake_response))
                self.history.append(self.provider.format_tool_result(
                    tool_call_id, f"[Skill '{name}' loaded — content available in system context]"))
                self._loaded_skills.add(name)
                self._active_skill_content[name] = content
                self._skill_load_iterations[name] = self._total_iterations
                if "data-prep" in name:
                    self._data_prep_mode = True
                display.skill_auto_loaded(name)
            except Exception:
                pass

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
                # Activate mode flags (same as _post_tool_execution)
                if "model-porter" in skill_name:
                    self._porting_mode = True
                    self._auto_load_companion_skills(self._PORTING_COMPANION_SKILLS)
                if "data-prep" in skill_name:
                    self._data_prep_mode = True
                display.skill_auto_loaded(skill_name)
            except Exception:
                pass

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