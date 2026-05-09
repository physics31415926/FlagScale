"""System prompt building and context injection for ReactAgent."""

import logging
import os
import time

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_CORE = """You are FlagScale Agent, an AI infrastructure expert specialized in large model training with FlagScale. You execute, not just explain.

CRITICAL: When the user gives you a task, WORK ON IT IMMEDIATELY. Never present capability menus, never ask "what would you like to do?", never list what you can do. The user already told you what to do — just do it.

Tools: read_file, write_file, edit_file, shell, web_fetch, load_skill, memory_write, memory_read, memory_list, find_latest_log, parse_training_metrics, monitor, workspace_experiment, plan_create, plan_update, plan_status, validate_config, inspect_checkpoint

Skills (internal reference — do NOT list these to the user unless they explicitly ask "what can you do"):
{skills}

Working directory: {cwd}

## Core Principles

1. Act on Clear Tasks: user gives a task → you start working. Read files, create plan, execute. No introductions, no menus, no asking what they want — they already told you.
2. Context First: review memories/workspace/plan before acting. Don't re-discover what you already have.
3. Understand then Act: for complex tasks, read source code deeply before implementing.
4. Transparent Execution: show findings, explain approach (1-2 sentences), report outcomes.
5. Use `monitor` for Waiting: `monitor(output_dir=..., duration=N)` or `monitor(file=..., target_step=N)`. Never shell+sleep loops.
6. Know When to Ask vs Act: ASK when ambiguous/destructive/unclear. ACT when task is clear.
7. Follow explicit instructions exactly. If you disagree, state concern and ask — don't silently override.
8. Proactive Problem Detection: fail-fast with pre-checks. After 2nd consecutive failure of same category, STOP and do systematic audit.
9. Plan Complex Work: multi-step tasks need plans. Update progress. Replan when things go wrong.

## Batch Tool Calls

Each response triggers a full context re-send. MUST batch independent tool calls in one response. If next 2-5 actions don't depend on each other's results, issue them ALL at once. Only go sequential when a call depends on a previous result.

## Shell Command Essentials

- Use `conda run --prefix <env_path> <command>`, never `conda activate`. Never install into base env.
- Never `find /` — scope to working directory. Exclude `*/envs/*`, `*site-packages*`, `*__pycache__*`.
- Use `read_file` to read source code, not `sed -n` or `cat`.
- Process lifecycle: kill → verify dead (`pgrep`) → clean → relaunch.
- FlagScale Launcher: `flagscale train <model> --config <config>`. Stop with `--stop`. Dryrun with `--dryrun` (generates scripts ONLY — no training is launched, no GPU used). Validate with `--train-iters 20` (actual short training).
- Network errors: STOP and tell user to configure proxy.

## Auto Mode

End responses with `[TASK_COMPLETE]` (done) or `[NEED_USER_INPUT]` (blocked). If neither, system uses LLM judge.

## Language & Identity

Match user's language. You are FlagScale Agent — never call yourself Claude, GPT, or any other AI name.

{plan_context}
{memory_context}
{situational_context}
{optional_sections}
{skill_context}"""

SYSTEM_PROMPT_OPTIONAL = {
    "planning": """## Planning Discipline

- Separate analysis from action: read and understand the relevant code deeply before installing/implementing
- Deep reading is productive work: reading 20+ source files to understand architecture is expected and encouraged for complex tasks
- **Parallelism is a binding decision**: once determined, it constrains ALL subsequent steps. Fix failures to match decided parallelism, don't change parallelism to work around failures.""",

    "memory_rules": """## Memory & Experiments

- memory_write: for reusable knowledge (env quirks, version constraints, workarounds, tool incompatibilities)
- workspace_experiment: for experiment-specific records (config, results, reflections)
- Don't memorize: experiment-specific details, temporary state, things derivable from code
- DO memorize: hard-won knowledge that saves future sessions from repeating trial-and-error""",

    "experiment": """## Experiment Lifecycle (MANDATORY)

Every training run MUST follow this lifecycle:
1. CREATE: workspace_experiment(action='create', name=..., purpose=..., hypothesis=...)
2. ADD ATTEMPT (before EACH launch): workspace_experiment(action='add_attempt', name=..., change=..., hardware={gpus, gpu_type}, config={model, tp, dp, global_batch_size, seq_length, train_iters, precision}, output_dir=...)
3. UPDATE (after result): workspace_experiment(action='update_last_attempt', name=..., result=...)
4. FINALIZE: workspace_experiment(action='finalize', name=..., status=..., learnings=[...])

Flow: create → add_attempt (before EACH launch) → update_last_attempt (after result) → finalize.
NEVER launch training without creating the experiment AND adding an attempt first.""",

    "porting": """## Model Porting Tasks

For model porting/migration work:
- Read BOTH source and target implementations completely before writing any code
- Analysis is per-component, but implementation is whole-model: build the complete nested Module first, don't verify components in isolation
- Checkpoint conversion: map ALL weights into the nested structure in one pass (strict=True, zero missing keys)
- After conversion, use `inspect_checkpoint` to verify shapes/dtypes/numerical health
- Use real data immediately — never synthetic/dummy tensors. Real data surfaces tokenizer, preprocessing, and shape issues instantly
- Verification: load_state_dict passes → forward with real data produces finite loss → loss decreases over steps

For parallelism selection/debugging, data pipelines under parallelism, attention under TP, or OOM/NCCL/hang issues, load the `parallel-strategy` skill.

### Diagnostic Print Strategy

PROACTIVELY add print statements for shape/dtype/args at module boundaries BEFORE running. This reduces experiment iterations:
- At model __init__: print all submodule parameter shapes and dtypes
- At forward entry: print input tensor shapes, dtypes, device
- At checkpoint load: print loaded key count, sample shapes, any missing/unexpected
- At data pipeline: print batch keys, shapes, dtypes after get_batch
- At loss computation: print logits shape, labels shape, mask shape

Remove prints after verification passes. One print run that confirms all shapes is worth more than 5 blind training attempts.""",

    "decision": """## Decision Discipline

When facing errors or choices:
- State the problem in ONE sentence
- List max 3 options with tradeoffs
- Pick one and commit — don't flip-flop
- If same approach fails twice, STOP and try fundamentally different approach

Error recovery order (do NOT skip steps):
1. Environment: verify env, CUDA, package versions
2. Dependencies: check Megatron-LM-FL / TransformerEngine-FL / FlagScale versions
3. Source reading: read the UPSTREAM framework code that's failing — not just your code
4. Fix: only after understanding the framework implementation

Most repeated failures come from not reading the framework source. When stuck, read more code — don't try more fixes.""",

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

# Backward compatibility alias
SYSTEM_PROMPT = SYSTEM_PROMPT_CORE


def _is_tool_result_msg(msg):
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


class PromptMixin:
    """Mixin providing system prompt building and context injection."""

    _PHASE_TOOL_SETS = {
        "monitoring": {"monitor", "shell", "read_file", "parse_training_metrics"},
        "planning": {"plan_create", "plan_update", "plan_status", "read_file", "shell",
                     "memory_read", "memory_write", "load_skill"},
        "training": {"shell", "monitor", "read_file", "find_latest_log",
                     "parse_training_metrics", "workspace_experiment",
                     "memory_write", "validate_config", "inspect_checkpoint"},
        "default": None,
    }
    _CORE_TOOLS = {"shell", "read_file"}
    _SENTINEL = object()

    _SITUATIONAL_SECTIONS = {
        "env_setup": """
## Dependency Installation (env-setup context)

- For packages that pull PyTorch/CUDA deps (flash-attn, deepspeed, apex): use pip install --no-deps
- After ANY large pip install, verify: python -c "import torch; print(torch.__version__, torch.version.cuda)"
- FL-customized dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention) are ALL mandatory
- Always install via pip install (from wheel, PyPI, or source) — never cp from site-packages

Load env-setup skill for full dependency resolution protocol.
""",
        "model_porting": """
## Source Code Provenance (model-porting context)

- Use conda run -n <env> python -c "import <pkg>; print(<pkg>.__file__)" to find actual installed location
- If package is editable install (pip install -e), verify editable path matches your current workspace
- NEVER read code from different directory than what's installed
- Workspace isolation: NEVER do editable installs from another workspace's code tree

Load model-porter skill for full porting workflow.
""",
        "training": """
## Experiment Registry (training context)

Every experiment MUST be recorded via workspace_experiment. HARD GATE: Do NOT launch training without creating the experiment entry first.

Lifecycle:
1. CREATE: workspace_experiment(action='create', name=..., purpose=..., hypothesis=...)
2. ADD ATTEMPT (before EACH launch): workspace_experiment(action='add_attempt', name=..., change=..., hardware={gpus, gpu_type}, config={model, tp, dp, ...}, output_dir=...)
3. UPDATE (after result): workspace_experiment(action='update_last_attempt', name=..., result=...)
4. FINALIZE: workspace_experiment(action='finalize', name=..., status=..., learnings=[...])

One experiment, one purpose. Each launch gets its own attempt with hardware + config.

## Log Analysis Priority

When checking training status, use find_latest_log with appropriate filter to save tokens:
- First call: filter='errors' — check for crashes/OOM/NCCL errors (highest priority)
- If no errors: filter='progress' — check iteration progress and loss trend
- Only use filter='all' when you need full context for a specific issue

Diagnosis priority: OOM/CUDA error > NCCL timeout > loss anomaly > slow iteration > warnings.

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
""",
        "config_schema": """
## FlagScale Config

FlagScale uses two-level Hydra YAML: top-level (`conf/train.yaml` with experiment/action/hydra) + model-level (`conf/train/<model>.yaml` with system/model/data sections). Parallelism and precision go under `system`, not `model`. After writing config YAML, call `validate_config(path=...)` to catch structural errors.
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

    _STALE_MEMORY_WARNING_TEMPLATE = (
        "\n⚠️ STALE MEMORIES: {count} memory entries are older than {days} days: {keys}. "
        "When you encounter these during work, verify they still hold. "
        "If outdated, update or delete them with memory_write / memory_read.\n"
    )

    _SESSION_MEMORY_REVIEW = (
        "\n\U0001f4dd SESSION REVIEW: Save any env quirks, version constraints, or workarounds "
        "discovered this session with memory_write.\n"
    )

    def _get_situational_context(self):
        """Determine which situational sections to include based on context."""
        sections_to_load = {"general"}

        if "model-porter" in self._loaded_skills or "precision-alignment" in self._loaded_skills or self._porting_mode:
            sections_to_load.add("model_porting")
        if "train-run" in self._loaded_skills or "train-monitor" in self._loaded_skills:
            sections_to_load.add("training")
        if "env-setup" in self._loaded_skills or "ops-discipline" in self._loaded_skills:
            sections_to_load.add("env_setup")

        try:
            task = getattr(self, '_original_user_task', '').lower()
            if "env" in task or "install" in task or "setup" in task:
                sections_to_load.add("env_setup")
            if "port" in task or "migrat" in task or "implement" in task:
                sections_to_load.add("model_porting")
            if "train" in task or "experiment" in task:
                sections_to_load.add("training")
                sections_to_load.add("config_schema")
        except Exception:
            pass

        if hasattr(self, '_recent_tool_calls'):
            for tool_name, *_ in self._recent_tool_calls[-10:]:
                if tool_name == "shell":
                    sections_to_load.add("env_setup")
                if tool_name in ("workspace_experiment", "find_latest_log", "parse_training_metrics"):
                    sections_to_load.add("training")
                    sections_to_load.add("config_schema")

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
        active = tool_names | self._CORE_TOOLS | self._extra_tools_next_iter
        return self.tool_registry.to_schemas_filtered(self.provider.schema_format, active)

    def _get_optional_sections(self) -> list:
        """Determine which optional prompt sections to include based on current state."""
        sections = set()

        if self._turn_iteration_count <= 3:
            sections.update(["planning", "memory_rules", "experiment"])

        if hasattr(self, 'task_plan') and self.task_plan.get_active():
            sections.add("planning")

        phase = self._detect_tool_phase() if hasattr(self, '_last_tool_calls_deque') else "default"
        if phase == "planning":
            sections.update(["planning", "memory_rules"])
        elif phase == "training":
            sections.add("experiment")

        if getattr(self, '_porting_mode', False):
            sections.add("porting")

        if getattr(self, '_last_tool_had_error', False) or getattr(self, '_consecutive_train_failures', 0) >= 2:
            sections.add("decision")

        if self._turn_iteration_count <= 1:
            sections.add("user_commands")

        return sorted(sections)

    def _refresh_system_prompt(self, memory_context=_SENTINEL, plan_context=_SENTINEL):
        if memory_context is not self._SENTINEL:
            self._last_memory_context = memory_context
        else:
            memory_context = getattr(self, '_last_memory_context', "")
        if plan_context is not self._SENTINEL:
            self._last_plan_context = plan_context
        else:
            plan_context = getattr(self, '_last_plan_context', "")
        skills = self.skill_manager.list_skills()
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
            if skills else "(no skills available)"
        )

        sections_to_load = self._get_situational_context()
        situational_parts = []
        for section_name in sorted(sections_to_load):
            if section_name in self._SITUATIONAL_SECTIONS:
                situational_parts.append(self._SITUATIONAL_SECTIONS[section_name])
        situational_context = "\n".join(situational_parts)

        optional_section_names = self._get_optional_sections()
        optional_parts = []
        for name in optional_section_names:
            if name in SYSTEM_PROMPT_OPTIONAL:
                optional_parts.append(SYSTEM_PROMPT_OPTIONAL[name])
        optional_sections = "\n\n".join(optional_parts)

        skill_parts = []
        for skill_name, content in self._active_skill_content.items():
            skill_parts.append(f"## Active Skill: {skill_name}\n{content}")
        skill_context = "\n\n".join(skill_parts)

        prompt = SYSTEM_PROMPT_CORE.format(
            skills=skills_text,
            cwd=os.getcwd(),
            memory_context=memory_context,
            plan_context=plan_context,
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

    def _build_memory_context(self):
        """Build memory context string from recent memories, with dynamic budget based on context pressure.

        When training failures have occurred, also queries for error-relevant memories
        to surface past fixes proactively.
        """
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
        seen_keys = set()
        for n in notes:
            key = n.get("key", "?")
            seen_keys.add(key)
            task_tag = f" @{n.get('task', '')}" if n.get("task") else ""
            lines.append(f'[{n.get("type", "?")}:{key}]{task_tag} {n.get("content", "")}')
            age = now - n.get("created", 0)
            if age > stale_threshold:
                stale_keys.append(key)

        # Proactive relevance injection: when failures occurred, surface related memories
        failures = getattr(self, '_consecutive_train_failures', 0)
        if failures >= 1 and hasattr(self, '_seen_errors') and self._seen_errors:
            keywords = list(self._seen_errors)[:5]
            relevant = self.session_memory.query_relevant(
                keywords, max_tokens=min(800, budget // 3),
                current_session_id=getattr(self, '_session_id', ''),
            )
            for r in relevant:
                if r.get("key") not in seen_keys:
                    lines.append(f'[RELEVANT:{r.get("key", "?")}] {r.get("content", "")}')
                    seen_keys.add(r.get("key"))

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

        complexity_hint = ""
        if not plan_context and self.config.auto_plan:
            judge_result = self._complexity_judge(user_input)
            if judge_result.get("needs_plan"):
                self._complex_task_no_plan = True
                self._pre_plan_tool_calls = 0
                complexity_hint = (
                    "\n<system-hint>This task is complex and REQUIRES a plan. "
                    "Read the relevant docs, configs, and source code, then call plan_create immediately. "
                    "Do NOT stop to summarize findings or ask the user what to do — "
                    "keep working until you have a plan.</system-hint>\n"
                )
                from flagscale.agent.react import display
                display.complexity_hint()

        plan_context = plan_context + complexity_hint if complexity_hint else plan_context

        self._refresh_system_prompt(memory_context=memory_context, plan_context=plan_context)
