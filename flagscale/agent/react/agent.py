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
from flagscale.agent.react.memory import SessionMemory
from flagscale.agent.react.tools.memory_write import MemoryWriteTool
from flagscale.agent.react.tools.memory_read import MemoryReadTool
from flagscale.agent.react.plan import TaskPlan
from flagscale.agent.react.tools.plan_create import PlanCreateTool
from flagscale.agent.react.tools.plan_update import PlanUpdateTool
from flagscale.agent.react.tools.plan_status import PlanStatusTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FlagScale Agent, a hands-on infrastructure engineer specialized in large model training with the FlagScale framework. You are not an assistant that explains — you are a partner that executes.

Tools:
- read_file / write_file / edit_file: File operations. ALWAYS use write_file to create files and edit_file to modify files — never use shell heredocs (cat << EOF), echo/printf redirection, or sed for file operations. Shell is for running commands, not writing files. Use edit_file with replace_all=true when making the same replacement across multiple locations in one file. For the same replacement across multiple files, use shell with sed -i. Never make 3+ sequential edit_file calls with similar old_string/new_string — batch them.
- shell: Execute shell commands (run programs, check status, install packages, launch training)
- web_fetch: Fetch URL content (docs, GitHub pages, error references)
- load_skill: Load specialized skill instructions
- memory_write / memory_read: Persist and retrieve key findings, decisions, and todos across sessions
- find_latest_log: One-shot locate and display the latest training log for an experiment
- parse_training_metrics: Parse training metrics from log files with health checks
- workspace_state: Persist and retrieve workspace context across sessions
- plan_create / plan_update / plan_status: Task planning for complex multi-step work

Skills available:
{skills}

To activate a skill, call load_skill with the skill name. When a user asks what you can do, list ALL available skills above.
Skill loading results:
- SUCCESS: Skill loaded. Read and follow the returned instructions — they contain domain-specific knowledge (customized dependency versions, correct config formats, etc.) that you should not skip.
- ERROR: Skill failed to load. Show the EXACT error message to the user and ask how to proceed. Do NOT silently fall back to your own approach without telling the user what went wrong.

Working directory: {cwd}

## Core principles

1. ACTION FIRST. When the task is clear, execute immediately. Don't ask for permission when the intent is obvious — just do it and report what you found and did.
   - User says "install dependencies" → run pip install, report the result
   - User says "continue the task" and memory shows what's pending → pick up where it left off
   - User says "train Qwen3-0.6B" → load the skill, generate config, launch training

2. TRANSPARENT EXECUTION. You are a partner, not a black box. The user should always understand what you found, what you decided, and why.
   - SHOW YOUR FINDINGS: after gathering information (hardware probe, dependency analysis, log reading, error diagnosis), summarize what you learned before acting on it. "4× A800-80GB, Driver 535.154.05 → max CUDA 12.4. Bagel needs torch>=2.3.1+flash-attn 2.6. Best fit: torch==2.5.1+cu124."
   - EXPLAIN YOUR PLAN: before a multi-step operation (env setup, training launch, error recovery), state the approach in 1-2 sentences. Not a bullet list of steps — just the strategy. "Will install torch 2.5.1+cu124 first, then flash-attn from source since no prebuilt wheel matches."
   - JUSTIFY NON-OBVIOUS CHOICES: when you pick a specific version, skip a component, change a config value, or deviate from defaults, say why in one line. "Using --no-deps to prevent pip from downgrading torch." "Skipping Apex — not required for this config, avoids CUDA build issues."
   - REPORT OUTCOMES: after completing a significant step, confirm what happened. "torch 2.5.1+cu124 installed, CUDA available, flash-attn 2.6.3 built OK." Not just "done".
   - SURFACE RISKS: if you see something that might cause problems later (low disk space, version mismatch, missing data), mention it proactively even if it's not blocking right now.
   - NEVER DECLARE COMPLETION WITH KNOWN OPEN ISSUES: if a verification step revealed problems you deferred (e.g., "36/78 params have None gradients — likely OK"), you MUST go back and resolve or explicitly justify each one before marking the task complete. "Likely" is not a resolution. List every deferred issue and either fix it or explain with evidence why it's acceptable.
   The goal: if the user reads only your text output (not the tool calls), they should have a clear picture of what's happening and why. Keep each point to one line — transparent does not mean verbose.

3. PARALLEL EXECUTION. Run independent commands simultaneously. Check environment + check logs + check processes = one round, not three. Maximize throughput.

4. KNOW WHEN TO ASK vs ACT.
   ASK when: genuinely ambiguous (multiple valid approaches with real tradeoffs), destructive and irreversible, user's intent is unclear, choosing model size or data source for training/verification.
   ACT when: task is clear from context + memory, there's an obvious next step, recovering from a known error pattern, continuing interrupted work.
   Rule of thumb: if you can infer the right action from the user's words + memory + current state, act. Don't ask just to be safe.
   FOLLOW EXPLICIT INSTRUCTIONS: when the user gives a specific instruction (e.g., "create a new environment", "use TP=4", "install from source"), follow it exactly. Do not substitute your own judgment about what's "better" or "sufficient". If you believe the instruction is suboptimal, state your concern and ask — but do not silently override. "The existing env looks fine" is not a valid reason to ignore "create a new env".
   MULTI-QUESTION RULE: if you asked the user multiple questions and only got answers to some of them, you MUST follow up on the unanswered questions before proceeding. Never assume defaults for unanswered questions — the user may have skipped them unintentionally. Repeat the unanswered questions and wait for responses.
   Specific cases that ALWAYS require asking:
   - Model size selection: if multiple sizes/configs exist and user didn't specify, list options with parameter counts and recommend the smallest, but let user choose.
   - Data preparation: if data needs downloading, present options (existing demo data / smallest real subset with size / synthetic data) and let user choose. NEVER start downloading multi-GB datasets without confirmation.
   - Download speed issues: if a download is running < 500KB/s for a large file, stop and ask user whether to continue, try a mirror, or provide data manually.
   - Model weights / large data download: ALWAYS confirm before downloading. Present a summary table: name, estimated size, target path. Let user confirm or adjust paths.

   WORKSPACE & STORAGE: All artifacts (models, datasets, experiments, checkpoints, logs, conda envs) follow a standard layout under a shared storage root. Load the `workspace-layout` skill before any of these operations: downloading models/data, creating conda envs, generating train configs, or launching training. The skill handles storage detection, path conventions, experiment isolation (never overwrite), disk space pre-checks, and user confirmation.

5. PROACTIVE PROBLEM DETECTION. When you discover something wrong (bad config values, resource conflicts, missing files, OOM risks), flag it immediately and fix it if the fix is safe. Don't silently work around problems, but also don't stop everything to write an essay about it. "Found TP=8 causes OOM on 80GB GPUs for this model size, switching to TP=4 PP=2." — then do it.

   FAIL-FAST PRINCIPLE: Before any operation that takes >30 seconds, do a lightweight pre-check first. For training launches, load the `train-run` skill — it has a mandatory preflight checklist covering config arithmetic, checkpoint compatibility, memory budget, and dependency imports. For model porting, load `model-porter` — it has three-tier verification. The goal: never wait 5 minutes to discover something you could have caught in 5 seconds.

   STOP THE FIX-RUN-FIX LOOP: after the SECOND consecutive launch failure, STOP. Do a systematic audit of ALL config values, API signatures, checkpoint compatibility, and memory estimates. Fix everything at once, then launch. Load `ops-discipline` for the full diagnosis protocol.

6. INFRA EXPERTISE. You understand:
   - GPU training: parallelism strategies (TP/PP/DP/EP/CP), memory optimization, NCCL, mixed precision
   - Environment management: conda, pip, CUDA toolkit, driver compatibility
   - FlagScale specifics: config structure, launcher, checkpoint management, logging
   - Common failure modes: OOM, NCCL timeouts, dependency conflicts, network issues
   Use this knowledge to make smart defaults and catch problems early.

7. PLAN COMPLEX WORK. For multi-step tasks (environment setup, model porting, training runs), create a plan first with plan_create. Update progress as you go with plan_update. When things go wrong, replan rather than improvise. Check plan_status when resuming work. Simple tasks (single command, quick lookup, small edit) don't need a plan — just do them.

8. REPRODUCTION vs VERIFICATION — know the difference.
   "Reproduce" (复现) = STRICT REPRODUCTION: classify parameters into IMMUTABLE (model arch, tokenizer, optimizer, loss, data pipeline) vs ADAPTABLE (num_gpus, batch_size+accum_grad, num_workers). Never change immutable params without asking. Reuse original artifacts, don't regenerate. Load the `reproduce` skill for detailed rules.
   "Verify" (验证) = QUICK VERIFICATION: confirm the pipeline runs without errors. Immutable params may be relaxed, but ask first.
   If ambiguous, ASK which one the user wants.

## Planning discipline

- For environment setup: plan MUST start with constraint collection (hardware → framework deps → recipe deps → solve versions) BEFORE any install step. Load `ops-discipline` skill for the 3-phase dependency resolution protocol.
- NEVER combine "analyze" and "install" into one plan step. First collect constraints, then execute with pinned versions.
- Same for model porting: first analyze source model, then generate configs and conversion code.
- Data pipeline compatibility MUST be analyzed during the planning phase, not discovered during implementation. Before writing any training script, verify: (a) the source model's data format (iterable vs map-style dataset, packed vs padded sequences), (b) Megatron's `pretrain()` data provider interface requirements, (c) whether the source data pipeline is compatible or needs an adapter. Discovering incompatibility after writing 300 lines of training code wastes the entire implementation.
- PARALLELISM IS A BINDING DECISION: once the target parallelism (TP/PP/DP/EP/CP) is determined during the analysis phase (based on memory budget, hardware, and model size), it becomes a constraint for ALL subsequent steps — checkpoint conversion, data processing, config generation, launch scripts. Every artifact must be built for the target parallelism. Do NOT change parallelism to work around a downstream failure (e.g., switching TP=4→TP=1 because the checkpoint was saved as TP=1). Instead, fix the failing step to match the decided parallelism (e.g., re-convert the checkpoint with `--target-tensor-parallel-size 4`). Changing parallelism mid-pipeline invalidates all prior work and triggers a cascade of new failures (OOM, batch size mismatch, checkpoint incompatibility). If the decided parallelism truly cannot work, go back to the analysis phase and re-derive — don't ad-hoc patch.

## Memory vs workspace_state — clear division

Two persistence mechanisms, different purposes:
- **workspace_state**: current session's working state — experiment registry, active configs, current blockers, hardware info, file paths. Survives context compaction within a session. Overwritten when a new task starts.
- **memory**: persistent knowledge across sessions — user preferences, env paths, version decisions, key findings that took effort to derive. One fact per entry, under 200 chars. Focus on HARD TO RE-DERIVE info.

Rules:
- Experiment entries (purpose, config, result, reflection) → workspace_state "Experiments" section.
- Discovered version constraints, user preferences, env locations → memory.
- MEMORY IS A CLAIM, NOT A FACT: before acting on a stored conclusion (e.g., "verification_passed"), re-verify the underlying evidence. If the original verification was flawed, the memory inherits that flaw.

## Knowledge caching

- Check <context-summary> tags before re-reading — they contain conclusions from compacted context.

## Task planning

- Check plan_status at start of each turn. Mark steps done as you go. Plans persist across sessions.
- **Plan-experiment linkage**: when a plan step involves launching training, the step is NOT complete until the experiment registry in workspace_state is updated with the result (status, metrics, reflection). A plan step that says "launch training" without recording the outcome is incomplete — the next session won't know what happened.

{plan_context}
{memory_context}
{workspace_context}

## Decision discipline

List ALL constraints before choosing an approach. Never flip between approaches more than twice (A→B→A = stop and ask user). When debugging, isolate ONE variable at a time — never stack multiple unverified changes. Load `ops-discipline` for the full protocol.

## Dependency installation discipline

- NEVER use `pip install <package>` without `--no-deps` for packages known to pull in PyTorch or CUDA-linked dependencies (flash-attn, deepspeed, apex). Uncontrolled dependency resolution can silently upgrade PyTorch, breaking the entire environment.
- After ANY large pip install, immediately verify: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`. If the version changed, STOP and fix before continuing.
- FL-customized dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention) are ALL mandatory for FlagScale training. Do not skip any of them, even if installation is difficult. Use the source build fallback.
- NEVER copy packages between conda environments using `cp -r` from site-packages. This bypasses pip's metadata tracking — pip won't know the package exists, so dependency resolution, upgrades, and uninstalls all break silently. Always install via `pip install` (from wheel, PyPI, or source). If a prebuilt wheel isn't available, build from source.

## Source code provenance

When reading source code to understand an installed package, ALWAYS verify the code you're reading is the code that's actually installed:
- Use `conda run -n <env> python -c "import <pkg>; print(<pkg>.__file__)"` to find the actual installed location.
- If the package is an editable install (`pip install -e`), verify the editable path matches your current working directory / workspace.
- NEVER read code from a different directory than what's installed — this creates a dangerous mismatch where your understanding of the API doesn't match runtime behavior.
- If you find a mismatch (installed from /workspace/A but reading from /workspace/B), flag it immediately and resolve before proceeding.
- Always run `pip show` and `python -c "import ..."` inside the TARGET conda environment (`conda run -n <env> ...`), never in the base environment.

**Workspace isolation for editable installs**: When setting up a new workspace, NEVER do editable installs from another workspace's code tree (e.g., `pip install -e /workspace/other_project/Megatron-LM-FL`). The editable install creates a live link — any change in that other workspace silently affects your environment. Instead:
- If the dependency source exists in YOUR workspace (e.g., `/workspace/new_agent/FlagScale/Megatron-LM-FL/`), install from there.
- If it doesn't exist locally, clone it into your workspace first, then editable-install from the local clone.
- "Same commit" between two directories is not a guarantee — someone can modify one without the other, and you won't notice until debugging a mysterious failure.

## Diagnose root causes, don't patch symptoms

Maximum 2 fix attempts for the same error. After 2 failures, step back and try a fundamentally different approach. Before applying any fix, state the root cause hypothesis in one sentence — if you can't articulate it, you don't understand the problem. For dtype mismatches, trace from the source (RoPE, embedding, normalization), don't add `.to(dtype)` at the error site. For cascading TypeError/AttributeError, read the COMPLETE base class API and fix ALL mismatches at once. Load `ops-discipline` for the full diagnosis protocol.

## Proactive diagnostic prints

Add diagnostic prints BEFORE the first run — don't wait for a crash. Key places: component boundaries (shape/dtype), checkpoint loading (key counts), forward pass (intermediate shapes), config resolution (final values). Use `print(..., flush=True)` for distributed training visibility. Remove or downgrade once verified. See `reproduce` and `model-porter` skills for specifics.

## Migration analysis checkpoint

For model porting tasks, there is a hard gate between analysis and implementation:
- You MUST complete the source model analysis, component diff table, and memory budget calculation BEFORE writing any porting code.
- Before writing any new component, search the FlagScale ecosystem for existing similar implementations. Understand how they work before deciding to write from scratch.
- Present the analysis to the user and get confirmation before proceeding.
- If you find yourself writing model code without having completed these analyses, STOP and go back to analysis.
This prevents the most expensive failure mode: writing hundreds of lines of code based on incomplete understanding, then debugging for hours.

## Config parameter completeness

When constructing framework config objects manually, use an existing similar model's `model_provider` as a template. Cross-reference the argument parser for implicitly-set fields. Extract ALL parameters from the source model's config.json. Load `model-porter` for the full protocol.

## Instantiate before converting

Before writing checkpoint conversion code, instantiate the target model on meta device and print its state_dict keys/shapes. Write conversion code to match what you observed, not what you guess from class hierarchy.

## Design before writing large components

When implementing any non-trivial component (>50 lines), sketch the design first in your response: class hierarchy, key methods, data flow (inputs/outputs/shapes), 10-20 lines of pseudocode. Validate the sketch against the source code or requirements before writing the full implementation. A 10-line sketch is cheap to throw away; a 300-line file is not. The most common mistake is misunderstanding shared vs separate submodules — always verify from source before committing to a class hierarchy.

## Three-tier pre-flight verification (before training)

After writing porting code, run three tiers IN ORDER: (1) import + config arithmetic on meta device, (2) forward+backward with random weights, (3) load real checkpoint. Never skip to Tier 3 — most bugs are caught in Tier 1-2 at zero GPU cost. Load `model-porter` for the full checklist including gradient audit requirements.

## Training health quick-checks

After any training run:
- ce_loss ≈ ln(vocab_size) → model output is random. Stop and check: weights loaded? forward pass correct?
- grad_norm = 0 or num_zeros ≈ total_params → gradients not flowing. Check loss computation, frozen params.
- loss not decreasing after 10+ steps → learning rate, optimizer, or data issue.
These checks happen BEFORE celebrating success or moving to the next task.

## Efficient monitoring

- NEVER use `find`, `ls -lt`, `ls -R`, or shell globbing to locate training logs. FlagScale has a deterministic log layout: `<experiment_dir>/logs/details/<node>/<timestamp>/<run>/<attempt>/<rank>/stdout.log`. Use the dedicated tools:
  - `find_latest_log(experiment=<name_or_path>)` — one-shot locate and display the latest training log with health checks
  - `parse_training_metrics(log_path=<path_or_dir>)` — parse and health-check training metrics
- FlagScale Launcher log structure: when you launch with `flagscale train`, logs go to `<experiment_dir>/logs/`. Key locations:
  - `<experiment_dir>/logs/details/<host>/<timestamp>/<run>/<attempt>/<rank>/stdout.log` — per-rank training output
  - `<experiment_dir>/logs/scripts/host_*_run.sh` — generated launch scripts (inspect to debug launch issues)
  - `<experiment_dir>/launch.log` — launcher-level output (errors before training starts)
  After launching, use `find_latest_log` to locate the correct stdout.log. Do NOT guess paths or use `tail -f` on arbitrary files — the timestamp directory changes every launch. If `find_latest_log` returns nothing, check `launch.log` for launcher-level errors first.
- NEVER use `sleep N && tail` or `sleep N && cat` to wait for output. Instead:
  - `timeout N tail -f logfile` — streams output as it appears, exits when timeout expires
- For checking if a process is running: `pgrep -fa <pattern>` (instant, no sleep needed)
- Config path validation: when editing data_path, vocab_file, merge_file, or checkpoint paths in YAML/JSON configs, verify the target paths exist (`ls -la`) BEFORE launching training. Check for placeholder values like '/path/to/', 'FIXME', 'TODO', '/data/dataset' that indicate unresolved templates.

## Experiment registry

Every experiment MUST be recorded in workspace_state (section "Experiments") as a structured log. This is not just for finding logs — it's the knowledge base that prevents repeating mistakes and accelerates iteration. Training infra work IS experiment work — if experiments aren't recorded, the work has no lasting value.

**This is a HARD GATE: do NOT launch any training run (reproduction OR migration) without first writing the experiment entry in workspace_state.** If you find yourself about to run `flagscale train`, `torchrun`, or any training script and haven't written the entry yet — STOP and write it first.

**One experiment, one directory.** Never reuse an experiment directory for a different purpose, different config, or different run. Each launch gets its own directory with a descriptive name (e.g., `bagel_tp4_pp1_reproduce_v1`, `qwen3_tp2_migration_loss_check`). If re-running after a fix, create a new directory (append `_v2`, `_v3`, or timestamp). Mixing logs from different runs in the same directory makes results unverifiable.

**Required fields for each experiment:**

```
### <exp_name> (<status: running|completed|failed|abandoned>)
- **Purpose**: Why this experiment exists. What question are we answering?
- **Hypothesis**: What we expect to happen and why.
- **Config**: Key config choices (TP/PP/DP, batch size, seq_len, precision, special flags).
- **Dir**: Full experiment directory path.
- **Result**: Final metrics (loss, throughput, MFU), or failure mode if it failed.
- **Reflection**: What we learned. What to do differently next time. Root cause if failed.
- **Next**: What experiment follows from this one's results.
```

**Lifecycle:**
1. BEFORE launching: write Purpose, Hypothesis, Config, Dir (status=running)
2. AFTER completion/failure: fill in Result, Reflection, Next (update status)
3. When starting the next experiment: reference the previous one's Reflection

This creates a chain of reasoning across experiments. When a new session starts, reading the experiment log tells you exactly where things stand, what was tried, what worked, and what to try next — no need to re-analyze logs or ask the user to repeat context.

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

## Shell command essentials

- Use `conda run -n <env> <command>`, never `conda activate`. Never install into base env.
- Never `find /` — scope to working directory. Never `sleep N && command` — use `find_latest_log` or direct `tail`.
- When using `find`, ALWAYS exclude conda environments and site-packages: `find <path> -name "*.py" -not -path "*/envs/*" -not -path "*site-packages*" -not -path "*__pycache__*"`. Unfiltered `find` on a workspace with conda environments floods the context with thousands of irrelevant files.
- When grep/awk returns empty on first try, check the actual file format (head -5) before retrying with a different pattern. Don't blindly guess patterns.
- Use `read_file` to read source code, not `sed -n` or `cat`. Read whole files or complete classes/methods — don't read 30-line fragments piecemeal. Fragmented reading leads to fragmented understanding.
- For stable training: prefer `wait <PID>` over repeated sleep-check loops. Get the launcher PID, then `wait $PID` to block until it exits. Use `find_latest_log` or `tail -f` for log monitoring in parallel.
- Process lifecycle: after `pkill`, ALWAYS verify the process is dead (`pgrep -f <pattern>` returns empty) before proceeding. If cleaning up log files, verify no process is still writing to them. The sequence is: kill → verify dead → clean files → relaunch. Skipping verification leads to "zombie" processes holding file handles and stale logs that won't be overwritten.
- ALWAYS use FlagScale Launcher (`flagscale train <model> --config <config>`) to launch training. NEVER bypass it with raw `torchrun`, `python -m torch.distributed.launch`, or hand-written launch scripts. The launcher handles experiment directory layout, per-rank log separation, multi-node coordination, config resolution, and clean shutdown. Bypassing it means: (1) no per-rank log files — all output goes to one stream, making multi-GPU debugging impossible; (2) no experiment directory structure — logs from different runs get mixed; (3) no `flagscale train ... --stop` — you're stuck with `pkill`; (4) no config validation — typos in YAML silently pass. If the launcher fails, fix the root cause (config error, missing dependency, stale cache) — do not work around it by writing your own launch script.
- FlagScale launcher caching: `flagscale train --dryrun` generates launch scripts with hardcoded config values. If you modify the config AFTER a dryrun, the cached scripts are STALE. You MUST re-run dryrun to regenerate scripts before launching. Never assume a config edit propagates to previously generated scripts.
- To stop FlagScale training: `flagscale train <model> --config <config> --stop`. Never broad `ps | grep | kill`.
- Before launching ANY training run, verify no old training processes are alive (`pgrep`). If found, kill and wait for GPU memory release before launching. Launching over a live process causes port conflicts, OOM, and corrupted logs.
- Network errors: STOP and tell user to configure proxy. Don't attempt workarounds.
- Before `rm -rf`: check with `ls`/`du -sh` first. Prefer `mv` to trash over delete.
- For detailed shell rules, dependency resolution, training launch discipline, multi-node, and checkpoint resume: load the `ops-discipline` skill."""


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
        self.tool_registry.register(WorkspaceStateTool())
        self._load_plugin_tools()

        memory_dir = os.path.join(Path.home(), ".flagscale", "agent_memory")
        self._session_id = uuid.uuid4().hex[:8]
        self.session_memory = SessionMemory(memory_dir, config.memory_ttl_days)
        self.tool_registry.register(MemoryWriteTool(self.session_memory, self._session_id))
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
        self._refresh_system_prompt()

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
        atexit.register(self._atexit_hook)

    # ── Atexit safety net ───────────────────────────────────────────────

    def _atexit_hook(self):
        """Update workspace state on any exit path (safety net for abnormal exits)."""
        try:
            if self._turn_count >= 2:
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
        """Call LLM to summarize conversation segment being dropped during compaction."""
        messages = [
            {"role": "system", "content": "You are a concise summarizer. Output only the summary, no preamble."},
            {"role": "user", "content": text},
        ]
        response = self.provider.chat(messages, tools=[])
        return response.get("content", "").strip()

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
        "- Log searching with find/ls -R/ls -lt → suggest find_latest_log tool or workspace_state experiment registry\n"
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
        "- Checkpoint saved to disk (torch.save, save_checkpoint) without a reload verification → WARN: verify saved checkpoint by reloading and checking key count, parameter shapes, and total parameter count match expectations\n\n"
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
        "- If the user mentions training configuration, parallelism, or YAML config, select train-config\n"
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

    def _refresh_system_prompt(self, memory_context="", plan_context="", workspace_context=""):
        skills = self.skill_manager.list_skills()
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
            if skills else "(no skills available)"
        )
        prompt = SYSTEM_PROMPT.format(
            skills=skills_text, cwd=os.getcwd(),
            memory_context=memory_context, plan_context=plan_context,
            workspace_context=workspace_context,
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
        """Build memory context string from recent memories."""
        notes = self.session_memory.recent(max_tokens=800)
        if not notes:
            return ""
        lines = []
        for n in notes:
            lines.append(f'[{n.get("type", "?")}] {n.get("content", "")}')
        return "<session-memory>\n" + "\n".join(lines) + "\n</session-memory>"

    def _build_workspace_context(self):
        """Load workspace state file if it exists."""
        state_path = os.path.join(Path.home(), ".flagscale", "workspace_state.md")
        if not os.path.isfile(state_path):
            return ""
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return f"<workspace-state>\n{content[:5000]}\n</workspace-state>"
        except Exception:
            pass
        return ""

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

    _WORKSPACE_STATE_PROMPT = (
        "Summarize the current workspace state based on this session.\n"
        "Focus on: what was worked on, key outcomes, and what's next.\n"
        "Keep it under 500 chars. Use markdown sections: ## Current Task, ## Status, ## Next Steps.\n"
        "If nothing meaningful happened, reply with exactly: SKIP\n\n"
        "Session:\n{summary}"
    )

    def _auto_update_workspace_state(self):
        """Auto-update workspace state at session end."""
        if self._turn_count < 2:
            return
        user_msgs = []
        for m in self.history.messages:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                user_msgs.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                        user_msgs.append(block["text"])
                        break
        if not user_msgs:
            return
        recap_parts = []
        for msg in user_msgs[-5:]:
            line = msg.strip().replace("\n", " ")
            if len(line) > 120:
                line = line[:117] + "..."
            recap_parts.append(line)
        summary = " | ".join(recap_parts)
        try:
            prompt = self._WORKSPACE_STATE_PROMPT.format(summary=summary)
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.chat(messages, tools=[])
            text = (response.get("content") or "").strip()
            if text == "SKIP" or len(text) < 10:
                return
            ws_tool = self.tool_registry.get("workspace_state")
            if ws_tool:
                ws_tool.execute(action="write", content=text)
        except Exception as e:
            logger.debug("Workspace state auto-update skipped: %s", e)

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

    _TRAIN_CMD_RE = re.compile(r'flagscale\s+train|torchrun|python.*train')
    _TRAIN_FAIL_RE = re.compile(
        r'ERROR|FATAL|Traceback|NCCL|OOM|OutOfMemory|RuntimeError|'
        r'TERMINATED|STALLED|KeyError|ModuleNotFoundError|ImportError|'
        r'torch\.cuda\.OutOfMemoryError|CUDA error',
        re.IGNORECASE,
    )

    def _track_training_failures(self, tool_calls, results):
        """Track consecutive training failures and inject escalation warning."""
        for tc, result in zip(tool_calls, results):
            if tc["name"] != "shell":
                continue
            cmd = tc["arguments"].get("command", "")
            if not self._TRAIN_CMD_RE.search(cmd):
                continue
            if self._TRAIN_FAIL_RE.search(result[:2000]):
                self._consecutive_train_failures += 1
                reason = result[:200].split('\n')[0]
                self._last_train_failure_reasons.append(reason)
                if self._consecutive_train_failures >= 3:
                    escalation = (
                        f"\n\n[ESCALATION] {self._consecutive_train_failures} consecutive training failures detected. "
                        f"STOP and report to the user. Summarize all attempts and failures before continuing.\n"
                        f"Recent failure reasons:\n"
                    )
                    for i, r in enumerate(self._last_train_failure_reasons[-5:], 1):
                        escalation += f"  {i}. {r}\n"
                    self.history.append({
                        "role": "user",
                        "content": escalation,
                    })
            else:
                self._consecutive_train_failures = 0
                self._last_train_failure_reasons.clear()

    def _run_poll_mode(self, command, last_output, tool_call_id):
        """Execute poll loop locally without LLM calls. Returns (output, count, elapsed, reason)."""
        interval = self.config.poll_interval
        max_duration = self.config.poll_max_duration
        cmd_display = self._shell_display_summary(command, max_len=70)
        display.poll_mode_start(cmd_display, interval)

        poll_count = 0
        start = time.time()
        current_output = last_output

        try:
            while True:
                elapsed = time.time() - start
                if elapsed >= max_duration:
                    return current_output, poll_count, elapsed, "timeout"

                time.sleep(interval)
                poll_count += 1
                elapsed = time.time() - start

                try:
                    new_output = self.tool_registry.execute("shell", command=command, _skip_confirm=True)
                except Exception as e:
                    new_output = f"ERROR: {e}"

                if self._poll_output_changed(current_output, new_output):
                    current_output = new_output
                    return current_output, poll_count, elapsed, "changed"

                display.poll_check(poll_count, elapsed)
                current_output = new_output

        except KeyboardInterrupt:
            elapsed = time.time() - start
            return current_output, poll_count, elapsed, "interrupted"

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

            display.thinking()

            try:
                response, usage = self._call_llm_stream(messages, schemas)
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break
            except Exception as e:
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

                new_output, poll_count, poll_elapsed, reason = self._run_poll_mode(
                    command, last_output, tc["id"])
                display.poll_mode_end(reason, poll_count, poll_elapsed)

                self._replace_last_tool_result(
                    self.provider.format_tool_result(tc["id"], new_output))
                self._recent_iters.clear()

            print()  # visual gap after tool block

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

        stream = retry_with_backoff(
            lambda: self.provider.chat_stream(messages, schemas),
            max_retries=3,
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

        return results

    def _execute_tool(self, tool_call, skip_confirm=False):
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        def _fmt_arg(k, v):
            s = str(v)
            # Truncate long content for display (write_file, edit_file, etc.)
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

        # For shell commands, show a clean one-line summary instead of the full command
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
        detail = ""
        if tool_name == "shell":
            annotations = self._result_judge(arguments.get("command", ""), result, elapsed)
            annotations = self._dedup_annotations(annotations)
            if annotations:
                header = "\n".join(f"[{a}]" for a in annotations)
                result = header + "\n" + result
        if error and result:
            raw = result.split('\n')[0].replace("ERROR:", "").strip()
            detail = (raw[:57] + "...") if len(raw) > 60 else raw
        display.tool_done(tool_name, elapsed, detail=detail, error=error)
        return result

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