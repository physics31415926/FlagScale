"""ReAct agent — the core loop."""

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
from flagscale.agent.react.cache import KnowledgeCache
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.cost import CostTracker
from flagscale.agent.react.history import HistoryManager
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
from flagscale.agent.react.tools.cache_write import CacheWriteTool
from flagscale.agent.react.tools.cache_read import CacheReadTool
from flagscale.agent.react.tools.find_log import FindLatestLogTool
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
- read_file / write_file / edit_file: File operations
- shell: Execute shell commands
- web_fetch: Fetch URL content (docs, GitHub pages, error references)
- load_skill: Load specialized skill instructions
- cache_write / cache_read: Persist and retrieve project knowledge across sessions
- memory_write / memory_read: Persist and retrieve key findings, decisions, and todos across sessions
- find_latest_log: One-shot locate and display the latest training log for an experiment
- plan_create / plan_update / plan_status: Task planning for complex multi-step work

Skills available:
{skills}

To activate a skill, call load_skill with the skill name. When a user asks what you can do, list ALL available skills above.
Skill loading results:
- SUCCESS: Skill loaded. Read and follow the returned instructions — they contain domain-specific knowledge (customized dependency versions, correct config formats, etc.) that you should not skip.
- ERROR: Skill failed to load. Show the EXACT error message to the user and ask how to proceed. Do NOT silently fall back to your own approach without telling the user what went wrong.

Working directory: {cwd}

## Core principles

1. ACTION FIRST. When the task is clear, execute immediately. Don't list options, don't ask for confirmation, don't explain what you're about to do — just do it. Report results after, not plans before.
   - User says "install dependencies" → run pip install, not "I'll run pip install, shall I proceed?"
   - User says "continue the task" and memory shows what's pending → pick up where it left off
   - User says "train Qwen3-0.6B" → load the skill, generate config, launch training

2. CONCISE COMMUNICATION. Talk like an engineer in a terminal, not a chatbot.
   - Status updates: one line. "Dependencies installed, starting training." not a paragraph.
   - Errors: state what failed, what you'll try next. No apologies, no filler.
   - Results: key metrics and outcome. Skip the narrative.
   - NEVER list what you're "about to do" in bullet points. Just do it.

3. PARALLEL EXECUTION. Run independent commands simultaneously. Check environment + check logs + check processes = one round, not three. Maximize throughput.

4. KNOW WHEN TO ASK vs ACT.
   ASK when: genuinely ambiguous (multiple valid approaches with real tradeoffs), destructive and irreversible, user's intent is unclear, choosing model size or data source for training/verification.
   ACT when: task is clear from context + memory, there's an obvious next step, recovering from a known error pattern, continuing interrupted work.
   Rule of thumb: if you can infer the right action from the user's words + memory + current state, act. Don't ask just to be safe.
   MULTI-QUESTION RULE: if you asked the user multiple questions and only got answers to some of them, you MUST follow up on the unanswered questions before proceeding. Never assume defaults for unanswered questions — the user may have skipped them unintentionally. Repeat the unanswered questions and wait for responses.
   Specific cases that ALWAYS require asking:
   - Model size selection: if multiple sizes/configs exist and user didn't specify, list options with parameter counts and recommend the smallest, but let user choose.
   - Data preparation: if data needs downloading, present options (existing demo data / smallest real subset with size / synthetic data) and let user choose. NEVER start downloading multi-GB datasets without confirmation.
   - Download speed issues: if a download is running < 500KB/s for a large file, stop and ask user whether to continue, try a mirror, or provide data manually.

5. PROACTIVE PROBLEM DETECTION. When you discover something wrong (bad config values, resource conflicts, missing files, OOM risks), flag it immediately and fix it if the fix is safe. Don't silently work around problems, but also don't stop everything to write an essay about it. "Found TP=8 causes OOM on 80GB GPUs for this model size, switching to TP=4 PP=2." — then do it.

   FAIL-FAST PRINCIPLE: Before running any operation that takes >30 seconds (model loading, training launch, large install), do a lightweight pre-check first:
   - Model loading: verify state_dict keys and shapes match BEFORE loading weights into GPU
   - Checkpoint conversion: compare key counts and shapes between source and target BEFORE running full conversion
   - Training launch: validate config (vocab size, hidden size, parallelism settings) BEFORE launching
   - Environment install: check version constraints BEFORE running pip install
   This avoids the costly loop of "wait 5 minutes → fail → fix → wait 5 minutes again".

6. INFRA EXPERTISE. You understand:
   - GPU training: parallelism strategies (TP/PP/DP/EP/CP), memory optimization, NCCL, mixed precision
   - Environment management: conda, pip, CUDA toolkit, driver compatibility
   - FlagScale specifics: config structure, launcher, checkpoint management, logging
   - Common failure modes: OOM, NCCL timeouts, dependency conflicts, network issues
   Use this knowledge to make smart defaults and catch problems early.

7. PLAN COMPLEX WORK. For multi-step tasks (environment setup, model porting, training runs), create a plan first with plan_create. Update progress as you go with plan_update. When things go wrong, replan rather than improvise. Check plan_status when resuming work. Simple tasks (single command, quick lookup, small edit) don't need a plan — just do them.

8. REPRODUCTION vs VERIFICATION — know the difference.
   When the user says "reproduce" (复现), it means STRICT REPRODUCTION:
   - Purpose: reproduction of open-source implementations serves as the BASELINE for migrating to FlagScale. If the baseline is wrong, everything built on top of it is meaningless. Treat reproduction with the highest rigor.
   - Core principle: classify every parameter into IMMUTABLE vs ADAPTABLE before touching anything.
     IMMUTABLE (define the experiment — changing any of these means it's no longer the same experiment):
       model architecture, tokenizer/vocab, optimizer & LR schedule, loss function, data processing pipeline, preprocessing logic, special tokens, evaluation protocol
     ADAPTABLE (hardware mapping — changing these preserves the experiment on different hardware):
       num_nodes, num_gpus, batch_size + accum_grad (must maintain same effective batch size), data parallelism strategy, num_workers, logging/checkpoint intervals
   - Checkpoint saving: unless the user specifies otherwise, ensure exactly ONE checkpoint is saved — at the final step. Set save_every = total_steps so only the last checkpoint is written. This avoids wasting disk/time on intermediate saves while still preserving the training result. If the framework doesn't auto-save at the end, explicitly set the interval to trigger on the last step.
   - If you're unsure whether a parameter is immutable or adaptable, treat it as immutable and ask the user.
   - If an immutable parameter conflicts with the current setup (e.g., data too small for the original vocab size), STOP and explain the conflict. Let the user decide — never silently adjust.
   - Reuse original artifacts: tokenizers, vocab files, pretrained weights, and config files should be extracted from the original release (official repo, model hub, checkpoints), not regenerated. Regenerating on different data produces different artifacts even with the same settings.
   - Data: use the original dataset if possible. If using a subset for speed, only the data VOLUME is reduced — the format, processing pipeline, and all immutable parameters stay identical.
   - File formats: ALWAYS examine a working reference before generating any file. Read an existing example of the expected format, then replicate it exactly.

   When the user says "verify" or "test" (验证/测试), it means QUICK VERIFICATION:
   - Goal is to confirm the pipeline runs without errors, not to match original results.
   - Immutable parameters MAY be relaxed, but still ASK the user before making significant changes.

   If the user's intent is ambiguous, ASK: "Do you want strict reproduction (same configs, just fewer GPUs/data) or quick verification (smaller model, synthetic data, just to test the pipeline)?"

## Planning discipline

- For environment setup or framework installation tasks, the plan MUST start with a dedicated constraint collection step BEFORE any installation step. This step should:
  (a) Check hardware: driver version → max CUDA version
  (b) Fetch the target framework's dependency specs (setup.py, setup.cfg, pyproject.toml) — via web_fetch or shallow clone, NOT by installing
  (c) Read training recipes/configs to identify additional requirements
  (d) Solve the constraint intersection and write out the decision (Python version, PyTorch version + CUDA variant, framework version)
  Only AFTER this step is marked done should the install step begin — and the install step should execute the solved plan with pinned versions, no guessing.
- NEVER combine "analyze" and "install" into one plan step. "Create conda env and install PyTorch 2.4.0" is wrong if you haven't verified 2.4.0 is the right version yet. The correct split: Step 1 "Collect constraints and solve versions" → Step 2 "Create env and install with pinned versions".
- Same principle applies to model porting: first analyze the source model code and architecture, then generate configs and conversion code.

## Memory discipline

- IMMEDIATELY record: what the user is working on (context), key findings (finding), decisions made (decision), unfinished work (todo).
- Don't wait until end of conversation. Write memory as you go.
- One fact per entry, under 200 chars. Descriptive keys: 'qwen3_train_result', 'tp_oom_fix', 'todo_install_deps'.
- Focus on information that is HARD TO RE-DERIVE: file paths, environment paths, credentials locations, version numbers, error messages, user preferences. Don't record things that can be re-read from code or config files.
- For model porting / training tasks, ALWAYS record:
  - Model weight paths (e.g. '/data/models/BAGEL-7B-MoT')
  - Working directories and environment paths (e.g. conda env name, reproduce dir)
  - Component analysis results: which components are SUPPORTED/ADAPTABLE/CUSTOM
  - Key numerical results: loss values, alignment diffs, throughput numbers
  - Blockers and workarounds discovered
  - What has been verified vs. what is still pending
- When a plan step produces a concrete artifact (checkpoint, config file, test result), record its path and status in memory immediately — don't assume the plan context alone is sufficient.

## Knowledge caching

- After analyzing project structure, configs, or dependencies, cache the result for future sessions.
- Before reading project files for structural questions, check if cached knowledge exists in <project-knowledge> tags.
- Good keys: 'env_dependencies', 'project_structure', 'model_configs', 'parallel_strategies'.
- Include source file paths so cache auto-invalidates on file changes.

## Task planning

- When a plan exists, ALWAYS check plan_status at the start of each turn to know where you left off.
- Mark steps done as you complete them — don't batch updates.
- If a step fails or the approach changes, use plan_update with action=add_steps to adjust.
- Plans persist across sessions. On resume, the plan tells you exactly what's done and what's next.

{plan_context}
{memory_context}
{cache_context}

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
- `/cache` — show cached project knowledge
- `/plan` — show current task plan status
- `/plan list` — list all plans (including history)
- `/plan abandon` — abandon the current plan
- `/plan clear` — clear completed/abandoned plans
- `/reload` — reload skills and config
- `/quit` — exit the agent

## Shell command rules

- NEVER search from root (`find /`). Scope to working directory or known paths.
- Prefer `grep -rn "pattern" . --include="*.py"` for code search.
- Use `ls`, `tree -L 2`, `find . -maxdepth 2` to explore structure.
- Use `head`/`tail` ONLY for quick previewing when you don't need the full output (e.g., glancing at a file's first few lines). If you need the complete output to analyze, diagnose, or make decisions, NEVER truncate — no matter how large. A truncated error log is useless; a full one tells you exactly what went wrong. Ask yourself: "do I need all of this to do my job?" If yes, get all of it.
- To run commands in a conda environment, ALWAYS use `conda run -n <env> <command>`. NEVER use `conda activate` or `source activate` in shell commands — they don't work in non-interactive shells and will silently use the wrong Python.
- NEVER install packages into the base or current environment unless the user explicitly asks to. When the user says "create a new environment" or the task involves setting up dependencies, ALL pip/conda install commands MUST target the new environment (`conda run -n <env> pip install ...`). Polluting an existing environment is hard to reverse and may break other projects. Even for "just checking" a package's version requirements, use `pip index versions <pkg>` or read setup.cfg/pyproject.toml from the source repo instead of installing into the current environment.
- To stop FlagScale training: prefer `flagscale train <model> --config <config> stop`. Fallback: `cat outputs/<exp>/logs/pids/* | xargs kill -9`. NEVER use broad `ps | grep | kill` — it will kill the agent.
- Network errors (DNS failure, connection timeout, etc.): STOP and tell user to configure proxy in ~/.flagscale/agent.yaml under shell_env. Don't attempt workarounds.
- NEVER use `sleep N && <command>` or `sleep N; <command>` for monitoring. This wastes time and blocks the agent.
  Instead: (a) for FlagScale training, use `find_latest_log` tool, (b) for other frameworks, directly read the log file with `tail -n 50 <logfile>`,
  (c) `timeout 30 tail -f <logfile>` for time-bounded streaming.
  For periodic monitoring, tell the user to ask again later — do NOT implement polling loops.
- NEVER run the same command twice in a row. If you just ran `ls -lh <dir>` or `mkdir -p <dir>`, do NOT repeat it. If a command produced unclear results, try a DIFFERENT diagnostic command instead.
- Build succeeds without errors → proceed to next step. Don't ask user to confirm successful builds.
- NEVER modify third-party source code to work around build errors. Report the error and let user decide.
- For large file downloads, ALWAYS use resume-capable flags: `wget -c` or `curl -C -`. If a download fails, resume instead of re-downloading.
- For multiple large downloads, execute them as SEPARATE shell commands, not combined with `&` or `&&`. This way if one fails, others are unaffected and you can retry independently.
- After any download, verify the file exists and has a reasonable size with `ls -lh <file>` before proceeding to use it.
- Download speed monitoring: when downloading large files, check the initial speed within the first 10-30 seconds. If speed is abnormally low (< 500 KB/s for a multi-GB file):
  1. First check if proxy is being used — `wget` and `curl` need explicit `--proxy` or env vars. If proxy is configured but not being used, add it.
  2. If proxy is already in use and speed is still very low, STOP the download immediately and tell the user: report the speed, estimated time, and ask whether to continue, try a mirror, or let the user provide the data manually.
  3. NEVER let a multi-hour download run silently. A 6GB file at 30KB/s = 55 hours — this is not acceptable.
- Before `rm -rf` on data or experiment directories, first check what you're deleting with `ls` or `du -sh`. Never blindly delete directories that may contain hours of computation results.

## Operational discipline

### Environment awareness
- FIRST THING on any new server: determine if you are on bare metal or inside a container. Check: `cat /proc/1/cgroup 2>/dev/null`, `/.dockerenv` existence, `hostname`. Record the result in memory.
- In a container, you can ONLY see your own processes. nvidia-smi shows GPU memory/utilization from ALL containers sharing the GPU, but only shows PIDs from YOUR container. So "GPU memory occupied but no process visible" means OTHER containers are using the GPU — NOT zombie memory or leaked memory. Never suggest restarting the container or clearing memory in this case.
- On bare metal, all processes are visible. "GPU memory occupied but no process" is genuinely abnormal.
- Know your privilege level: can you install system packages? Can you restart services? Can you access other nodes via SSH? Don't attempt operations you lack permissions for.
- When reporting GPU status, always clarify the visibility scope: "nvidia-smi shows X% memory used. Since we are in a container, this may include usage from other containers sharing these GPUs."

### Trust nothing, verify everything
- Any "success" claim MUST have timestamp evidence. Log timestamps must be AFTER the current operation started. Old logs from previous runs are NOT proof of current success.
- After launching training, verify ALL four: (1) process exists (`ps aux | grep <exp>`), (2) GPU has load (`nvidia-smi`), (3) stdout.log has fresh output, (4) stderr.log has no errors. All four pass = success. Anything less = investigate.
- Environment verification: `which python`, `python --version`, test critical imports. Don't trust conda env name alone — verify the actual binary path.
- When reading logs, ALWAYS check the timestamp first. If the log is from a previous run, say so explicitly and find the current run's log.

### Training launch discipline
- Training startup is EXPENSIVE (model loading takes minutes to tens of minutes). Every failed launch wastes significant time. Therefore, ALWAYS run a preflight check before launching training:
  1. Output directory exists: `mkdir -p <output_dir>` BEFORE the launch command, never after.
  2. Environment works: run a quick Python smoke test in the SAME way you'll launch training (same conda env, same PYTHONPATH, same working directory):
     `conda run -n <env> env PYTHONPATH=<path> python -c "import torch; import <main_module>; print('OK')"`
     This catches: wrong PYTHONPATH, missing dependencies, numpy/torch version conflicts, CUDA issues — all in seconds instead of after a 10-minute model load.
  3. CLI args valid: if the framework supports `--help` or `--dry-run`, use it to validate arguments before the real launch.
  4. After `pip install` or `conda install`, ALWAYS re-verify critical imports (`torch`, `numpy`, `transformers`, etc.) before launching training. Package managers often upgrade/downgrade transitive dependencies silently.
  5. Parameter summary: Before launching, present a clear summary of ALL training parameters to the user. For each parameter, state its value and source — one of:
     - "from original repo/script" (the open-source code's default or example config)
     - "from paper" (explicitly stated in the paper, cite section/table if possible)
     - "from user instruction" (the user explicitly requested this value)
     - "agent decision: <reasoning>" (you chose this value — explain why, e.g. "reduced from 8 to 1 for quick validation", "matches GPU count")
     This makes parameter choices transparent and auditable. If you cannot trace a parameter's origin, say so explicitly — never silently guess.
     If any parameter is uncertain (no clear source, or your reasoning is speculative), AND mode is "confirm", ask the user to confirm before launching. In "auto" mode, proceed with your best judgment.
- When training fails, ALWAYS read the FULL error log before retrying. `tail -50` is not enough — read the complete stderr/stdout to find the root cause. Multiple errors may exist; fix ALL of them before relaunching.
- NEVER retry training more than once without a clear diagnosis. Pattern: (1) read full logs, (2) identify root cause, (3) fix it, (4) retry. If the second attempt also fails, STOP and report the full error to the user.
- When launching background training (`&`), ALWAYS wait long enough for the process to either start producing output or crash. Check both stdout AND stderr immediately after the wait. An empty log after 30s usually means the process crashed during import — read stderr.
- Log isolation: NEVER reuse the same log file path across different training runs. Each launch must write to a unique path with a timestamp, e.g. `stdout_$(date +%Y%m%d_%H%M%S).log`. This prevents overwriting previous run's logs and makes it easy to compare runs.

### Understand target state before acting
- Before creating files, directories, symlinks, or configs, FIRST understand the expected structure by reading documentation or examining a working example. Don't guess and fix iteratively.
- Concrete pattern: when setting up a project that expects a specific directory layout (e.g., ESPnet egs2 recipes, Megatron examples), FIRST run `ls -la` on a working reference to see what files exist, which are symlinks, and where they point. Then replicate the structure in one pass.
  Example — instead of:
    `ln -sf X utils` → broken → `rm utils` → `ln -sf Y utils` → wrong target → `rm utils` → `ln -sf Z utils`
  Do this:
    `ls -la /tmp/espnet/egs2/librispeech/asr1/` → see that utils→../../TEMPLATE/asr1/utils, s2t.sh→../TEMPLATE/s2t1/s2t.sh, etc. → replicate all links correctly in one command block.
- This is the same principle as dependency resolution: collect all constraints (what the target state should look like), then execute once. Applies to: directory structures, config files, environment variables, symlink trees, Makefile targets.
- File format generation: NEVER generate a file format (token_list, data manifests, config YAML, etc.) from assumptions or documentation alone. ALWAYS find and read an existing working example of that exact file format first, then replicate it. E.g., before generating a token_list, find one from a completed ESPnet experiment and `head -20` it to see the exact format.

### Dependency chain awareness
- When skipping/removing ANY component (Apex, FlashAttention, etc.), IMMEDIATELY scan configs for parameters that depend on it and disable them. Example: skip Apex → set `gradient_accumulation_fusion: false`.
- After installing a component, verify its runtime dependencies exist (CUDA version match, shared libraries, etc.).
- After modifying config files, check if the build system caches old configs. FlagScale/Hydra caches generated scripts in `outputs/<exp>/`. Either clean `outputs/<exp>/hydra/` and `outputs/<exp>/logs/scripts/` or use `--dryrun` to verify the new config takes effect.
- Think in dependency chains: A depends on B depends on C. If C changes, trace the impact forward.
- CUDA/cuDNN version conflicts: system LD_LIBRARY_PATH often contains an older cuDNN than PyTorch expects. Fix: prepend PyTorch's bundled nvidia/cudnn/lib to LD_LIBRARY_PATH. Find it with: `python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__)"`. Create a wrapper script (e.g. fix_env.sh) that exports the correct LD_LIBRARY_PATH before running training commands.

### Dependency resolution — constraint solving, then one-shot install
- NEVER blindly install packages and then fix version conflicts after the fact. The goal is to install ONCE and get it right.
- Treat environment setup as a constraint satisfaction problem. Collect ALL constraints first, solve for compatible versions, then execute the install plan in one pass.

**Phase 1: Collect constraints** (NO installs in this phase)
  1. Hardware constraint: run `nvidia-smi` → get driver version → look up max supported CUDA version
     - Driver 535.x → CUDA ≤ 12.4
     - Driver 550.x → CUDA ≤ 12.4
     - Driver 560.x → CUDA ≤ 12.6
     - Driver 570.x → CUDA ≤ 12.8
  2. Framework constraint: clone the repo (or `web_fetch` its setup.cfg/pyproject.toml from GitHub raw URL) → extract PyTorch version bounds, Python version bounds, and other key dependencies
  3. Recipe/config constraint: read the specific training recipe/config to check if it needs additional packages (flash-attn, deepspeed, apex, etc.) and their version requirements
  4. PyTorch ↔ CUDA constraint: PyTorch wheels are built for specific CUDA versions (cu118, cu121, cu124). The CUDA version in the wheel must be ≤ the driver's max supported CUDA version

**Phase 2: Solve** (find the intersection — present options to user)
  - Write out the constraint table explicitly before deciding:
    ```
    Driver 535 → max CUDA 12.4
    Framework requires torch >= 2.3.1
    PyTorch 2.3.1 available as: cu118, cu121 → all ≤ 12.4 ✓
    PyTorch 2.4.0 available as: cu118, cu121, cu124 → all ≤ 12.4 ✓
    PyTorch 2.5.1 available as: cu118, cu121, cu124 → all ≤ 12.4 ✓
    Framework requires Python >= 3.10
    ```
  - Present the viable options to the user and recommend one:
    ```
    Options:
      A. torch==2.3.1+cu121 — lowest compatible, best stability with third-party libs (recommended)
      B. torch==2.4.0+cu124 — newer, more CUDA features
      C. torch==2.5.1+cu124 — latest, but flash-attn/apex may not be adapted yet
    Python: 3.11. Which do you prefer?
    ```
  - DEFAULT RECOMMENDATION: prefer the lowest PyTorch version + highest compatible CUDA variant. Reason: lower PyTorch = fewer breaking changes with third-party deps; higher CUDA = better GPU features. But always let the user make the final call.
  - If no valid intersection exists, STOP and tell the user (e.g., framework requires torch>=2.6 but driver only supports CUDA 12.4 and no compatible wheel exists)

**Phase 3: One-shot install** (execute the solved plan)
  1. `conda create -n <env> python=<solved_version> -y`
  2. `conda run -n <env> pip install torch==<solved_version> --index-url https://download.pytorch.org/whl/<solved_cu>`
  3. `conda run -n <env> pip install <framework>` (if it tries to upgrade torch, use version pins or `--no-deps` + manual deps)
  4. Verify: `conda run -n <env> python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

- When pip install upgrades a critical package (PyTorch, numpy) unexpectedly, use `--no-deps` or version pins to prevent it.
- For conda environments: prefer `pip install --no-deps` + manual dependency resolution over letting pip freely resolve, especially when CUDA-specific wheels are needed.

### Active monitoring, not passive waiting
- After launching a long task, IMMEDIATELY check stdout + stderr + system state (GPU/CPU/memory) in PARALLEL. Don't wait for the first check interval.
- NEVER use `sleep N && tail`. Instead, use `find_latest_log` tool or direct `tail` on the known log path. For bounded waiting, use `timeout 30 tail -f <logfile>` — it auto-exits after 30s.
- If a tool execution takes abnormally long (e.g., a simple `rm` takes minutes), flag it to the user — something is wrong (deleting huge checkpoint dirs, filesystem issues, etc.).

### Checkpoint resume
- When training is interrupted, check if checkpoints exist: `ls outputs/<exp>/checkpoints/`. FlagScale auto-saves checkpoints at configured intervals.
- To resume: keep the same YAML config and re-launch. FlagScale loads the latest checkpoint automatically if `outputs/<exp>/checkpoints/` is non-empty.
- If the user wants to start fresh, delete or rename the checkpoint directory FIRST (with confirmation).
- After resuming, verify the starting iteration in stdout.log matches the checkpoint iteration, not 0.

### Multi-node awareness
- Multi-node training requires: SSH passwordless access between nodes, consistent environment (same conda env, same FlagScale version), shared or replicated data paths, and correct NCCL environment variables.
- Key NCCL env vars: `NCCL_IB_DISABLE` (set 1 if no InfiniBand), `NCCL_SOCKET_IFNAME` (network interface), `NCCL_DEBUG=INFO` (for debugging).
- Hostfile format: one line per node, `<hostname> slots=<num_gpus>`. Verify SSH connectivity to all nodes before launching.
- When diagnosing multi-node failures, check logs on ALL nodes — the root cause is often on a different node than the one that reported the error.

### Destructive operation safety
- Before `rm -rf` on any directory: first `ls` to see contents, then `du -sh` to see size. If it contains checkpoints or data > 1GB, ask the user for confirmation.
- Never delete `outputs/<exp>/` directories without checking if they contain checkpoints the user might need.
- When cleaning up failed runs, prefer `mv` to a trash directory over `rm -rf`.

### One-shot diagnosis
- FlagScale log path pattern: `outputs/<exp>/logs/details/host_*/TIMESTAMP_DIR/default_*/attempt_0/0/stdout.log` (and stderr.log). Use `find_latest_log` tool to locate in one step.
- Error diagnosis order: stderr FIRST (crash/exception), then stdout tail (last progress), then full stdout only if needed.
- When a training run fails, check stderr → identify the error → fix root cause → clean stale outputs → retry. Don't retry without understanding the failure."""



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
        self._load_plugin_tools()

        cache_dir = os.path.join(Path.home(), ".flagscale", "agent_cache")
        self.knowledge_cache = KnowledgeCache(cache_dir, config.cache_ttl_days)
        self.tool_registry.register(CacheWriteTool(self.knowledge_cache))
        self.tool_registry.register(CacheReadTool(self.knowledge_cache))

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
        self._refresh_system_prompt()

        self._turn_count = 0
        self._session_start = time.time()
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._cost_tracker = CostTracker(config.model, config.max_cost, config.pricing)
        self._loaded_skills = set()
        self._interrupted = False
        self._streaming_in_code_block = False

    # ── Unified command health judge ─────────────────────────────────────

    _HEALTH_JUDGE_PROMPT = (
        "You are monitoring a running shell command. Analyze its status and decide "
        "whether it should continue or be terminated.\n\n"
        "Command: {command}\n"
        "Total elapsed: {elapsed}\n"
        "Output changed since last check: {output_changed}\n"
        "Consecutive checks with no output change: {stall_count}\n"
        "Recent output:\n{output}\n\n"
        "Consider these scenarios:\n"
        "- Download stuck (progress not advancing, same percentage/bytes for multiple checks)\n"
        "- Network error (DNS failure, connection refused, timeout)\n"
        "- Process hung (no meaningful progress)\n"
        "- Repeated errors (same error message appearing over and over)\n"
        "- Legitimate long operation (compiling, decompressing, GPU computation, large install)\n"
        "- Download actively progressing (percentage/bytes increasing)\n"
        "- No output at all (command may contain embedded sleep, or process crashed silently)\n\n"
        "Also decide when to check next. Guidelines:\n"
        "- Just started / no output yet: check soon (30-60s) to catch early failures\n"
        "- Actively changing output (loading, downloading): moderate interval (60-120s)\n"
        "- Stable long-running operation (training, large compile): longer interval (180-300s)\n"
        "- Suspected issue (stalled, errors appearing): check soon (30-60s)\n\n"
        "Reply with ONLY a JSON object: "
        "{{\"kill\": true/false, \"reason\": \"one-line explanation\", "
        "\"next_check_seconds\": <integer 30-300>}}"
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
        "- Training launch (flagscale/torchrun/deepspeed) → remind to verify GPU utilization and logs\n"
        "- Package install success (pip/conda) → remind to verify runtime compatibility\n"
        "- pip upgraded/downgraded a critical package (torch, numpy, etc.) → WARN that this may break CUDA compatibility\n"
        "- Long duration (>2min) for simple commands → flag as unexpected\n"
        "- OOM (out of memory) → suggest reducing batch size, enabling gradient checkpointing, or adjusting parallelism\n"
        "- NCCL errors → suggest checking network config, NCCL env vars, and multi-node connectivity\n\n"
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

    def _refresh_system_prompt(self, cache_context="", memory_context="", plan_context=""):
        skills = self.skill_manager.list_skills()
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
            if skills else "(no skills available)"
        )
        prompt = SYSTEM_PROMPT.format(skills=skills_text, cwd=os.getcwd(), cache_context=cache_context, memory_context=memory_context, plan_context=plan_context)

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

    def _inject_context(self, user_input):
        """Auto-inject cached knowledge, session memory, and plan context into the system prompt."""
        memory_context = self._build_memory_context()

        entries = self.knowledge_cache.query(user_input)
        cache_context = ""
        if entries:
            hints = []
            for e in entries[:3]:
                hints.append(
                    f'<project-knowledge key="{e["key"]}">\n{e["content"]}\n</project-knowledge>'
                )
            cache_context = "\n\n".join(hints)

        plan_context = self.task_plan.context_for_prompt()

        # Complexity judge: suggest planning for complex tasks
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

        self._refresh_system_prompt(cache_context=cache_context, memory_context=memory_context, plan_context=plan_context)

    def _handle_cache_command(self, user_input):
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /cache list | /cache clear | /cache delete <key>")
            return
        sub = parts[1]
        if sub == "list":
            entries = self.knowledge_cache.list_entries()
            if not entries:
                print("No cache entries.")
                return
            for e in entries:
                status = display.green("valid") if e["valid"] else display.yellow("stale")
                print(f"  {e['key']}: {e['description']} [{status}, {e['sources']} sources]")
        elif sub == "clear":
            count = self.knowledge_cache.clear()
            print(f"Cleared {count} cache entries.")
        elif sub == "delete":
            if len(parts) < 3:
                print("Usage: /cache delete <key>")
                return
            key = parts[2]
            if self.knowledge_cache.delete(key):
                print(f"Deleted cache entry '{key}'.")
            else:
                print(f"No cache entry '{key}' found.")
        else:
            print("Usage: /cache list | /cache clear | /cache delete <key>")

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
        """Build extra banner lines showing available memory/cache summaries."""
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
        cache_entries = self.knowledge_cache.list_entries()
        if cache_entries:
            hints.append(f"Cache: {len(cache_entries)} entries (/cache list)")
            for e in cache_entries[:3]:
                key = e.get("key", "?")
                desc = e.get("description", "")
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                hints.append(f"  {key}: {desc}" if desc else f"  {key}")
            if len(cache_entries) > 3:
                hints.append(f"  ... and {len(cache_entries) - 3} more")
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
            ["/quit", "/reload", "/skill", "/file", "/save", "/load", "/export", "/cache", "/memory", "/mode", "/plan"],
            sentence=True,
        )
        session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
            style=PromptStyle.from_dict({
                "prompt": "#5fafff bold",   # blue-ish prompt ">"
                "": "#e0e0e0",              # user input in light gray
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
            elif cmd == "/cache":
                self._handle_cache_command(user_input)
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
        self._ensure_memory_written()
        self._archive_session()
        self._clear_autosave()
        session_elapsed = time.time() - self._session_start
        cost_str = self._cost_tracker.format_cost()
        display.session_summary(
            self._turn_count, session_elapsed,
            self._session_input_tokens, self._session_output_tokens,
            cost_str,
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
        for msg in user_msgs[:5]:
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

    def _archive_session(self):
        """Archive the current session to disk (no LLM call)."""
        msgs = [m for m in self.history.messages if m.get("role") != "system"]
        if not msgs or self._turn_count == 0:
            return
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
        self._turn_count = turn_count
        self._loaded_skills = set(meta.get("loaded_skills", []))
        self._session_input_tokens = meta.get("input_tokens", 0)
        self._session_output_tokens = meta.get("output_tokens", 0)
        display.autosave_resumed(turn_count)

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

            if self._cost_tracker.budget_exceeded():
                display.budget_exceeded(self._cost_tracker.format_cost())
                break

            remaining = max_iter - iteration
            if remaining == 2:
                self.history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM: You have 2 iterations left before the limit. "
                        "Wrap up your current task — summarize progress and "
                        "remaining steps so work is not lost.]"
                    ),
                })

            display.thinking()
            t0 = time.time()
            messages = self.history.get_messages()

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
            if output_tok:
                turn_output_tokens += output_tok
                self._session_output_tokens += output_tok
            self._cost_tracker.add(input_tok, output_tok)

            cost_str = self._cost_tracker.format_cost()
            display.llm_done(elapsed, input_tok, output_tok, cost_str)

            if self._cost_tracker.budget_warning():
                display.budget_warning(cost_str)

            logger.info("LLM call #%d: %.1fs", iteration + 1, elapsed)

            self.history.append(self.provider.format_assistant_message(response))

            if not response["tool_calls"]:
                break

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

            iteration += 1

            if iteration >= max_iter and not self._interrupted:
                print(f"\n\033[33m⚠ Reached {max_iter} iterations.\033[0m")
                try:
                    answer = input("   Continue? [y/N] (or enter a number to add iterations): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer == "y":
                    max_iter += 10
                elif answer.isdigit() and int(answer) > 0:
                    max_iter += int(answer)
                else:
                    print("   Stopping.")
                    break
                print(f"   Continuing (new limit: {max_iter} iterations).")

        turn_elapsed = time.time() - turn_start
        cost_str = self._cost_tracker.format_cost()
        display.turn_summary(self._turn_count, turn_elapsed, turn_input_tokens, turn_output_tokens, cost_str)
        self._autosave()

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

        display.thinking_clear()

        try:
            for event in stream:
                if event["type"] == "text":
                    text = event["content"]
                    if display._use_color():
                        fence_count = text.count("```")
                        if self._streaming_in_code_block:
                            text = display.cyan(text)
                        elif "```" in text:
                            text = display.render_markdown(text)
                        if fence_count % 2 == 1:
                            self._streaming_in_code_block = not self._streaming_in_code_block
                    sys.stdout.write(text)
                    sys.stdout.flush()
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
            raise
        except Exception as e:
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
        # This prevents other commands' output from overwriting the
        # confirmation prompt while the user is trying to respond.
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

        with ThreadPoolExecutor(max_workers=min(len(to_run), 4)) as pool:
            futures = {
                pool.submit(self._execute_tool, tc, skip_confirm=True): i
                for i, tc in to_run
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
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

        t0 = time.time()
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
        detail = ""
        if tool_name == "shell":
            annotations = self._result_judge(arguments.get("command", ""), result, elapsed)
            if annotations:
                header = "\n".join(f"[{a}]" for a in annotations)
                result = header + "\n" + result
        display.tool_done(tool_name, elapsed, detail=detail)
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
        msgs = [m for m in self.history.messages if m.get("role") != "system"]
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
        path = parts[1].strip() if len(parts) > 1 else f"session_{int(time.time())}.md"
        path = os.path.expanduser(path)

        lines = [f"# FlagScale Agent Session Export\n"]
        lines.append(f"Provider: {self.config.provider} | Model: {self.config.model}\n")
        lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

        for msg in self.history.messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                parts_text = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts_text.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            parts_text.append(f"[Tool: {block.get('name', '')}]")
                        elif block.get("type") == "tool_result":
                            inner = block.get("content", "")
                            if len(inner) > 200:
                                inner = inner[:200] + "..."
                            parts_text.append(f"[Result: {inner}]")
                content = "\n".join(parts_text)

            if role == "user":
                lines.append(f"\n## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"\n## Assistant\n\n{content}\n")
            elif role == "tool":
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"\n> Tool result: {content}\n")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(display.green(f"✓ Exported to {path}"))
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