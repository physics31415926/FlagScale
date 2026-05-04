---
name: ops-discipline
description: Operational discipline rules for FlagScale infrastructure work. Covers understand-before-act, reading strategy, tool-first principle, decision discipline, shell safety, dependency resolution, training launch, monitoring, checkpoint resume, multi-node, and diagnosis patterns.
keywords:
  - shell
  - install
  - dependency
  - training
  - launch
  - monitor
  - checkpoint
  - multi-node
  - environment
  - setup
  - debug
  - diagnosis
  - resume
  - safety
  - reading
  - planning
requires: []
suggests: []
---

# Operational Discipline

These rules govern how you execute infrastructure operations. Follow them strictly.

---

## Understand before act — the #1 rule

This is the most important rule. Violating it causes cascading waste.

- **Complex tasks (model porting, architecture changes, multi-file refactors)**: spend at least 30% of your effort on reading and understanding BEFORE writing any code. Build a mental model first.
- **Read complete files, not fragments.** Using `sed -n '200,250p'` to peek at 50-line slices forces you to re-read the same file many times. Use `read_file` tool to read the full file (or large ranges). One complete read beats ten partial reads.
- **Never declare "I understand" after reading one file.** Complex systems have hidden dependencies. Read ALL related files before forming conclusions.
- **Never make strong assertions from a single data point.** "X is completely broken!" based on one observation → investigate → "actually X is fine" is a waste pattern. Instead: observe → hypothesize → verify → conclude.
- **Before implementing, list what you DON'T know.** If the list is long, keep reading. If it's short and bounded, start implementing.

### Anti-patterns to avoid
- Reading a file 40 times in fragments instead of once completely
- "关键发现！" → "但等等——" loops (premature conclusions that get reversed)
- Starting implementation after reading 20% of the relevant code, then doing 80% of reading during debugging

---

## Reading strategy — minimize re-reads

- **Use `read_file` tool with line ranges**, not `sed`/`cat`/`head`/`tail` through shell. The tool adds line numbers for reference.
- **First read: full file.** Get the complete picture. Note key line numbers.
- **Subsequent reads: targeted ranges.** Use line numbers from the first read.
- **Record key findings in workspace_state** so they survive context compaction. If you discovered that `attn_mask_type="causal"` ignores custom masks, write it down immediately.
- **Never re-read a file you read in the last 5 turns** unless it was modified. Check `<context-summary>` tags first — they contain conclusions from compacted context.

---

## Tool-first principle — use specialized tools over shell

- **`find_latest_log`** for locating training logs — don't manually construct `find` commands with timestamp patterns
- **`parse_training_metrics`** for extracting loss/grad/throughput — don't grep and awk through log files
- **`read_file`** for reading code — don't `cat` or `sed -n`
- **`workspace_state`** for persisting findings — don't rely on memory alone
- **`plan_create` / `plan_update`** for complex tasks — don't keep the plan only in your head

If you find yourself writing a 3-line shell pipeline to extract information, check if a tool already does it.

---

## Decision discipline — no flip-flopping

- **Before choosing between approaches, LIST ALL CONSTRAINTS.** What does the framework support? What are the limits? What has been tried?
- **Evaluate each approach against ALL constraints before picking one.**
- **Once you pick an approach, commit to it.** Run it to completion or to a clear failure signal. Don't abandon at the first difficulty.
- **If an approach fails, record WHY it failed** (in workspace_state or plan notes) before trying the next one. This prevents revisiting failed approaches.
- **Never flip between approaches more than twice.** If you've tried A→B→A, stop and ask the user for guidance.

### The "solve one problem at a time" rule
When debugging, isolate ONE variable at a time:
1. Form a hypothesis about the root cause
2. Design a verification experiment that tests ONLY that hypothesis
3. Run it and interpret the result
4. If confirmed: fix it and verify the fix
5. If refuted: record "not this" and move to the next hypothesis

Do NOT: change three things at once, then wonder which one helped.

---

## Shell command rules

- NEVER search from root (`find /`). Scope to working directory or known paths.
- Prefer `grep -rn "pattern" . --include="*.py"` for code search.
- Use `ls`, `tree -L 2`, `find . -maxdepth 2` to explore structure.
- Use `head`/`tail` ONLY for quick previewing. If you need the complete output to analyze or diagnose, NEVER truncate — a truncated error log is useless.
- To run commands in a conda environment, ALWAYS use `conda run -n <env> <command>`. NEVER use `conda activate` or `source activate` — they don't work in non-interactive shells.
- NEVER install packages into the base or current environment unless explicitly asked. ALL pip/conda install commands MUST target the new environment (`conda run -n <env> pip install ...`).
- To stop FlagScale training: prefer `flagscale train <model> --config <config> --stop`. Fallback: `cat outputs/<exp>/logs/pids/* | xargs kill -9`. NEVER use broad `ps | grep | kill` — it will kill the agent.
- Before launching ANY training run, verify no old training processes are alive (`pgrep`). If found, kill and wait for GPU memory release. Launching over a live process causes port conflicts, OOM, and corrupted logs.
- Network errors: STOP and tell user to configure proxy in ~/.flagscale/agent.yaml under shell_env. Don't attempt workarounds.
- NEVER use `sleep N && <command>` for monitoring. Use `find_latest_log` tool, direct `tail`, or `timeout 30 tail -f <logfile>`.
- NEVER run the same command twice in a row. If results are unclear, try a DIFFERENT diagnostic command.
- Build succeeds without errors → proceed. Don't ask user to confirm successful builds.
- NEVER modify third-party source code to work around build errors.
- For large downloads, ALWAYS use resume-capable flags: `wget -c` or `curl -C -`.
- For multiple large downloads, execute as SEPARATE shell commands, not combined with `&` or `&&`.
- After any download, verify with `ls -lh <file>` before proceeding.
- Download speed monitoring: if < 500 KB/s for a multi-GB file, check proxy first, then STOP and ask user.
- Before `rm -rf` on data or experiment directories, first check with `ls` or `du -sh`. Never blindly delete.

## Environment awareness

- FIRST thing on any new server: `nvidia-smi`, `cat /etc/os-release`, `which conda`, `echo $CUDA_HOME`. Save to workspace_state.
- Check disk space (`df -h`) before large downloads or builds.
- Check GPU memory (`nvidia-smi`) before launching training.
- If the user mentions a conda env, verify it exists (`conda env list`) before using it.

## Training launch discipline

- NEVER launch training without verifying: (1) config is valid, (2) data paths exist, (3) model weights are accessible, (4) GPU count matches config.
- After launch, IMMEDIATELY check logs (within 30 seconds) for startup errors. Don't assume success.
- If training fails within the first 100 iterations, it's likely a config or environment issue, not a training issue.

## Trust nothing, verify everything

- After `pip install X`: verify with `python -c "import X; print(X.__version__)"`.
- After `git clone`: verify the directory exists and has expected content.
- After writing a config file: `cat` it back and verify key values.
- After starting a process: verify it's running with `ps` or check its output.

## Understand target state before acting

- Before creating directory structures, symlinks, or config files: FIRST examine a working example. Don't guess and fix iteratively.
- When setting up a project that expects a specific directory layout, FIRST run `ls -la` on a working reference, then replicate in one pass.
- File format generation: NEVER generate from assumptions alone. ALWAYS find and read an existing working example first.

## Dependency chain awareness

- When skipping/removing ANY component (Apex, FlashAttention, etc.), IMMEDIATELY scan configs for parameters that depend on it and disable them.
- After installing a component, verify its runtime dependencies exist.
- After modifying config files, check if the build system caches old configs. Clean `outputs/<exp>/hydra/` and `outputs/<exp>/logs/scripts/` or use `--dryrun`.
- Think in dependency chains: A depends on B depends on C. If C changes, trace the impact forward.
- CUDA/cuDNN version conflicts: system LD_LIBRARY_PATH often contains an older cuDNN than PyTorch expects. Fix: prepend PyTorch's bundled nvidia/cudnn/lib to LD_LIBRARY_PATH.

## Dependency resolution — constraint solving, then one-shot install

NEVER blindly install packages and fix version conflicts after the fact. Install ONCE and get it right.

**Phase 1: Collect constraints** (NO installs)
1. Hardware: `nvidia-smi` → driver version → max CUDA version
   - Driver 535.x → CUDA ≤ 12.4, 550.x → ≤ 12.4, 560.x → ≤ 12.6, 570.x → ≤ 12.8
2. Framework: clone or `web_fetch` setup.cfg/pyproject.toml → extract PyTorch/Python version bounds
3. Recipe/config: check for additional packages (flash-attn, deepspeed, apex) and their version requirements
4. PyTorch ↔ CUDA: wheel CUDA version must be ≤ driver's max supported CUDA version

**Phase 2: Solve** (find the intersection)
- Write out the constraint table explicitly before deciding
- Present viable options to user and recommend one
- DEFAULT: prefer lowest PyTorch version + highest compatible CUDA variant
- If no valid intersection exists, STOP and tell the user

**Phase 3: One-shot install**
1. `conda create -n <env> python=<solved_version> -y`
2. `conda run -n <env> pip install torch==<version> --index-url https://download.pytorch.org/whl/<cu>`
3. `conda run -n <env> pip install <framework>` (use version pins or `--no-deps` if it tries to upgrade torch)
4. Verify: `conda run -n <env> python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

When pip install upgrades a critical package unexpectedly, use `--no-deps` or version pins.

## Active monitoring

- After launching a long task, IMMEDIATELY check stdout + stderr + system state in PARALLEL.
- NEVER use `sleep N && tail`. Use `find_latest_log` tool or direct `tail`.
- If a tool execution takes abnormally long, flag it to the user.

## Checkpoint resume

- When training is interrupted, check: `ls outputs/<exp>/checkpoints/`. FlagScale auto-saves at configured intervals.
- To resume: keep the same YAML config and re-launch. FlagScale loads the latest checkpoint automatically.
- If user wants to start fresh, delete or rename the checkpoint directory FIRST (with confirmation).
- After resuming, verify the starting iteration matches the checkpoint iteration, not 0.

## Multi-node awareness

- Multi-node training requires: SSH passwordless access, consistent environment, shared/replicated data paths, correct NCCL env vars.
- Key NCCL env vars: `NCCL_IB_DISABLE` (set 1 if no InfiniBand), `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG=INFO`.
- Hostfile format: one line per node, `<hostname> slots=<num_gpus>`. Verify SSH connectivity before launching.
- When diagnosing multi-node failures, check logs on ALL nodes.

## Destructive operation safety

- Before `rm -rf`: first `ls` then `du -sh`. If > 1GB or contains checkpoints, ask user.
- Never delete `outputs/<exp>/` without checking for checkpoints.
- Prefer `mv` to trash over `rm -rf`.

## One-shot diagnosis

- FlagScale log path: `outputs/<exp>/logs/details/host_*/TIMESTAMP_DIR/default_*/attempt_0/0/stdout.log`. Use `find_latest_log` tool.
- Error diagnosis order: stderr FIRST → stdout tail → full stdout only if needed.
- When training fails: check stderr → identify error → fix root cause → clean stale outputs → retry. Don't retry without understanding the failure.

## Root cause diagnosis protocol

1. If the same category of error recurs after a fix attempt, STOP patching and diagnose the root cause. Common pattern: fixing a version mismatch by upgrading one package causes a cascade of new mismatches — the root cause is the original incompatible install, not each downstream error.
2. Before applying any fix, state the root cause hypothesis in one sentence. If you can't articulate it, you don't understand the problem.
3. Maximum 2 fix attempts for the same error. After 2 failures, step back and try a fundamentally different approach.
4. dtype mismatches (e.g., fp32 tensors in bf16 pipelines) are architecture-level issues. Trace the dtype from its source (RoPE, embedding, normalization) rather than adding `.to(dtype)` at the error site. For multimodal models, proactively add dtype casts at every component boundary during implementation.
5. Cascading TypeError/AttributeError on module init or forward means you didn't read the base class API. After the FIRST such error, read the COMPLETE `__init__` and `forward` signatures of the base class, then fix ALL mismatches at once.
6. Before calling any base class method from custom code, read its IMPLEMENTATION, not just its signature. Framework methods often do more than their name suggests (e.g., `get_query_key_value_tensors()` internally applies `linear_qkv` again).

## Fail-fast preflight checklist

Before any operation that takes >30 seconds, do a lightweight pre-check:
- **Model loading**: verify state_dict keys and shapes match BEFORE loading weights into GPU
- **Checkpoint conversion**: compare key counts and shapes between source and target BEFORE running full conversion
- **Training launch**: validate config arithmetic, verify ALL framework dependencies are importable (`python -c "import megatron, transformer_engine, apex, flash_attn"`)
- **Checkpoint ↔ parallelism**: verify checkpoint TP/PP matches runtime parallelism BEFORE launching
- **Memory budget**: `model_params × 2 (bf16) + gradients × 2 + optimizer_states × (8/DP)` — if it exceeds GPU memory, do NOT launch
- **Config arithmetic**: `global_batch_size % (micro_batch_size × DP) == 0`, `num_heads % TP == 0`, `num_kv_heads % TP == 0`
- **Config types**: verify values match expected types (int vs float, string vs list). Read argparse definitions for non-obvious types.
- **Environment install**: check version constraints BEFORE running pip install
