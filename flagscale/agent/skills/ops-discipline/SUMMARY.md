# Operational Discipline — Summary

## Overview

Operational discipline rules for FlagScale infrastructure work. Covers understand-before-act, reading strategy, tool-first principle, decision discipline, shell safety, dependency resolution, training launch, and monitoring.

## Core Rules

### 1. Understand Before Act — The #1 Rule

- **Complex tasks**: spend at least 30% of effort on reading and understanding BEFORE writing code
- **Read complete files, not fragments**: Use `read_file` tool to read full files, not `sed -n` slices
- **Never declare "I understand" after reading one file**: Complex systems have hidden dependencies
- **Before implementing, list what you DON'T know**: If the list is long, keep reading

**Anti-patterns to avoid**:
- Reading a file 40 times in fragments instead of once completely
- Premature conclusions that get reversed ("关键发现！" → "但等等——" loops)
- Starting implementation after reading 20% of relevant code

### 2. Reading Strategy — Minimize Re-reads

- Use `read_file` tool with line ranges, not `sed`/`cat`/`head`/`tail`
- **First read**: full file to get complete picture
- **Subsequent reads**: targeted ranges using line numbers from first read
- **Record key findings in workspace_state** so they survive context compaction
- **Never re-read a file you read in the last 5 turns** unless it was modified

### 3. Tool-First Principle

Use specialized tools over shell commands:
- `find_latest_log` for locating training logs
- `parse_training_metrics` for extracting loss/grad/throughput
- `read_file` for reading code
- `workspace_state` for persisting findings
- `plan_create` / `plan_update` for complex tasks

### 4. Decision Discipline — No Flip-Flopping

- **Before choosing approaches, LIST ALL CONSTRAINTS**
- **Evaluate each approach against ALL constraints before picking one**
- **Once you pick an approach, commit to it**: Run to completion or clear failure
- **If an approach fails, record WHY** before trying the next one
- **Never flip between approaches more than twice**: If A→B→A, stop and ask user

**Solve one problem at a time**:
1. Form hypothesis about root cause
2. Design verification experiment testing ONLY that hypothesis
3. Run it and interpret result
4. If confirmed: fix and verify. If refuted: record "not this" and move on

### 5. Shell Command Rules

- **NEVER search from root** (`find /`). Scope to working directory or known paths
- Use `conda run -n <env> <command>` for conda environments (NEVER `conda activate` in non-interactive shells)
- **NEVER install into base or current environment** unless explicitly asked
- To stop training: `flagscale train <model> --config <config> --stop` or `cat outputs/<exp>/logs/pids/* | xargs kill -9`
- **Before launching training, verify no old processes** (`pgrep`)
- **NEVER use `sleep N && <command>` for monitoring**
- **NEVER run the same command twice in a row**
- **NEVER modify third-party source code** to work around build errors
- For large downloads: `wget -c` or `curl -C -` (resume-capable)
- Before `rm -rf`: first `ls` then `du -sh`

### 6. Environment Awareness

First thing on any new server:
```bash
nvidia-smi
cat /etc/os-release
which conda
echo $CUDA_HOME
```
Save to workspace_state.

Check before operations:
- Disk space: `df -h` before large downloads/builds
- GPU memory: `nvidia-smi` before launching training
- Conda env exists: `conda env list` before using it

### 7. Training Launch Discipline

**Before launch, verify**:
1. Config is valid
2. Data paths exist
3. Model weights are accessible
4. GPU count matches config

**After launch**:
- IMMEDIATELY check logs (within 30 seconds) for startup errors
- If training fails within first 100 iterations → likely config or environment issue

### 8. Trust Nothing, Verify Everything

- After `pip install X`: `python -c "import X; print(X.__version__)"`
- After `git clone`: verify directory exists and has expected content
- After writing config: `cat` it back and verify key values
- After starting process: verify it's running with `ps` or check output

### 9. Dependency Resolution — Constraint Solving, Then One-Shot Install

**NEVER blindly install and fix conflicts after**. Install ONCE and get it right.

**Phase 1: Collect constraints** (NO installs)
1. Hardware: `nvidia-smi` → driver version → max CUDA version
2. Framework: read setup.cfg/pyproject.toml → extract PyTorch/Python bounds
3. Recipe: check for additional packages (flash-attn, apex) and versions

**Phase 2: Solve** (find intersection)
- Write out constraint table explicitly
- Present viable options to user
- DEFAULT: prefer lowest PyTorch + highest compatible CUDA

**Phase 3: One-shot install**
```bash
conda create -n <env> python=<version> -y
conda run -n <env> pip install torch==<version> --index-url https://download.pytorch.org/whl/<cu>
conda run -n <env> pip install <framework>
conda run -n <env> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 10. Checkpoint Resume

- Check: `ls outputs/<exp>/checkpoints/`
- To resume: keep same YAML config and re-launch (FlagScale auto-loads latest)
- To start fresh: delete or rename checkpoint directory FIRST (with confirmation)
- After resuming: verify starting iteration matches checkpoint, not 0

### 11. Multi-Node Awareness

Requirements:
- SSH passwordless access
- Consistent environment
- Shared/replicated data paths
- Correct NCCL env vars

Key NCCL env vars:
- `NCCL_IB_DISABLE=1` (if no InfiniBand)
- `NCCL_SOCKET_IFNAME`
- `NCCL_DEBUG=INFO`

Hostfile format: `<hostname> slots=<num_gpus>`

### 12. One-Shot Diagnosis

- FlagScale log path: `outputs/<exp>/logs/details/host_*/TIMESTAMP_DIR/default_*/attempt_0/0/stdout.log`
- Use `find_latest_log` tool
- Error diagnosis order: stderr FIRST → stdout tail → full stdout only if needed
- When training fails: check stderr → identify error → fix root cause → clean stale outputs → retry

## When to Load Full Skill

Load the full `ops-discipline` skill when:
- Setting up complex multi-node training environments
- Debugging dependency resolution issues
- Understanding detailed shell safety rules
- Learning about FlagScale-specific operational patterns
- Troubleshooting training launch or checkpoint resume issues

This summary covers the core operational principles. For detailed examples, anti-patterns, and edge cases, load the full skill.
