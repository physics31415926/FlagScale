---
name: infer-env-setup
description: Set up inference environment for vllm-plugin-FL on hardware backends. Covers SSH connection, Docker container creation,
  vLLM + plugin + FlagGems installation, and workspace initialization. Handles MetaX, Ascend, Moore Threads, and new backends
  via user-provided configuration template.
keywords:
- inference
- 推理
- 环境
- setup
- install
- docker
- container
- 容器
- ssh
- vllm
- plugin
- FlagGems
- 环境搭建
- 推理环境
- metax
- ascend
- 摩尔
- moore
parameters:
- name: hardware_backend
  description: Target hardware backend (e.g., metax, ascend, musa)
  default: metax
- name: ssh_host
  description: SSH host alias or connection string for the target machine (e.g., metax_c550, user@host). Ask user if not provided.
- name: vllm_version
  description: Target vLLM version to install (e.g., 0.20.2)
requires:
- workspace-layout
suggests:
- ops-discipline
constraints:
- id: ssh_connection_required
  description: SSH connection info must be confirmed before any remote operation
  trigger:
    tools:
    - shell
    keywords:
    - ssh
    - docker
  prompt: Check if the agent is running remote commands without having confirmed the SSH host/alias with the user
  correction: 'Before any remote operation, confirm SSH connection info. If ssh_host parameter is not set, ask the user: "What is the SSH alias or connection string for the target machine?" Verify connectivity with a simple `ssh <host> hostname` before proceeding. If SSH output is abnormal (empty, mixed with banners, or errors) for 3+ consecutive attempts, STOP and ask the user to verify the connection manually or provide alternative access method.'
- id: no_gpu_install_without_container
  description: Never install vLLM or plugin directly on host — always use Docker container
  trigger:
    tools:
    - shell
    keywords:
    - pip install vllm
    - pip install vllm-plugin
    - pip install flaggems
  prompt: Check if the agent is installing inference packages on the host machine instead of inside the Docker container
  correction: All inference packages must be installed inside the Docker container, not on the host.
- id: vllm_target_device_empty
  description: vLLM must be installed with VLLM_TARGET_DEVICE=empty on non-NVIDIA hardware
  trigger:
    tools:
    - shell
    keywords:
    - pip install vllm
    - pip install -e
  prompt: Check if vLLM is being installed without VLLM_TARGET_DEVICE=empty on non-NVIDIA hardware
  correction: 'On non-NVIDIA hardware, install vLLM with: VLLM_TARGET_DEVICE=empty pip install vllm (CPU-only, no CUDA compilation).'
- id: network_host_required
  description: Docker container must use --network host (bridge mode has DNS issues on GPU machines)
  trigger:
    tools:
    - shell
    keywords:
    - docker run
    - docker create
  prompt: Check if docker run is missing --network host flag
  correction: Always use --network host for inference containers on GPU machines (bridge mode causes DNS resolution failures).
- id: image_matches_backend
  description: Docker image must match the target hardware backend (e.g., vllm-metax for MetaX, ascend-pytorch for Ascend)
  trigger:
    tools:
    - shell
    keywords:
    - docker run
    - docker pull
  prompt: Check if the Docker image matches the target hardware_backend. MetaX must use vllm-metax image, Ascend must use ascend image, Moore Threads must use musa image.
  correction: 'Use the correct image for the target backend. Check the Container Setup section for the baseline image tag. Do not mix images across backends.'
- id: device_mount_required
  description: Docker container must mount the correct device path for the hardware backend
  trigger:
    tools:
    - shell
    keywords:
    - docker run
    - docker create
  prompt: Check if docker run is missing the --device flag for the target backend. MetaX needs /dev/dri, Ascend needs /dev/davinci*, Moore Threads needs /dev/mthreads.
  correction: 'Always mount the correct device: MetaX → `--device /dev/dri`, Ascend → `--device /dev/davinci0 --device /dev/davinci1 ...`, Moore Threads → `--device /dev/mthreads`. Check Container Setup section for exact flags.'
- id: workspace_volume_required
  description: Docker container must mount the workspace volume for code and model access
  trigger:
    tools:
    - shell
    keywords:
    - docker run
    - docker create
  prompt: Check if docker run is missing the workspace volume mount (-v). Without it, code, models, and logs are not accessible inside the container.
  correction: 'Always mount the workspace volume: `-v <host_workspace_path>:/workspace`. This provides access to plugin code, model weights, and adapt-logs. Check Container Setup section for the correct host path per machine.'
- id: check_device_occupancy
  description: Check if compute devices are occupied before running workloads
  trigger:
    tools:
    - shell
    keywords:
    - pytest
    - python examples/
    - vllm serve
  prompt: Check if the agent verified device occupancy (mx-smi, npu-smi, mtsmi, nvidia-smi) before launching a workload. If devices are occupied, inform the user.
  correction: 'Before running tests or inference, check device occupancy with the backend monitoring tool. If devices are in use, inform the user and ask which devices to use or whether to wait.'
- id: pin_vllm_version
  description: vLLM must be installed with a pinned version from pyproject.toml
  trigger:
    tools:
    - shell
    keywords:
    - pip install vllm
  prompt: Check if vLLM is being installed without a pinned version (==X.Y.Z)
  correction: 'Always pin vLLM version: check plugin pyproject.toml for the required version, then use `pip install vllm==X.Y.Z`.'
- id: fresh_workspace
  description: Every adaptation task must start with a fresh clone — never reuse existing directories
  trigger:
    tools:
    - shell
    keywords:
    - git clone
    - mkdir
    - cd /workspace
  prompt: Check if the agent is cloning into or reusing an existing directory that may contain stale state from other tasks
  correction: 'Always start from a fresh clone in a dedicated directory (e.g., /workspace/adapt/<backend>-vllm-<version>/). Before cloning, explicitly check if the target directory already exists: `ls /workspace/adapt/<backend>-vllm-<version>/ 2>/dev/null && echo EXISTS || echo OK`. If it EXISTS, ask the user whether to remove and re-clone or reuse. Do NOT silently cd into an existing directory.'
- id: platform_awareness
  description: The agent may run on Windows locally while the remote target is Linux — never use Linux-only commands in local pipes
  trigger:
    tools:
    - shell
    keywords:
    - grep
    - sed
    - awk
    - xargs
  prompt: Check if the agent is using Linux-only commands (grep, sed, awk, xargs) in a local shell pipe on Windows. These must only be used inside SSH commands (remote execution).
  correction: 'On Windows local shell, use findstr instead of grep, or run the command remotely via SSH. All grep/sed/awk usage must be inside `ssh <host> "..."` quotes (remote execution). For local file searching, use the IDE search tools instead of shell pipes.'
- id: jumpserver_single_line
  description: JumpServer (bastion host) SSH proxy only supports single-line commands — no heredocs, multi-line strings, or interactive input
  trigger:
    tools:
    - shell
    keywords:
    - ssh
    - docker exec
  prompt: Check if the agent is sending multi-line commands, heredocs (<<EOF), or commands with unescaped newlines through SSH to a JumpServer-proxied host. JumpServer captures and replays stdin line-by-line, breaking multi-line constructs.
  correction: 'When SSH goes through JumpServer (bastion proxy), strictly use single-line commands. Break complex operations into multiple `ssh <host> "<single_command>"` calls. Avoid heredocs (<<EOF), multi-line strings, and semicolons joining very long command chains. For complex scripts, write the script to a file first (`ssh <host> "echo ''<content>'' > /tmp/script.sh"`), then execute it (`ssh <host> "bash /tmp/script.sh"`).'
soft_constraints:
- id: use_tmux_for_long_commands
  description: Long-running commands (install, build, test) should use tmux to survive SSH timeouts
  trigger:
    tools:
    - shell
    keywords:
    - pip install
    - git clone
  prompt: Check if a long-running command (install, clone) is being run without tmux protection
  suggestion: 'For commands that take more than a few minutes, wrap in tmux: `tmux new-session -d -s work "<command>"`, then monitor with `tmux capture-pane -t work -p`.'
context_injection:
  always:
  - Critical Rules
  - Remote Access
  - Execution Logs
  by_tool:
    shell:
    - Remote Access
    - Container Setup
    - Environment Setup
---
# Inference Environment Setup

Set up the inference environment for vllm-plugin-FL on hardware backends.

## When to Use This Skill

Use this skill when:
- Setting up a new inference environment for hardware adaptation
- Creating Docker containers for vllm-plugin-FL testing
- Installing vLLM, plugin, and FlagGems on a new machine
- Reconnecting to an existing environment after a break

## Critical Rules

1. **Confirm SSH connection first** — ask user for SSH host/alias if not provided, verify with `ssh <host> hostname` before any work.
2. **All work happens inside Docker containers** — never install inference packages on the host.
3. **Fresh workspace isolation** — every adaptation task starts with a fresh clone (local and remote). Do NOT reuse existing directories or mix with other projects. Use a dedicated directory per task (e.g., `adapt/<backend>-vllm-<version>/`).
4. **Local edit → sync → remote test** — edit code locally (or on host), sync to container workspace, then run tests inside container. Don't edit files inside the container directly.
5. **Check device occupancy before tests** — use the backend's monitoring tool (see Container Setup) to confirm compute devices are free.
6. **vLLM installs as CPU-only** (`VLLM_TARGET_DEVICE=empty`) — the plugin provides hardware-specific backends.
7. **Pin vLLM version** — check plugin's `pyproject.toml` for the required version, never `pip install vllm` without `==X.Y.Z`.
8. **Check container existence** before creating — reuse running containers, start stopped ones.
9. **Use `--network host`** for Docker containers on GPU machines.
10. **Use tmux for long-running commands** — SSH sessions will timeout otherwise.

---

## Remote Access

All operations run on remote GPU machines via SSH. The agent does NOT have direct access to GPUs.

### Step 0: Confirm Connection & Gather Environment Info

If `ssh_host` is not provided, **ask the user**:
> "What is the SSH alias or connection string for the target hardware? (e.g., `metax_c550`, `ssh user@host -p port`)"

Once obtained, run the following **environment probe** in one shot to confirm connectivity and collect baseline info:

```bash
ssh <ssh_host> "echo '=== hostname ===' && hostname && \
  echo '=== date ===' && date && \
  echo '=== device info ===' && (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || mx-smi 2>/dev/null || npu-smi info 2>/dev/null || echo 'no device tool found') && \
  echo '=== device processes ===' && (nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null || echo 'N/A — check with backend tool') && \
  echo '=== disk space ===' && df -h /workspace /home 2>/dev/null | head -5 && \
  echo '=== docker containers ===' && docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' && \
  echo '=== docker images (vllm) ===' && docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -i vllm"
```

This gives you:
- **Connectivity**: hostname + date confirms SSH works
- **Hardware**: device type, count, memory
- **Device occupancy**: whether other processes are using compute devices (if occupied, inform user before proceeding)
- **Disk**: available space for models/containers
- **Existing containers**: whether to reuse or create new
- **Available images**: whether to pull or use existing

If SSH fails, ask the user to check their `~/.ssh/config` or provide full connection details (host, port, user, key file).

### Command Execution Pattern

All remote commands follow this pattern:
```bash
# Direct host command
ssh <ssh_host> "<command>"

# Command inside Docker container
ssh <ssh_host> "docker exec <container_name> env PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin bash -c '<command>'"
```

For long-running commands, use tmux:
```bash
ssh <ssh_host> "tmux new-session -d -s adapt '<command>'"
ssh <ssh_host> "tmux capture-pane -t adapt -p"  # check progress
```

### Execution Logs

All test and inference commands **must** tee output to log files. This allows efficient re-reading without re-running commands, especially after context compaction.

**Log directory**: `/workspace/adapt-logs/` (create on first use)

**Naming convention**: `<stage>_<YYYYMMDD_HHMMSS>.log`

```bash
# Create log directory
ssh <ssh_host> "docker exec <container> bash -c 'mkdir -p /workspace/adapt-logs'"

# Example: tee test output to log file
ssh <ssh_host> "docker exec <container> bash -c 'VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v 2>&1 | tee /workspace/adapt-logs/unit_$(date +%Y%m%d_%H%M%S).log'"
```

**Rules**:
- Every test/inference command uses `2>&1 | tee /workspace/adapt-logs/<stage>_<timestamp>.log`
- When diagnosing failures, **read the log file** (`read_file` or `tail`) instead of re-running the command
- Keep logs until PR is merged — they serve as evidence of test passes
- Log files are mounted via the workspace volume, accessible from both inside and outside the container

**Retrieving logs later**:
```bash
# List available logs
ssh <ssh_host> "ls -lt /workspace/adapt-logs/"

# Read a specific log (last 100 lines)
ssh <ssh_host> "tail -100 /workspace/adapt-logs/unit_20250520_143022.log"
```

---

## Container Setup

Each hardware backend has different Docker images, device paths, and system configuration. Select the section matching `hardware_backend`.

### MetaX C550

- **SSH alias**: `metax_c550`
- **Machine**: `bm-turing-hz1-zone1-MC550-64G-1-15` (8x MetaX C550 64GB)
- **MACA**: 2.33.0, Driver 2.15.9
- **GPU device path**: `/dev/dri`
- **GPU query tool**: `mx-smi`
- **Triton backend**: `triton/backends/metax/compiler.py`
- **FlagGems backend path**: `flag_gems/_metax/`
- **Platform check**: `current_platform.is_metax()`

**Docker image:**
- Baseline: `cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.19.0-maca.ai3.5.3.502-torch2.8-py312-ubuntu22.04-amd64`
- Check for newer local images:
  ```bash
  ssh metax_c550 "docker images --format '{{.Repository}}:{{.Tag}}' | grep vllm-metax"
  ```
- Query registry for newer tags:
  ```bash
  ssh metax_c550 "curl -s https://cr.metax-tech.com/v2/public-ai-release/maca/vllm-metax/tags/list 2>/dev/null | python3 -m json.tool || echo 'Registry query failed — use baseline image'"
  ```

**Container provides**: Python 3.12, torch 2.8.0+metax, MACA runtime, conda at `/opt/conda`

**Check existing container:**
```bash
ssh metax_c550 "docker ps -a --filter name=vllm-plugin-test --format '{{.Status}}'"
```

**Create container (if needed):**
```bash
ssh metax_c550 "docker run -d --name vllm-plugin-test \
  --privileged \
  --network host \
  --device /dev/dri \
  -v /home/secure/flagos-test:/workspace \
  cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.19.0-maca.ai3.5.3.502-torch2.8-py312-ubuntu22.04-amd64 \
  sleep infinity"
```

**Execute commands inside container:**
```bash
ssh metax_c550 "docker exec vllm-plugin-test env PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin bash -c '<command>'"
```

**System mirrors (China mainland):**
```bash
# apt (aliyun)
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" > /etc/apt/sources.list
apt-get update && apt-get install -y git

# pip (Tsinghua)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Test model**: Qwen3.5-35B-A3B, path `/workspace/models/Qwen3.5-35B-A3B`, TP=2, `--enforce-eager`

### Ascend 910B

- **SSH alias**: *(ask user)*
- **Machine**: *(ask user — e.g., 8x Ascend 910B 64GB)*
- **CANN**: *(ask user — e.g., 8.0.RC3)*
- **Device path**: `/dev/davinci*`, `/dev/devmm_svm`, `/dev/hisi_hdc`
- **Device query tool**: `npu-smi info`
- **Triton backend**: *(N/A — Ascend uses custom kernels, not Triton)*
- **FlagGems backend path**: *(N/A or ask user)*
- **Platform check**: `current_platform.is_ascend()`

**Docker image:**
- Baseline: *(ask user — typically from Ascend Hub, e.g., `ascendhub.huawei.com/public-ascendhub/ascend-pytorch:<version>`)*
- Check for local images:
  ```bash
  ssh <ssh_host> "docker images --format '{{.Repository}}:{{.Tag}}' | grep -i ascend"
  ```

**Container provides**: Python 3.10+, torch_npu, CANN toolkit

**Create container (if needed):**
```bash
ssh <ssh_host> "docker run -d --name vllm-plugin-test \
  --privileged \
  --network host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v <host_workspace>:/workspace \
  <ascend_image> \
  sleep infinity"
```

**Execute commands inside container:**
```bash
ssh <ssh_host> "docker exec vllm-plugin-test bash -c '<command>'"
```

**Test model**: *(ask user — model path, TP size, extra flags)*

### Moore Threads S4000 (摩尔线程)

- **SSH alias**: *(ask user)*
- **Machine**: *(ask user — e.g., 8x MTT S4000 48GB)*
- **MUSA**: *(ask user — e.g., MUSA 4.0, Driver rc3.1.0)*
- **Device path**: `/dev/mthreads`
- **Device query tool**: `mtsmi`
- **Triton backend**: `triton/backends/musa/compiler.py`
- **FlagGems backend path**: *(ask user — e.g., `flag_gems/_musa/`)*
- **Platform check**: `current_platform.is_musa()`

**Docker image:**
- Baseline: *(ask user — typically from Moore Threads registry)*
- Check for local images:
  ```bash
  ssh <ssh_host> "docker images --format '{{.Repository}}:{{.Tag}}' | grep -i musa"
  ```

**Container provides**: Python 3.10+, torch_musa, MUSA toolkit

**Create container (if needed):**
```bash
ssh <ssh_host> "docker run -d --name vllm-plugin-test \
  --privileged \
  --network host \
  --device /dev/mthreads \
  -v <host_workspace>:/workspace \
  <musa_image> \
  sleep infinity"
```

**Execute commands inside container:**
```bash
ssh <ssh_host> "docker exec vllm-plugin-test bash -c '<command>'"
```

**Test model**: *(ask user — model path, TP size, extra flags)*

### Adding a New Backend

When adapting for a hardware backend not listed above, **ask the user** to provide the following information. Present this template:

> To set up the Container Setup section for your hardware, I need the following info:
>
> | Field | Example (MetaX) | Your value |
> |-------|-----------------|------------|
> | SSH alias or connection string | `metax_c550` | |
> | Machine description (GPU model × count, memory) | 8x MetaX C550 64GB | |
> | Runtime/SDK version | MACA 2.33.0, Driver 2.15.9 | |
> | Device path (`--device` flag) | `/dev/dri` | |
> | Device query tool (like nvidia-smi) | `mx-smi` | |
> | Triton backend path (if applicable) | `triton/backends/metax/compiler.py` | |
> | FlagGems backend path (if applicable) | `flag_gems/_metax/` | |
> | Platform check function | `current_platform.is_metax()` | |
> | Docker image (full tag) | `cr.metax-tech.com/...` | |
> | Container provides (Python, torch, runtime) | Python 3.12, torch 2.8.0+metax | |
> | Host workspace path (for -v mount) | `/home/secure/flagos-test` | |
> | Test model (name, path, TP size) | Qwen3.5-35B-A3B, /workspace/models/..., TP=2 | |
> | Extra docker flags (if any) | `--privileged` | |
> | System mirrors needed? (apt/pip) | Yes — China mainland | |

Once the user provides this, fill in a new `### <Backend Name>` section following the MetaX format.

---

## Environment Setup

All commands run inside the container. System mirrors (apt, pip) are configured in the Container Setup section above.

### Step 3: Install vLLM (CPU-only)

```bash
VLLM_TARGET_DEVICE=empty pip install vllm==<target_version>
```

This installs vLLM without compiling CUDA kernels — the plugin provides the hardware backend.

### Step 4: Clone and install vllm-plugin-FL

**Development model**: local edit → sync to container → test remotely.

The agent edits code **locally** (or on the host machine), then syncs changes to the container's workspace for testing. This avoids installing dev tools inside the container and keeps the container environment clean.

**Initial clone (on host or local machine):**

Always start from a **fresh clone** — never reuse an existing directory that may contain stale state from other tasks.

```bash
# On the remote host (outside container), create a fresh workspace for this adaptation
ssh <ssh_host> "mkdir -p /workspace/adapt/<backend>-vllm-<version> && cd /workspace/adapt/<backend>-vllm-<version> && git clone <vllm-plugin-FL-repo-url> vllm-plugin-FL && cd vllm-plugin-FL && git checkout main && git checkout -b adapt/<backend>-vllm-<version>"

# Locally, also use a fresh clone (do NOT reuse other working directories)
mkdir adapt-<backend>-vllm-<version> && cd adapt-<backend>-vllm-<version>
git clone <vllm-plugin-FL-repo-url> vllm-plugin-FL && cd vllm-plugin-FL && git checkout main && git checkout -b adapt/<backend>-vllm-<version>
```

Since the workspace is volume-mounted (`-v /host/path:/workspace`), the code is immediately visible inside the container.

**Install plugin inside container (editable mode):**
```bash
ssh <ssh_host> "docker exec <container> bash -c 'cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL && pip install -e .'"
```

**Code sync after local edits:**

If editing on a separate local machine (not the remote host), sync changes with:
```bash
# Option 1: git push + pull (preferred — clean history, works because container uses --network host)
# Local:
git add -A && git commit -m "wip: <description>" && git push
# Remote (inside container):
ssh <ssh_host> "docker exec <container> bash -c 'cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL && git pull'"

# Option 2: rsync (fast incremental sync, no commit needed)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
  ./vllm-plugin-FL/ <ssh_host>:/workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/

# Option 3: scp for single files or directories
scp ./path/to/file.py <ssh_host>:/workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/path/to/file.py
scp -r ./src/dir/ <ssh_host>:/workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/src/dir/
```

Container uses `--network host`, so git operations inside the container have full network access (same as host). If git clone/pull fails due to proxy issues, configure git proxy (see CLAUDE.md).

**After syncing, no reinstall needed** for editable installs (`pip install -e .`) — Python picks up changes immediately. Exception: if you modify `setup.py`/`pyproject.toml` or add new entry points, re-run `pip install -e .`.

Plugin code always starts from the **latest `main` branch**. Create a new branch for the adaptation work (e.g., `adapt/metax-vllm-0.20.2`). If working on a fork, fork first then create the branch.

### Step 5: Install FlagGems

```bash
cd /workspace/adapt/<backend>-vllm-<version>
git clone <flaggems-repo-url> FlagGems
cd FlagGems
git checkout main                    # use latest main
pip install -e .
# Record version for later
git log --oneline -1                 # note the commit hash
```

FlagGems uses the **latest `main` branch**. Record the commit hash — it will be included in the PR description and adaptation record.

### Step 6: Verify installation

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import vllm_plugin_fl; print('Plugin loaded')"
python -c "import flag_gems; print('FlagGems loaded')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
```

### Step 7: Create adapt-logs directory

```bash
mkdir -p /workspace/adapt-logs
```

This directory is used by `infer-hw-adapt` to store test and inference logs. Creating it during setup ensures it's ready when testing begins.

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `VLLM_PLUGINS=fl` | Activate the FL plugin | Required for all tests |
| `VLLM_TARGET_DEVICE=empty` | CPU-only vLLM install | Only during pip install |
| `MODEL_PATH` | Model weights location | `/workspace/models/Qwen3.5-35B-A3B` |
| `TP_SIZE` | Tensor parallel size | `2` |
| `PP_SIZE` | Pipeline parallel size | `1` |

---

## Related Skills

- `infer-hw-adapt` — hardware adaptation testing, patching, and PR submission (use after environment is set up)
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts
