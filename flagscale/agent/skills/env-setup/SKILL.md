---
name: env-setup
description: Set up FlagScale training environment on GPU servers. Install conda env, FlagScale, and all FL-customized dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention). Supports both pip install and source build fallback. Handles CUDA compatibility detection, multi-node deployment, and Docker image setup.
keywords:
  - 安装
  - 环境
  - setup
  - install
  - env
  - 环境搭建
  - 训练环境
  - conda
  - megatron
  - transformer-engine
  - apex
  - flash-attention
  - 依赖
  - 编译
  - build
  - cuda
  - driver
  - 驱动
  - 多机
  - multi-node
parameters:
  - name: env_name
    description: Conda environment name
    default: flagscale-train
  - name: python_version
    description: Python version
    default: "3.12"
  - name: deps_dir
    description: Directory to clone source dependencies. If shared storage is detected, use <workspace_root>/code/deps/ so all nodes can access the same builds. Only fall back to a local path if no shared storage exists.
    default: <workspace_root>/code/deps/
requires: [workspace-layout]
suggests: []
constraints:
  - id: env_setup_conda_prefix_not_shared_storage
    description: Conda environment must be created with --prefix on shared storage (not local /tmp). Local paths prevent multi-node access.
    severity: warning
    check_phase: pre
    trigger:
      tools: [shell]
      keywords: [conda create, conda env create]
    prompt: "Check if conda env is being created on shared storage (--prefix) rather than local path"
    correction: "Use --prefix on shared storage for multi-node access."
  - id: env_setup_pip_install_not_verify_import
    description: After every pip install, immediately verify with "python -c 'import <package>'" to catch corrupt installs or import hangs.
    severity: warning
    check_phase: post
    trigger:
      tools: [shell]
      keywords: [pip install]
    prompt: "Check if pip install was followed by import verification"
    correction: "Verify with: python -c 'import <package>'"
  - id: env_setup_pip_install_flagscale_without_no_deps
    description: pip install flagscale must use --no-deps to prevent PyTorch silent upgrade.
    severity: error
    check_phase: pre
    trigger:
      tools: [shell]
      keywords: [pip install, flagscale]
    prompt: "Check if pip install flagscale uses --no-deps flag"
    correction: "Use: pip install --no-deps -e . (or pip install --no-deps flagscale)"
    max_violations: 0
  - id: env_setup_pip_install_flash_attn_missing_no_deps
    description: pip install flash-attn must use --no-deps to prevent PyTorch from being overridden.
    severity: error
    check_phase: pre
    trigger:
      tools: [shell]
      keywords: [pip install, flash-attn, flash_attn]
    prompt: "Check if pip install flash-attn uses --no-deps flag"
    correction: "Use: pip install --no-deps flash-attn"
    max_violations: 0

warnings:
  - id: cuda_version_check
    description: "Check CUDA/driver version before installing GPU packages"
    severity: warning
    trigger:
      keywords: [install, build, compile, pip install, conda install]
    prompt: "Check if CUDA/driver version was verified before installing GPU-dependent packages"
    reminder: "Run nvidia-smi and check CUDA version before installing torch/TE/apex/flash-attn."
    max_reminders: 1
  - id: verify_import_after_install
    description: "Verify import after pip install to catch corrupt installs"
    severity: warning
    trigger:
      keywords: [pip install]
    prompt: "Check if pip install was followed by import verification"
    reminder: "After pip install, verify with: python -c 'import <package>; print(<package>.__version__)'"
    max_reminders: 3

context_injection:
  always: ["Strategy", "CRITICAL: Source-of-truth principle"]
  by_tool:
    shell: ["General rules", "Step 1", "Step 2", "Step 3"]
---

# FlagScale Training Environment Setup

Set up a complete FlagScale training environment on a GPU server. All dependencies use FL-customized versions.

## Strategy

Environment setup is a constraint satisfaction problem. Collect ALL constraints first, solve for compatible versions, then install once.

### CRITICAL: Source-of-truth principle

**NEVER reference or inspect existing environments when determining what to install.** Existing environments (even `flagscale-train`, even on the same machine) may have different hardware, editable installs pointing to other workspaces, patched packages, or stale versions. They tell you NOTHING useful about what the CURRENT environment needs.

The ONLY valid sources of truth for dependency versions are:
1. FlagScale's own `requirements/*.txt`, `setup.py`, `setup.cfg`, `pyproject.toml`
2. The upstream repos of FL-customized dependencies: Megatron-LM-FL, TransformerEngine-FL
3. The actual hardware (driver version, GPU type) — queried fresh with nvidia-smi

**Do NOT run `pip list`, `conda list`, `pip show` in any existing environment.** Do NOT look at what another environment has installed. These are irrelevant and misleading.

### General rules

1. ALL installs go into the target conda environment — NEVER install into base or current environment. Use `conda run -n <env> pip install ...` for every pip command. To check dependency versions without installing, read setup.cfg/pyproject.toml from the source repo or use `pip index versions <pkg>`.
2. Try pip install with pinned versions (fast, from FlagScale PyPI)
3. If pip fails, fall back to source clone + build
4. Never modify dependency source code to work around errors — report to user
5. **After EVERY pip install, VERIFY the import works.** DO NOT assume a successful pip exit code means the package is usable. Immediately test: `python -c "import <package>; print(<package>.__version__)"`. For large packages (torch, flash-attn, apex), if `import` hangs >10s, the install is corrupt and must be redone. On NFS/shared storage, use `timeout 15 python -c "import <package>"` to catch hangs quickly without blocking the session.
6. **Auto-fetch FL dependencies**: When Megatron-LM-FL or TransformerEngine-FL source code is needed (for analysis, compilation, or debugging) and is not available locally, pull the latest automatically — don't ask the user. Repos: `https://github.com/flagos-ai/Megatron-LM-FL.git`, `https://github.com/flagos-ai/TransformerEngine-FL.git` (use `--recursive` for TE-FL)
7. **ALL FL-customized dependencies are MANDATORY.** Do NOT skip Megatron-LM-FL, TransformerEngine-FL, Apex, or Flash-Attention. These are not optional — FlagScale training will fail or produce incorrect results without them. If one is difficult to install, try the source build fallback. Only skip a dependency if the user explicitly requests it after being warned of the consequences.
8. **If the user asks to create a new environment, create a new environment.** Do not reuse an existing one, even if it appears to have the right packages. Existing environments may have editable installs pointing to other workspaces, patched packages, or stale versions. A fresh environment is the only way to guarantee a clean, reproducible baseline. If you believe reusing is genuinely better, explain why and ask — but do not silently substitute.
9. **NEVER copy packages between environments using `cp -r` from site-packages.** This bypasses pip's metadata tracking — pip won't know the package exists, so dependency resolution, upgrades, and uninstalls all break silently. Always install via `pip install` (from wheel, PyPI, or source build). If a prebuilt wheel isn't available, build from source — it takes longer but produces a properly registered package.
10. **Prefer shared storage for conda environments.** If the working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment with `--prefix <shared_path>/envs/<name>` instead of `-n <name>`. This ensures all nodes can access the same environment in multi-node training without duplication. Use `--prefix` for ALL subsequent `conda run` commands targeting this environment. Only use `-n` if no shared storage is available.
11. **Conda envs and pip packages MUST go on shared storage, not local paths.** Even if `/tmp` or local disk has more space or is faster, the conda environment prefix and pip install target MUST be on shared storage (e.g., `/share/.../envs/<name>`). The only exception is `TMPDIR` for pip's temporary build cache — that can point to local storage to speed up compilation, but the final installed packages must land in the shared prefix.

## Step 0: Determine Dependency Source Directory

**Before anything else, determine `deps_dir` — the directory for cloning and building source dependencies.**

1. If workspace-layout skill has been loaded and `workspace_root` is known (from memory or detection), set `deps_dir = <workspace_root>/code/deps/`. This ensures all nodes in multi-node training can access the same builds.
2. If shared storage is available but workspace_root is not yet set, detect it now (see workspace-layout Step 1) and use it.
3. Only if NO shared storage is available, fall back to a local path.

**Summary**: `deps_dir` is always on shared storage when available. Never hardcode `/opt/flagscale/deps` — this path is local to one node and invisible to others.

Record `deps_dir` in memory after determining it.

## Step 1: Constraint Collection (NO installs in this step)

Collect ALL version constraints before installing anything. Do NOT look at existing environments.

### 1a. Hardware constraint — driver → max CUDA

```bash
nvidia-smi --query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader | head -1 && echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
nvcc --version 2>/dev/null || echo "nvcc not found"
```

The `GPU_COUNT=` line gives the exact GPU count. Use that number in all subsequent references — never count nvidia-smi output lines manually.

Driver → max CUDA version (for PyTorch wheel selection):
- Driver 570.x → CUDA ≤ 12.8 → wheels: cu118, cu121, cu124, cu126, cu128
- Driver 560.x → CUDA ≤ 12.6 → wheels: cu118, cu121, cu124, cu126
- Driver 550.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 535.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 530.x → CUDA ≤ 12.1 → wheels: cu118, cu121
- Driver 520.x → CUDA ≤ 11.8 → wheels: cu118

### 1b. FlagScale framework constraint — read from source

Read FlagScale's own dependency declarations (NOT from any installed environment):

```bash
cat requirements.txt
cat requirements/cuda/train.txt
cat requirements/cuda/base.txt
cat setup.py
```

Also fetch the setup configs of the two FL forks to check their torch/python requirements:

```bash
# Megatron-LM-FL: check setup.py for torch/python_requires
web_fetch https://raw.githubusercontent.com/flagos-ai/Megatron-LM-FL/main/setup.py
# TransformerEngine-FL: check setup.py for torch/python/minor version requirements
web_fetch https://raw.githubusercontent.com/flagos-ai/TransformerEngine-FL/main/setup.py
```

### 1c. FL-customized dependency analysis (as important as PyTorch itself)

FlagScale requires four FL-customized / special packages. ALL four are MANDATORY:

| Package | Source | Install method (primary) | Fallback |
|---------|--------|-------------------------|----------|
| Megatron-LM-FL | flagos-ai PyPI / GitHub | pip from FlagScale PyPI | source build |
| TransformerEngine-FL | flagos-ai PyPI / GitHub | pip from FlagScale PyPI | source build (--recursive) |
| Apex | NVIDIA GitHub | source build (APEX_CUDA_EXT=1) | N/A — must build from source |
| Flash-Attention | Dao-AILab GitHub | source build (--no-deps) | N/A — must build from source |

For each, analyze:
- **Megatron-LM-FL & TransformerEngine-FL**: Check the PyPI index URL `https://resource.flagos.net/repository/flagos-pypi-hosted/simple` for available versions. If a compatible wheel exists, use it (fast). If the wheel fails or is outdated, fall back to source build.
- **Apex**: Always source build. Must compile with `APEX_CUDA_EXT=1` matching PyTorch's CUDA version. Check that the nvcc toolkit version matches torch.version.cuda (not just driver CUDA version).
- **Flash-Attention**: Always source build. The version must match the installed PyTorch version. Use `--no-deps` to prevent pip from upgrading PyTorch. Check: GPU compute capability ≥ 8.0 required for flash-attn v2.x.

### 1d. Solve — write the FULL compatibility table

Write a COMPLETE compatibility table covering ALL components. Do NOT skip to Step 2 until this table is written and verified:

```
COMPATIBILITY ANALYSIS TABLE
============================
Hardware: N×GPU_TYPE, Driver DRI_VER → max CUDA CUDA_MAX
FlagScale requirements:
  Python: py_req
  PyTorch: torch_req
  CUDA toolkit required: cuda_toolkit_needed

| # | Component | Required Version | Install Method | Notes |
|---|-----------|-----------------|---------------|-------|
| 1 | Conda env | python=py_ver | conda create --prefix | path: <shared>/envs/env_name (or -n if no shared storage) |
| 2 | PyTorch | torch_ver+cuXXX | pip | --extra-index-url https://download.pytorch.org/whl/cuXXX |
| 3 | FlagScale | editable | pip -e ".[cuda-train]" | from project root |
| 4 | Megatron-LM-FL | mlm_ver | pip/source | FlagScale PyPI / git clone + build |
| 5 | TransformerEngine-FL | te_ver | pip/source | FlagScale PyPI / git clone --recursive + build |
| 6 | Apex | master | source build | git clone NVIDIA/apex + APEX_CUDA_EXT=1 |
| 7 | Flash-Attention | fa_ver | source build | --no-deps to protect PyTorch |
```

CRITICAL CHECKLIST before proceeding:
- [ ] All versions in the table are derived from FlagScale source files (NOT existing envs)
- [ ] Shared storage checked — conda env path uses --prefix on shared FS if available
- [ ] CUDA toolkit version matches PyTorch's CUDA (not driver's)
- [ ] GPU compute capability ≥ required by flash-attn
- [ ] Megatron-LM-FL wheel exists on FlagScale PyPI at the needed version
- [ ] TransformerEngine-FL wheel exists on FlagScale PyPI at the needed version
- [ ] Apex build flags include APEX_CUDA_EXT=1
- [ ] Flash-attn install uses --no-deps

Present the table and ASK FOR CONFIRMATION. Do NOT proceed to Step 2 until the user confirms.
After confirmation, annotate your response with [ENV_COMPAT_ANALYZED].

## Step 2: Conda Environment

### 2a. Check shared storage FIRST

**CRITICAL**: If the current working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment on the shared storage — NOT on the local node. This ensures all nodes in multi-node training can access the same environment without duplication.

```bash
# Check if we're on shared storage
df -h . | grep -E '^[^/]' | head -5

# Check available shared mount points
ls -d /share /mnt/share /mnt/cfs /mnt/dfs 2>/dev/null
```

If shared storage is found (e.g., `/share/project/...`), use `--prefix` instead of `--name`:

```bash
# Create env in shared storage — use --prefix with full path
conda create --prefix /share/project/<path>/envs/{env_name} python={python_version} -y

# For all subsequent commands, use --prefix (not -n):
conda run --prefix /share/project/<path>/envs/{env_name} <command>
```

If NO shared storage is found, fall back to `-n`:

```bash
conda create -n {env_name} python={python_version} -y
# In non-interactive shells (agent), use: conda run -n {env_name} <command>
# In interactive shells (user), use: conda activate {env_name}
```

### 2b. Verify

```bash
python --version
```

## Step 3: Install FlagScale

### 3a. Pin PyTorch FIRST (before installing FlagScale)

**CRITICAL**: `pip install -e ".[cuda-train]"` will pull in ALL requirements, including PyTorch from the requirements files. If those requirements specify a different CUDA version than what your driver supports, pip will silently upgrade PyTorch and all CUDA libraries. This is the #1 cause of wasted time in environment setup.

**Always pin PyTorch before FlagScale install:**

```bash
# Install exact PyTorch version from Step 1 compatibility analysis
pip install torch=={torch_version}+{cu_tag} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/{cu_tag}
# Verify CUDA version is correct
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 3b. Install FlagScale editable

From the FlagScale project root:

```bash
pip install -e ".[cuda-train]"
```

**If pip tries to upgrade PyTorch during this step**, abort and use the two-phase approach:
```bash
# Phase 1: install FlagScale without deps
pip install --no-deps -e .
# Phase 2: install remaining deps from requirements (PyTorch already pinned, won't change)
pip install -r requirements/cuda/train.txt
```

This ensures PyTorch stays at the pinned version.

Verify:
```bash
flagscale --help
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# CRITICAL: confirm torch version did NOT change from what was installed in 3a
```

**Important**: `requirements/cuda/train.txt` includes a megatron-core whl from the FlagScale PyPI. This is a base version — it may lack modules like `megatron.plugin.platform` that FlagScale's training code requires. If training fails with import errors from megatron, proceed to Step 4.

## Step 4: FL-Customized Dependencies

These are FlagScale's customized forks. Install order matters — Megatron-LM-FL first, then the rest.

### 4a. Megatron-LM-FL

**Try pip first:**
```bash
pip install megatron_core==0.1.0+megatron0.15.0rc7 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

Verify:
```bash
python -c "from megatron.plugin.platform import get_platform; print('OK:', get_platform())"
```

**If pip fails or module missing, build from source:**
```bash
mkdir -p {deps_dir}
git clone https://github.com/flagos-ai/Megatron-LM-FL.git {deps_dir}/Megatron-LM-FL
cd {deps_dir}/Megatron-LM-FL
pip install --no-build-isolation . -v
```

Verify again with the same python command above.

### 4b. TransformerEngine-FL

**Try pip first:**
```bash
pip install transformer_engine==0.1.0+te2.9.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

Verify:
```bash
python -c "import transformer_engine; print('TE version:', transformer_engine.__version__)"
```

**If pip fails, build from source:**
```bash
pip install nvidia-mathdx --extra-index-url https://pypi.nvidia.com
git clone --recursive https://github.com/flagos-ai/TransformerEngine-FL.git {deps_dir}/TransformerEngine-FL
cd {deps_dir}/TransformerEngine-FL
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . -v
```

Note: Source build takes 10-30 minutes. Do NOT interrupt or ask for confirmation during compilation — just wait for it to finish.

### 4c. NVIDIA Apex

```bash
git clone https://github.com/NVIDIA/apex.git {deps_dir}/apex
cd {deps_dir}/apex
NVCC_APPEND_FLAGS='--threads 4' APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    pip install --no-build-isolation . -v
```

Verify:
```bash
python -c "import apex; print('Apex OK')"
```

**Common issue**: CUDA version mismatch between system nvcc and PyTorch's CUDA. If Apex build fails with version check error, report the exact error to the user — do NOT modify Apex source code.

**IMPORTANT: Pure-Python vs CUDA Extensions**

Apex has two install modes:
- **Full install** (with `APEX_CUDA_EXT=1`): Compiles CUDA extensions for fused kernels. Required for `gradient_accumulation_fusion`, fused Adam, fused layer norm, etc.
- **Pure-Python install** (without CUDA flags or `pip install apex`): Only Python wrappers, NO fused kernels. Many Megatron features silently fall back to slower paths or fail with `RuntimeError: ... requires APEX CUDA extensions`.

**If you see `gradient_accumulation_fusion requires APEX CUDA extensions`**: Apex was installed in pure-Python mode. You must either:
1. Reinstall with CUDA extensions (recommended): use the build command above with `APEX_CUDA_EXT=1`
2. OR disable ALL fusion flags at once: `gradient_accumulation_fusion: false`, `bias_gelu_fusion: false`, `bias_swiglu_fusion: false` — and note the performance impact

Never disable just one fusion flag — if APEX CUDA extensions are missing, ALL fused kernels are unavailable.

### 4d. Flash-Attention 2

**CRITICAL**: Always use `--no-deps` when installing flash-attn. Without it, pip may upgrade PyTorch to an incompatible version, causing cascading failures (triton mismatch, CUDA version conflicts). The PyTorch version was already pinned in Step 3 — do not let flash-attn override it.

```bash
git clone --branch v2.8.1 --depth 1 https://github.com/Dao-AILab/flash-attention.git {deps_dir}/flash-attention
cd {deps_dir}/flash-attention
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
    pip install --no-build-isolation --no-deps . -v
```

**CUDA toolkit vs driver version**: Flash-attn compilation requires the CUDA **toolkit** version (nvcc) to match PyTorch's CUDA version, NOT the driver version. Check with `nvcc --version` (toolkit) vs `nvidia-smi` (driver). If nvcc is missing or wrong version, install the matching CUDA toolkit or set `CUDA_HOME` to the correct path.

After installing, verify PyTorch was NOT changed:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
If the version differs from what was installed in Step 3, flash-attn broke the environment. Uninstall flash-attn, reinstall the correct PyTorch, and retry with `--no-deps`.

Verify:
```bash
python -c "import flash_attn; print('Flash-Attention version:', flash_attn.__version__)"
```

## Step 5: Final Verification

Run a comprehensive check:

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')

from megatron.plugin.platform import get_platform
print(f'Megatron platform: {get_platform()}')

import transformer_engine
print(f'TransformerEngine: {transformer_engine.__version__}')

import apex
print('Apex: OK')

import flash_attn
print(f'Flash-Attention: {flash_attn.__version__}')

print('All dependencies ready!')
"
```

**Post-install verification gate — do NOT proceed to training or model porting until ALL checks pass:**

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| PyTorch CUDA | `python -c "import torch; assert torch.cuda.is_available()"` | No error |
| PyTorch version unchanged | Compare against version from Step 3 | Exact match |
| Megatron-LM-FL | `python -c "from megatron.plugin.platform import get_platform"` | No ImportError |
| TransformerEngine-FL | `python -c "import transformer_engine"` | No ImportError |
| Apex | `python -c "import apex"` | No ImportError |
| Flash-Attention | `python -c "import flash_attn"` | No ImportError |

If ANY check fails, fix it before moving on. Do not proceed with "we'll fix it later" — dependency issues compound during training and are much harder to debug.

### 5b. Package provenance check

Verify that each FL dependency is installed from the correct source — not from a different workspace or stale editable install:

```bash
pip show megatron-core transformer-engine apex flash-attn 2>/dev/null | grep -E "^(Name|Location|Editable)"
```

For each package:
- If `Editable project location` is shown, verify it points to a directory within the CURRENT workspace (not a different `/workspace/X/` directory)
- If the editable path points to a different workspace, the installed code won't match the code you'll read for debugging — reinstall from the correct source tree within your workspace
- For non-editable installs, verify the `Location` is inside the target conda environment's `site-packages/`

**Cross-workspace editable installs are NEVER acceptable.** Even if two directories are at the same git commit today, they can diverge silently. If the dependency source doesn't exist in your workspace, clone it locally first (`git clone <repo> /workspace/<your_workspace>/<dep>/`), then editable-install from the local clone.

This check prevents the most insidious debugging trap: reading source code from one directory while the runtime uses code from a completely different directory.

## Step 6: Multi-Node Deployment

When setting up multiple nodes for distributed training:

1. Ensure the same conda environment and dependencies are installed on ALL nodes
2. Verify passwordless SSH between nodes:
   ```bash
   ssh -o BatchMode=yes <other_node> hostname
   ```
3. Verify NCCL connectivity between nodes:
   ```bash
   # On each node, check IB/RoCE NICs are up
   ibstat 2>/dev/null || rdma link show 2>/dev/null || echo "No RDMA detected"
   ```
4. Set consistent NCCL environment variables across all nodes:
   ```bash
   export NCCL_IB_DISABLE=0        # Enable IB if available
   export NCCL_NET_GDR_LEVEL=5     # GPUDirect RDMA level
   export NCCL_SOCKET_IFNAME=eth0  # Fallback interface (adjust to actual)
   ```
5. Verify shared filesystem is mounted at the same path on all nodes (for checkpoints and data)

## Error Handling Rules

1. **Network errors** (git clone fails, pip timeout): Tell user to configure proxy. Do NOT try alternative URLs or workarounds.
2. **Build errors** (compilation fails): Report the exact error to user. Do NOT modify dependency source code.
3. **Version mismatch**: Report versions found and let user decide. Do NOT skip version checks by patching code.
4. **Successful builds**: Proceed to next step automatically. Do NOT ask user to confirm after each successful install.

## Alternative: Docker Image

If source builds are too complex, recommend the official training Docker image:

```bash
docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
docker run -itd --gpus all --shm-size=500g --name <name> harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856 /bin/bash
docker exec -it <name> /bin/bash
# In non-interactive shells (agent), use: conda run -n flagscale-train <command>
```

This image has all dependencies pre-installed.

## Download Best Practices

- Always use `wget -c` (resume) instead of plain `wget` for large files.
- For files > 1GB, verify size after download: `ls -lh <file>`.
- Use proxy when available: check `echo $HTTP_PROXY` before downloading.
- For git clone on large repos, use `--depth 1` to avoid fetching full history.
- If a download fails, resume instead of deleting and re-downloading.
- Run large downloads as separate commands, not chained with `&&` or `&`, so failures are isolated.

---

## Related Skills

- `topo-detect` — detect hardware topology after environment setup
- `train-config` — generate training configuration files
- `train-run` — launch training after environment is ready
