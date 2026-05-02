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
    description: Directory to clone source dependencies
    default: /opt/flagscale/deps
---

# FlagScale Training Environment Setup

Set up a complete FlagScale training environment on a GPU server. All dependencies use FL-customized versions.

## Strategy

Environment setup is a constraint satisfaction problem. Collect ALL constraints first, solve for compatible versions, then install once.

1. ALL installs go into the target conda environment — NEVER install into base or current environment. Use `conda run -n <env> pip install ...` for every pip command. To check dependency versions without installing, read setup.cfg/pyproject.toml from the source repo or use `pip index versions <pkg>`.
2. Try pip install with pinned versions (fast, from FlagScale PyPI)
3. If pip fails, fall back to source clone + build
4. Never modify dependency source code to work around errors — report to user
5. After any large `pip install`, verify critical packages were not unexpectedly upgraded: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
6. **Auto-fetch FL dependencies**: When Megatron-LM-FL or TransformerEngine-FL source code is needed (for analysis, compilation, or debugging) and is not available locally, pull the latest automatically — don't ask the user. Repos: `https://github.com/flagos-ai/Megatron-LM-FL.git`, `https://github.com/flagos-ai/TransformerEngine-FL.git` (use `--recursive` for TE-FL)
7. **ALL FL-customized dependencies are MANDATORY.** Do NOT skip Megatron-LM-FL, TransformerEngine-FL, Apex, or Flash-Attention. These are not optional — FlagScale training will fail or produce incorrect results without them. If one is difficult to install, try the source build fallback. Only skip a dependency if the user explicitly requests it after being warned of the consequences.
8. **If the user asks to create a new environment, create a new environment.** Do not reuse an existing one, even if it appears to have the right packages. Existing environments may have editable installs pointing to other workspaces, patched packages, or stale versions. A fresh environment is the only way to guarantee a clean, reproducible baseline. If you believe reusing is genuinely better, explain why and ask — but do not silently substitute.
9. **NEVER copy packages between environments using `cp -r` from site-packages.** This bypasses pip's metadata tracking — pip won't know the package exists, so dependency resolution, upgrades, and uninstalls all break silently. Always install via `pip install` (from wheel, PyPI, or source build). If a prebuilt wheel isn't available, build from source — it takes longer but produces a properly registered package.

## Step 1: Constraint Collection (NO installs in this step)

Collect all version constraints before installing anything.

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

### 1b. Framework constraint — required PyTorch / Python versions

For FlagScale: read `requirements.txt` and Megatron-LM-FL's setup requirements.
For other frameworks (ESPnet, DeepSpeed, Fairseq, etc.): fetch setup.cfg / pyproject.toml from the source repo:

```bash
# Option A: fetch from GitHub raw URL (no install needed)
web_fetch https://raw.githubusercontent.com/<org>/<repo>/master/setup.cfg

# Option B: clone with --depth 1 and read locally
git clone --depth 1 <repo_url> /tmp/<repo>
grep -A 20 "install_requires" /tmp/<repo>/setup.cfg
grep "python_requires" /tmp/<repo>/setup.cfg
```

Extract: `torch>=X.Y.Z`, `python_requires>=X.Y`, and any other critical deps.

### 1c. Recipe/config constraint — additional packages

Read the specific training recipe or config to check for additional requirements:
- Flash-Attention (requires specific CUDA compute capability)
- DeepSpeed, Apex, TransformerEngine
- Domain-specific packages (e.g., ESPnet's `espnet[s2t]` extras)

### 1d. Solve — find the intersection

Write out the constraint table explicitly, then present options to the user:

```
Example:
  Driver 535.154 → max CUDA 12.4
  ESPnet setup.cfg → torch >= 2.3.1, python >= 3.10
  Available PyTorch wheels for cu121: 2.3.1, 2.4.0, 2.4.1, 2.5.0, ...
  Available PyTorch wheels for cu124: 2.4.0, 2.5.0, ...

  Options:
    A. torch==2.3.1+cu121 — lowest compatible, best stability with third-party libs (recommended)
    B. torch==2.4.0+cu124 — newer, more CUDA features
    C. torch==2.5.0+cu124 — latest, flash-attn/apex may not be adapted yet
  Python: 3.11. Which do you prefer?
```

Default recommendation: lowest PyTorch + highest compatible CUDA (most stable + best GPU features). But always let the user choose.

If no valid intersection exists, STOP and tell the user why (e.g., framework requires torch>=2.6 but no compatible wheel for this driver).

## Step 2: Conda Environment

```bash
conda create -n {env_name} python={python_version} -y
# In non-interactive shells (agent), use: conda run -n {env_name} <command>
# In interactive shells (user), use: conda activate {env_name}
```

Verify:
```bash
python --version
```

## Step 3: Install FlagScale

From the FlagScale project root:

```bash
pip install -e ".[cuda-train]"
```

This installs FlagScale itself plus base pip dependencies (PyTorch, sentencepiece, transformers, tiktoken, etc.).

Verify:
```bash
flagscale --help
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

**Important**: `requirements/cuda/train.txt` includes a megatron-core whl from the FlagScale PyPI. This is a base version — it may lack modules like `megatron.plugin.platform` that FlagScale's training code requires. If training fails with import errors from megatron, proceed to Step 3.

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

### 4d. Flash-Attention 2

**CRITICAL**: Always use `--no-deps` when installing flash-attn. Without it, pip may upgrade PyTorch to an incompatible version, causing cascading failures (triton mismatch, CUDA version conflicts). The PyTorch version was already pinned in Step 3 — do not let flash-attn override it.

```bash
git clone --branch v2.8.1 --depth 1 https://github.com/Dao-AILab/flash-attention.git {deps_dir}/flash-attention
cd {deps_dir}/flash-attention
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
    pip install --no-build-isolation --no-deps . -v
```

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
