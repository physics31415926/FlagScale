# FlagScale Training Environment Setup — Summary

## Overview

Set up a complete FlagScale training environment on GPU servers. All dependencies use FL-customized versions (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention).

## Strategy

Environment setup is a constraint satisfaction problem: collect ALL constraints first, solve for compatible versions, then install once.

Key rules:
- ALL installs go into the target conda environment — NEVER install into base or current
- Try pip install first (fast), fall back to source build if pip fails
- Never modify dependency source code to work around errors
- ALL FL-customized dependencies are MANDATORY — do not skip any
- If user asks to create a new environment, create a new one (never silently reuse existing)
- NEVER copy packages between environments using `cp -r` from site-packages

## Installation Flow

### Step 1: Constraint Collection (NO installs)

Collect before installing anything:

1. **Hardware constraint**: `nvidia-smi` → driver version → max CUDA version
   - Driver 570.x → CUDA ≤ 12.8
   - Driver 560.x → CUDA ≤ 12.6
   - Driver 550.x → CUDA ≤ 12.4
   - Driver 535.x → CUDA ≤ 12.4

2. **Framework constraint**: Read `requirements.txt` and Megatron-LM-FL setup requirements

3. **Existing environment check**: `conda env list`, check for stale editable installs

### Step 2: Create Conda Environment

```bash
conda create -n {env_name} python={python_version} -y
conda run -n {env_name} pip install torch==<version> --index-url https://download.pytorch.org/whl/cu<cuda>
```

### Step 3: Install FL Dependencies (in order)

| Package | Method | Notes |
|---------|--------|-------|
| FlagScale | `pip install -e .` | Editable install from repo root |
| Megatron-LM-FL | `pip install -e .` | From FlagScale/Megatron-LM-FL/ |
| TransformerEngine-FL | pip wheel or source build | Needs `--recursive` for git clone |
| Apex | pip wheel or source build | CUDA extensions required |
| Flash-Attention | pip wheel or source build | Match CUDA + PyTorch version |

### Step 4: Verification

```bash
conda run -n {env_name} python -c "
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
import megatron; print('Megatron-LM-FL: OK')
import transformer_engine; print('TE-FL:', transformer_engine.__version__)
import apex; print('Apex: OK')
import flash_attn; print('Flash-Attention:', flash_attn.__version__)
"
```

## Common Pitfalls

- ❌ Installing into base environment → conflicts with system packages
- ❌ Reusing existing environment → stale editable installs from other workspaces
- ❌ Skipping FL dependencies → training fails or produces incorrect results
- ❌ PyTorch CUDA version mismatch with driver → `CUDA error: no kernel image`
- ❌ `cp -r` from site-packages → pip metadata broken, silent dependency issues
- ❌ Chaining large downloads with `&&` → one failure kills the chain

## Multi-Node Deployment

1. Install on one node first, verify all tests pass
2. Copy conda environment to shared storage or replicate on each node
3. Verify passwordless SSH between all nodes
4. Verify NCCL communication: `python -c "import torch.distributed; ..."`
5. Verify shared filesystem is mounted at the same path on all nodes

## Docker Alternative

```bash
docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
docker run -itd --gpus all --shm-size=500g --name <name> <image> /bin/bash
```

Pre-installed with all dependencies. Use when source builds are too complex.

## Error Handling

- **Network errors**: Tell user to configure proxy. Don't try alternative URLs.
- **Build errors**: Report exact error. Don't modify dependency source code.
- **Version mismatch**: Report versions found, let user decide.
- **Successful builds**: Proceed automatically, don't ask for confirmation.

## When to Load Full Skill

Load the full `env-setup` skill when:
- Setting up environment from scratch on a new server
- Debugging dependency installation failures
- Building FL dependencies from source (TransformerEngine-FL, Apex)
- Setting up multi-node deployment
- Resolving CUDA/driver compatibility issues

This summary covers the essentials. For detailed version compatibility tables, source build commands, and troubleshooting, load the full skill.
