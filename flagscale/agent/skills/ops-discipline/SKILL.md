---
name: ops-discipline
description: Operational discipline rules for FlagScale infrastructure work. Covers reading strategy, shell safety, dependency resolution, training launch, checkpoint resume, multi-node, and diagnosis patterns.
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
requires: []
suggests: []
---

# Operational Discipline

Detailed operational rules for infrastructure work. The system prompt covers principles; this skill covers execution details.

---

## Reading strategy — depth over speed

- **Understand before implementing.** For complex tasks (env setup, config, training launch), read the relevant docs, example configs, and source code BEFORE writing anything. This is productive work, not wasted time.
- **Read complete files, not fragments.** One complete read beats ten partial reads.
- **First read: full file.** Note key line numbers. Subsequent reads: targeted ranges.
- **Record key findings in memory_write** so they survive context compaction.
- **Never re-read a file you read in the last 5 turns** unless it was modified.
- **Breadth matters:** for a training config, read at least: the getting-started doc, an existing example config, and the model's source code. For env setup, read install docs and version constraints.

---

## Shell command rules

- Prefer `grep -rn "pattern" . --include="*.py"` for code search.
- Use `head`/`tail` ONLY for quick previewing. Never truncate error logs you need to diagnose.
- To stop FlagScale training: prefer `flagscale train <model> --config <config> --stop`. Fallback: `cat outputs/<exp>/logs/pids/* | xargs kill -9`. NEVER use broad `ps | grep | kill`.
- NEVER run the same command twice in a row. If results are unclear, try a DIFFERENT diagnostic.
- NEVER modify third-party source code to work around build errors.
- For large downloads: `wget -c` or `curl -C -`. Execute as SEPARATE commands, not combined with `&&`.
- After any download, verify with `ls -lh <file>`.
- Download speed < 500 KB/s for multi-GB file → check proxy, then STOP and ask user.

## Environment awareness

- FIRST thing on any new server: `nvidia-smi`, `cat /etc/os-release`, `which conda`, `echo $CUDA_HOME`. Save to workspace_state.
- Check disk space (`df -h`) before large downloads or builds.
- Check GPU memory (`nvidia-smi`) before launching training.

## Training launch discipline

- NEVER launch without verifying: (1) config valid, (2) data paths exist, (3) model weights accessible, (4) GPU count matches config.
- After launch, IMMEDIATELY check logs (within 30 seconds) for startup errors.
- If training fails within first 100 iterations → config or environment issue, not training issue.

## FlagScale log structure and error detection

FlagScale logs are organized per-rank under the output directory:
```
outputs/<exp>/logs/details/host_<N>_<hostname>/<timestamp>/<run_id>/attempt_<N>/<rank>/
  ├── stdout.log   (training metrics, progress)
  └── stderr.log   (errors, warnings, stack traces)
```

**Critical rule: ALWAYS check stderr.log files after launch.**
- Training can appear "running" (wandb init, imports) while actually crashing on rank > 0.
- Errors like missing APEX, TransformerEngine, flash_attn show up ONLY in stderr.log.
- Use: `find outputs/<exp>/logs -name 'stderr.log' -newer <ref_file> -exec tail -5 {} +`
- Or use the monitor tool with `output_dir=outputs/<exp>` to auto-scan all stderr files.

**Don't monitor stdout for 300s when stderr already has the answer.**
- Wait 10-15 seconds after launch
- Check stderr.log for ALL ranks (not just rank 0)
- If stderr shows errors → training is DEAD, act on the error immediately
- Common stderr-only errors: APEX not installed correctly, gradient_accumulation_fusion requires fused kernels, flash_attn version mismatch

## Trust nothing, verify everything

- After `pip install X`: `python -c "import X; print(X.__version__)"`.
- After `git clone`: verify directory exists with expected content.
- After writing config: read it back and verify key values.
- After starting a process: verify it's running.

## Understand target state before acting

- Before creating directory structures or config files: FIRST examine a working example.
- File format generation: NEVER generate from assumptions. Find and read an existing example first.

## Dependency chain awareness

- When skipping/removing ANY component, IMMEDIATELY scan configs for parameters that depend on it.
- After modifying config files, check if the build system caches old configs (e.g., Hydra caches resolved config in `outputs/<exp>/hydra/`).
- Think in dependency chains: A → B → C. If C changes, trace impact forward.
- CUDA/cuDNN conflicts: system LD_LIBRARY_PATH often has older cuDNN than PyTorch expects. Fix: prepend PyTorch's bundled nvidia/cudnn/lib.

## APEX installation modes

APEX has two install modes with VERY different capabilities:
- **Pure Python** (`pip install apex` or `pip install -v --no-build-isolation --no-cuda-ext .`): Only provides Python utilities (amp, normalization wrappers). Does NOT provide fused CUDA kernels.
- **Full CUDA** (`pip install -v --no-build-isolation .` with CUDA toolkit): Provides fused kernels: `fused_weight_gradient_mlp_cuda`, `fused_adam`, `fused_layer_norm_cuda`, etc.

**If APEX is installed pure-python:**
- `gradient_accumulation_fusion: true` will CRASH (requires `fused_weight_gradient_mlp_cuda`)
- `bias_gelu_fusion: true` may crash (requires fused kernels)
- FusedAdam optimizer unavailable

**Action when pure-python APEX detected:**
1. Disable ALL fusion flags that require APEX CUDA: `gradient_accumulation_fusion: false`, `bias_gelu_fusion: false`
2. OR install APEX with CUDA extensions (requires matching CUDA toolkit version)
3. Verify: `python -c "import fused_weight_gradient_mlp_cuda"` — if ImportError, it's pure-python

## Dependency resolution — constraint solving, then one-shot install

NEVER blindly install packages and fix conflicts after the fact.

**Phase 1: Collect constraints** (NO installs)
1. Hardware: `nvidia-smi` → driver version → max CUDA version
2. Framework: check setup.cfg/pyproject.toml → PyTorch/Python version bounds
3. Recipe/config: additional packages and their version requirements
4. PyTorch ↔ CUDA: wheel CUDA version must be ≤ driver's max

**Phase 2: Solve** — write constraint table, present options, recommend one. If no valid intersection, STOP.

**Phase 3: One-shot install**
1. `conda create --prefix <root>/envs/<name> python=<version> -y`
2. `conda run --prefix <env> pip install torch==<version> --index-url https://download.pytorch.org/whl/<cu>`
3. `conda run --prefix <env> pip install <framework>` (version pins or `--no-deps` if it upgrades torch)
4. Verify: `conda run --prefix <env> python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

## Checkpoint resume

- Check: `ls outputs/<exp>/checkpoints/`. FlagScale auto-saves at configured intervals.
- To resume: keep same YAML config and re-launch. FlagScale loads latest checkpoint automatically.
- To start fresh: delete or rename checkpoint directory FIRST (with confirmation).
- After resuming, verify starting iteration matches checkpoint, not 0.

## Multi-node awareness

- Requires: SSH passwordless access, consistent environment, shared/replicated data paths, correct NCCL env vars.
- Key NCCL vars: `NCCL_IB_DISABLE` (set 1 if no InfiniBand), `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG=INFO`.
- Hostfile: one line per node, `<hostname> slots=<num_gpus>`. Verify SSH before launching.
- When diagnosing multi-node failures, check logs on ALL nodes.

## Root cause diagnosis

- dtype mismatches (fp32 in bf16 pipelines) are architecture-level. Trace dtype from source (RoPE, embedding, normalization) rather than adding `.to(dtype)` at error site.
- Cascading TypeError/AttributeError on module init → read the COMPLETE base class API, fix ALL mismatches at once.
- Before calling any base class method, read its IMPLEMENTATION, not just signature.

## Fail-fast preflight checklist

Before operations >30 seconds:
- **Model loading**: verify state_dict keys/shapes match BEFORE loading to GPU
- **Checkpoint conversion**: compare key counts/shapes between source and target
- **Training launch**: validate config arithmetic, verify ALL dependencies importable
- **Memory budget**: `params × 2 (bf16) + grads × 2 + optimizer × (8/DP)` — if exceeds GPU memory, don't launch
- **Config arithmetic**: `global_batch_size % (micro_batch_size × DP) == 0`, `num_heads % TP == 0`
