---
name: train-run
description: Launch, stop, and manage FlagScale distributed training jobs. Covers server connection, environment checks, GPU availability, preflight validation, training launch (CLI and legacy), stop commands, log directory structure, and quick verification paths.
keywords:
  - train
  - training
  - launch
  - start
  - stop
  - run
  - preflight
  - dry run
  - dryrun
  - 训练
  - 启动训练
  - 开始训练
  - 停止训练
  - 分布式训练
  - GPU
  - finetune
  - pretrain
parameters:
  - name: model_name
    description: Model name (e.g., qwen3, llama3)
  - name: exp_dir
    description: Experiment output directory
---

# FlagScale Training Launch

Launch, stop, and manage FlagScale distributed training jobs on GPU servers.

## Prerequisites

- SSH access to training server
- Docker container with FlagScale environment (or bare metal with conda)
- FlagScale repo with conda environment activated
- Training config files ready (see train-config skill)

---

## Step 1: Connect to Server

SSH into training server, enter Docker container, activate conda env, cd to FlagScale project root.

```bash
sudo docker exec -it <container_name> bash
conda activate <env_name>
cd <flagscale_project_path>
```

---

## Step 2: Check Environment and GPU Availability

### Determine Environment Type

Do this once per server, remember the result:

```bash
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
  echo "CONTAINER environment"
else
  echo "BARE METAL environment"
fi
```

### Check GPU Status

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

**Interpreting GPU status — depends on environment**:

- **Container**: `nvidia-smi` shows memory/utilization from ALL containers sharing the GPU, but only shows PIDs from the current container. If GPUs show memory occupied but no processes visible, this means OTHER containers are using those GPUs — NOT leaked memory. Report to user: "GPUs X-Y are in use by other containers, GPUs Z are available."
- **Bare metal**: all processes are visible. If GPUs show memory occupied with no PID, that is genuinely abnormal (zombie GPU memory). Can try `nvidia-smi --gpu-reset` or report to user.

**Go/no-go**: Target GPUs must show near-zero memory used. If occupied, alert user with the correct explanation based on environment type.

### Multi-Node GPU Check

```bash
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
  host=$(echo "$line" | awk '{print $1}')
  echo "=== $host ==="
  ssh $host "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
done < <hostfile_path>
```

---

## Step 3: Preflight Check

**ALWAYS run this before starting training.** Environment may have changed since last session.

### 3a. Core Dependencies

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')
from megatron.plugin.platform import get_platform
print(f'Megatron platform: {get_platform()}')
import transformer_engine
print(f'TransformerEngine: {transformer_engine.__version__}')
import apex; print('Apex: OK')
import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')
print('All dependencies OK')
"
```

If ANY import fails, stop and tell the user which dependency is broken. Suggest running `/skill env-setup` to fix.

### 3b. GPU Availability

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

All target GPUs must show near-zero memory usage.

### 3c. Data Path Validation

```bash
DATA_PATH="<data_path from config>"
ls -lh ${DATA_PATH}.bin ${DATA_PATH}.idx
```

If files are missing, stop and tell the user. Suggest running `/skill data-prep`.

### 3d. Topology Freshness (Optional)

If memory contains `topo_compute` from a previous topo-detect run, do a quick sanity check:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
```

If GPU count or model differs from what's in memory, warn the user that topology data is stale.

### 3e. Dry Run

```bash
flagscale train <model> --dryrun
```

If dry run fails, report the error. Do NOT proceed to actual training.

**Only after all checks pass, proceed to start training.**

---

## Step 4: Start / Stop Training

```bash
# Start (CLI)
flagscale train <model>
# Start (legacy)
python run.py --config-path ./examples/<model>/conf --config-name train action=run

# Stop (CLI)
flagscale train <model> --stop
# Stop (legacy)
python run.py --config-path ./examples/<model>/conf --config-name train action=stop

# Dry run (validate config only)
flagscale train <model> --dryrun
```

---

## Log Directory Structure

FlagScale training logs are organized as follows. Understanding this structure is CRITICAL — you MUST use the correct commands to find logs, never guess paths.

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   ├── scripts/host_0_<hostname>_run.sh      # actual launch script
│   ├── scripts/host_0_<hostname>_stop.sh     # stop script
│   └── details/host_0_<hostname>/
│       ├── 20260424_153816.588538/           # timestamp dir (YYYYMMDD_HHMMSS.us)
│       │   └── default_<hash>/attempt_0/
│       │       ├── 0/stdout.log  stderr.log  # rank 0
│       │       ├── 1/stdout.log  stderr.log  # rank 1
│       │       └── .../                      # one dir per rank
│       └── 20260424_162209.763893/           # another run (newer!)
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key points:
- `exp_dir` comes from `experiment.exp_dir` in `train.yaml`
- Each training launch creates a NEW timestamp directory under `details/host_X_<hostname>/`
- Multiple runs accumulate — you MUST find the LATEST timestamp dir, not the first one
- Each rank (GPU process) has its own `stdout.log` and `stderr.log`
- Rank 0's stdout.log contains the main training output (loss, iteration, etc.)
- stderr.log contains errors, warnings, and import failures

### Finding the Latest Logs

```bash
EXP_DIR=$(grep 'exp_dir:' examples/<model>/conf/train.yaml | awk '{print $2}')
LATEST=$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" | head -1)
tail -30 ${ATTEMPT}/0/stdout.log
tail -30 ${ATTEMPT}/0/stderr.log
```

One-liners:
```bash
tail -30 "$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ | sort | tail -1)"/*/attempt_0/0/stdout.log
tail -30 "$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ | sort | tail -1)"/*/attempt_0/0/stderr.log
```

**NEVER do this:**
- Don't hardcode timestamp dirs like `20260424_153816.588538`
- Don't use `find -name stdout.log` without sorting — it may return old runs
- Don't use `sleep N && tail` — check directly

---

## Quick Verification Paths

When the user wants to quickly verify a training setup works:

1. **Minimal config**: `train_iters: 3-5`, `micro_batch_size: 1`, `global_batch_size: DP × 1`
2. **Single GPU first**: Start with 1 GPU (TP=1, PP=1, DP=1) before scaling
3. **Smallest dataset**: Use the smallest available split or demo data
4. **Dry run**: Use `flagscale train <model> --dryrun` to validate config without launching
5. **Stage-by-stage**: If the recipe has stages, run one stage at a time to isolate failures

### Common Pitfalls

- `micro_batch_size` must divide `global_batch_size / (TP * PP * DP)`
- Megatron checkpoint format: `--load` path must contain `latest_checkpointed_iteration.txt`
- Multi-node: verify NCCL connectivity before launching full training
- OOM on first iteration: reduce `micro_batch_size` or enable activation checkpointing before reducing parallelism

---

## Error Handling

### Launch Failures

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `ModuleNotFoundError: megatron.*` | Megatron-LM-FL not installed or wrong PYTHONPATH | Check `pip list \| grep megatron`, reinstall if needed |
| `NCCL error: unhandled system error` | Network issue between nodes or wrong NCCL config | Check `NCCL_SOCKET_IFNAME`, verify SSH connectivity |
| `RuntimeError: CUDA out of memory` | Model too large for GPU memory | Reduce `micro_batch_size`, enable activation checkpointing, or increase TP/PP |
| `FileNotFoundError: data path` | Data files missing or wrong path in config | Verify data path with `ls`, check train.yaml data section |
| `Address already in use` | Previous training process still running | Kill old processes: `pkill -f torchrun`, wait, retry |
| `Hydra config error` | YAML syntax error or missing required field | Run `flagscale train <model> --dryrun` to validate config |
| Process starts but exits silently | Import error or early crash | Check stderr.log of rank 0 immediately |

### Recovery Steps

1. Read FULL stderr.log (not just tail) — multiple errors may exist
2. Fix ALL identified issues before relaunching
3. Clean Hydra cache if config was changed: `rm -rf outputs/<exp>/hydra/ outputs/<exp>/logs/scripts/`
4. Never retry more than once without a clear diagnosis

---

## Related Skills

- `train-config` — generate and validate training configuration YAML files
- `train-monitor` — monitor running training jobs, check health, detect anomalies
- `env-setup` — install FlagScale and all dependencies
- `topo-detect` — detect hardware topology for parallelism planning
- `data-prep` — prepare training data in Megatron binary format
