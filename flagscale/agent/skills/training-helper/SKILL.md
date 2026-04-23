---
name: training-helper
description: Start, stop, and monitor FlagScale distributed training jobs on GPU servers (single-node and multi-node). Use when user asks to train a model with FlagScale, check training status, monitor loss, start/stop training, or modify training configuration (parallel strategy, batch size, data path, etc.).
keywords:
  - train
  - training
  - 训练
  - 启动训练
  - 开始训练
  - 停止训练
  - 分布式训练
  - GPU
  - loss
  - finetune
  - pretrain
---

# FlagScale Training

Manage FlagScale distributed training: configure, launch, monitor, and stop jobs on GPU servers. Supports both single-node and multi-node training.

## Prerequisites

- SSH access to training server (credentials in TOOLS.md)
- Docker container with FlagScale environment
- FlagScale repo with conda environment

## Workflow

### 1. Connect to Server

SSH into training server, enter Docker container, activate conda env, cd to FlagScale project root.

```bash
sudo docker exec -it <container_name> bash
conda activate <env_name>
cd <flagscale_project_path>
```

### 2. Check GPU Availability

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

**Go/no-go**: Target GPUs must show 0 MiB memory used. If occupied, alert user.

For multi-node, check all nodes:
```bash
# Read hostfile and check each node
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
  host=$(echo "$line" | awk '{print $1}')
  echo "=== $host ==="
  ssh $host "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
done < <hostfile_path>
```

### 3. Configure Training

Two config files:

**Main config** `examples/<model>/conf/train.yaml` — runner & environment:
- `defaults.train`: model size (e.g. `7b`)
- `experiment.runner.nnodes`: node count (1 for single-node, N for multi-node)
- `experiment.runner.nproc_per_node`: GPUs per node
- `experiment.runner.hostfile`: path to hostfile (multi-node only, null for single-node)
- `experiment.envs.CUDA_VISIBLE_DEVICES`: GPU list

**Hostfile** `examples/<model>/conf/hostfile.txt` — multi-node only:
```
# Format: ip slots=<num_gpus> type=<gpu_type>[optional]
# First entry is master node
10.0.0.1 slots=8 type=A100
10.0.0.2 slots=8 type=A100
10.0.0.3 slots=8 type=A100
10.0.0.4 slots=8 type=A100
```

For single-node training, set `hostfile: null` and `nnodes: 1` (or omit both).

**Model config** `examples/<model>/conf/train/<size>.yaml` — model & parallelism:

#### Parallelism Configuration (Megatron-LM)

All Megatron-LM parallel modes are available. The fundamental constraint is:

**DP × TP × PP × CP × EP = total GPUs** (DP is implicit, computed automatically)

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| Tensor Parallel (TP) | `system.tensor_model_parallel_size` | Splits individual layers across GPUs. Use 1-8, typically within a single node for NVLink bandwidth. |
| Pipeline Parallel (PP) | `system.pipeline_model_parallel_size` | Splits layers across pipeline stages. Use when model doesn't fit in TP alone. |
| Virtual Pipeline Parallel (VPP) | `system.num_layers_per_virtual_pipeline_stage` | Interleaves pipeline stages to reduce bubble ratio. Set to a divisor of `num_layers / PP`. |
| Context Parallel (CP) | `system.context_parallel_size` | Splits sequence dimension for long-context training. Use when seq_length is very large (>8K). |
| Sequence Parallel (SP) | `system.sequence_parallel` | Splits activation memory along sequence dim within TP groups. Almost always `true` when TP > 1. |
| Expert Parallel (EP) | `system.expert_model_parallel_size` | Distributes MoE experts across GPUs. Only for MoE models (DeepSeek, Mixtral, etc.). |
| Data Parallel (DP) | (implicit) | Computed as `total_GPUs / (TP × PP × CP × EP)`. Replicates model across DP groups. |
| Distributed Optimizer | `system.use_distributed_optimizer` | Shards optimizer states across DP ranks (ZeRO-1). Almost always `true` for large models. |

#### Communication Overlap Options

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| Overlap Grad Reduce | `system.overlap_grad_reduce` | Overlaps gradient all-reduce with backward pass. |
| Overlap Param Gather | `system.overlap_param_gather` | Overlaps parameter all-gather with forward pass (requires distributed optimizer). |

#### Recomputation (Activation Checkpointing)

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| Recompute Method | `system.recompute_method` | `uniform` or `block`. `uniform` recomputes all layers evenly. |
| Recompute Granularity | `system.recompute_granularity` | `full` or `selective`. `selective` only recomputes attention. |
| Recompute Num Layers | `system.recompute_num_layers` | Number of layers to recompute per pipeline stage. |

#### Batch Size Configuration

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| Micro Batch Size | `model.micro_batch_size` | Per-GPU per-step batch size. |
| Global Batch Size | `model.global_batch_size` | Total batch size. Must be divisible by `DP × micro_batch_size`. Gradient accumulation steps = `global_batch_size / (DP × micro_batch_size)`. |

#### FlagScale-Specific Parallel Features

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| DualPipeV | `system.use_dualpipev` | FlagScale's dual pipeline schedule for DeepSeek-style models. |
| MoE FB Overlap | `system.moe_fb_overlap` | Overlaps MoE forward/backward with communication. |
| Per-Node Task | `experiment.runner.per_node_task` | Each node runs independently (nnodes=1, node_rank=0). |

#### Parallelism Strategy Guidelines

**Single-node (1-8 GPUs):**
- Small models (≤13B): TP only, PP=1
- Medium models (13B-70B): TP=8, PP=1 if fits; otherwise TP=4/8, PP=2
- Always enable `sequence_parallel: true` when TP > 1
- Always enable `use_distributed_optimizer: true`

**Multi-node (16+ GPUs):**
- TP within node (≤8), PP across nodes
- For MoE: EP across nodes, TP within node
- For long sequences: CP=2/4/8, keep TP×CP ≤ node size
- VPP: set `num_layers_per_virtual_pipeline_stage` to reduce PP bubble when PP ≥ 4

**Constraint validation:**
```
total_GPUs = nnodes × nproc_per_node
TP × PP × CP × EP must divide total_GPUs evenly
DP = total_GPUs / (TP × PP × CP × EP)
global_batch_size must be divisible by (DP × micro_batch_size)
num_layers must be divisible by PP
If VPP: num_layers / PP must be divisible by num_layers_per_virtual_pipeline_stage
```

Use `sed -i` for changes, verify with `grep`.

### 4. Start / Stop Training

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

### 5. Monitor Training

#### Determine which node and rank to monitor

**Critical — the loss log is on the last node's last rank of the latest attempt.**

The monitoring target depends on the hostfile content, which may change between runs. Always re-read the hostfile to determine the correct node.

**For multi-node (with hostfile):**

```bash
# Parse hostfile to find the last node
HOSTFILE="<hostfile_path>"
LAST_HOST=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | tail -1 | awk '{print $1}')
NNODES=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l)
LAST_NODE_RANK=$((NNODES - 1))
echo "Last node: $LAST_HOST (rank $LAST_NODE_RANK)"
```

**For single-node (no hostfile):**
```bash
LAST_HOST="localhost"
LAST_NODE_RANK=0
```

#### Find the loss log

Log directory structure (created by torchrun):
```
<details_dir>/host_<node_rank>_<host>/<timestamp>/<run_id>/attempt_<N>/<rank>/stdout.log
```

The loss is printed by the **last rank** in the **last attempt** on the **last node**:

```bash
# Determine details_dir from config or default
DETAILS_DIR="outputs/logs/details"

# Find the log directory for the last node
HOST_DIR="$DETAILS_DIR/host_${LAST_NODE_RANK}_${LAST_HOST}"

# Navigate to latest timestamp -> run_id -> latest attempt -> highest rank
LATEST_TS=$(ls -t "$HOST_DIR" 2>/dev/null | head -1)
RUN_ID=$(ls "$HOST_DIR/$LATEST_TS" 2>/dev/null | head -1)
LATEST_ATTEMPT=$(ls "$HOST_DIR/$LATEST_TS/$RUN_ID" 2>/dev/null | grep "attempt_" | sort -t_ -k2 -n | tail -1)
LAST_RANK=$(ls "$HOST_DIR/$LATEST_TS/$RUN_ID/$LATEST_ATTEMPT" 2>/dev/null | sort -n | tail -1)

LOSS_LOG="$HOST_DIR/$LATEST_TS/$RUN_ID/$LATEST_ATTEMPT/$LAST_RANK/stdout.log"
STDERR_LOG="$HOST_DIR/$LATEST_TS/$RUN_ID/$LATEST_ATTEMPT/$LAST_RANK/stderr.log"
echo "Loss log: $LOSS_LOG"
```

**For multi-node with no shared filesystem** (`no_shared_fs: true`):
- Log directories use `host` instead of `host_<rank>_<ip>` pattern
- Must SSH to the last node to read logs:
```bash
ssh $LAST_HOST "tail -5 $LOSS_LOG"
```

#### Health check

Training is normal when **both** conditions are true:
1. Loss is being printed (new lines in loss log)
2. GPUs are active (utilization > 0%, memory occupied)

```bash
# Single-node
tail -1 $LOSS_LOG && echo "===GPU===" && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Multi-node — check all nodes
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
  host=$(echo "$line" | awk '{print $1}')
  echo "=== $host ==="
  ssh $host "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
done < "$HOSTFILE"
```

#### Anomaly detection

| Symptom | Likely Cause |
|---------|-------------|
| GPU memory occupied but 0% util, no new loss | Hanging / NCCL deadlock |
| GPU memory 0, no new loss | Training crashed or was killed |
| `nan` in loss or grad norm | Numerical instability, reduce LR or check data |
| `skipped iterations` increasing | Loss scale too high, will auto-recover |
| Loss suddenly spikes | Data issue or LR too high |
| NCCL timeout errors in stderr | Network issue between nodes, check IB/RoCE |
| OOM (Out of Memory) | Reduce micro_batch_size, increase TP/PP, enable recomputation |

When anomaly detected: check stderr logs on the relevant node, report findings, suggest fix.

```bash
# Check errors — single-node
tail -20 $STDERR_LOG

# Check errors — multi-node (check last node first, then others)
ssh $LAST_HOST "tail -20 $STDERR_LOG"
```

#### Periodic monitoring

Report every N steps (as requested by user):
- **iteration** / total
- **lm loss** value and trend
- **elapsed time per iteration** (ms)
- **grad norm**
- **GPU status** (utilization + memory) on all nodes
- **Node health** — verify all nodes are responsive

Alert immediately if training stops or anomalies detected.
