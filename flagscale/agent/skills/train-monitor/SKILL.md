---
name: train-monitor
description: Monitor FlagScale distributed training jobs. Locate logs, check training health, detect anomalies (NaN loss, OOM, NCCL timeout, hangs), parse training metrics (loss, grad norm, throughput), and provide periodic status reports. Supports single-node and multi-node monitoring.
keywords:
  - monitor
  - monitoring
  - loss
  - log
  - logs
  - status
  - check
  - anomaly
  - OOM
  - nan
  - hang
  - grad norm
  - throughput
  - 监控
  - 日志
  - 训练状态
  - 训练监控
  - 查看loss
parameters:
  - name: exp_dir
    description: Experiment output directory (from train.yaml experiment.exp_dir)
  - name: nproc_per_node
    description: Number of GPUs per node
    default: "8"
---

# FlagScale Training Monitor

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies, and report metrics.

## Log Directory Structure

FlagScale training logs follow this structure:

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   └── details/host_0_<hostname>/
│       ├── 20260424_153816.588538/           # timestamp dir
│       │   └── default_<hash>/attempt_0/
│       │       ├── 0/stdout.log  stderr.log  # rank 0
│       │       ├── 1/stdout.log  stderr.log  # rank 1
│       │       └── .../
│       └── 20260424_162209.763893/           # newer run
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key facts:
- Each training launch creates a NEW timestamp directory
- Multiple runs accumulate — always find the LATEST timestamp dir
- Training metrics (loss, iteration) are printed by the **last rank**
- Errors can appear on **any rank**'s stderr

---

## Step 1: Locate Latest Logs

**ALWAYS run these commands first.** Never guess log paths.

```bash
EXP_DIR=<exp_dir from train.yaml>
LATEST=$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" 2>/dev/null | head -1)
NPROC={nproc_per_node}
LAST_RANK=$((NPROC - 1))
```

Verify the path is valid:
```bash
echo "Latest log dir: $LATEST"
echo "Attempt dir: $ATTEMPT"
ls "$ATTEMPT/" 2>/dev/null | head -5
```

If `LATEST` or `ATTEMPT` is empty, training either hasn't started or logs are in an unexpected location. Check `EXP_DIR` value in `train.yaml`.

### Multi-Node: Finding the Loss Log

For multi-node training, the loss log is on the **last node's last rank**:

```bash
LAST_HOST_DIR=$(ls -d ${EXP_DIR}/logs/details/host_*/ 2>/dev/null | sort | tail -1)
LATEST=$(ls -d ${LAST_HOST_DIR}[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" 2>/dev/null | head -1)
tail -30 ${ATTEMPT}/${LAST_RANK}/stdout.log
```

---

## Step 2: Health Check

Training is normal when BOTH conditions are true:
1. Loss is being printed (new lines in stdout.log)
2. GPUs are active (utilization > 0%, memory occupied)

### Quick Health Check

```bash
# Latest loss line + GPU status
tail -1 ${ATTEMPT}/${LAST_RANK}/stdout.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

### Process Check

```bash
cat ${EXP_DIR}/logs/pids/*.pid 2>/dev/null | xargs -I{} ps -p {} -o pid,stat,etime --no-headers
```

If no processes found, training has stopped. Check stderr for the reason.

### Training Progress

```bash
# Last 30 lines of training output (last rank)
tail -30 ${ATTEMPT}/${LAST_RANK}/stdout.log

# Errors on rank 0 (most common error location)
tail -30 ${ATTEMPT}/0/stderr.log

# Errors on last rank
tail -30 ${ATTEMPT}/${LAST_RANK}/stderr.log
```

---

## Step 3: Anomaly Detection

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| GPU memory occupied but 0% util, no new loss | Hanging / NCCL deadlock | Check stderr on all ranks, check network |
| GPU memory 0, no new loss | Training crashed or was killed | Check stderr for error, check PID |
| `nan` in loss or grad norm | Numerical instability | Reduce LR, check data for corruption |
| `skipped iterations` increasing | Loss scale too high | Will auto-recover, monitor |
| Loss suddenly spikes | Data issue or LR too high | Check data batch, check LR schedule |
| NCCL timeout errors in stderr | Network issue between nodes | Check IB/RoCE, check firewall |
| OOM (Out of Memory) | Model too large for GPU memory | Reduce micro_batch_size, increase TP/PP, enable recomputation |

### Check All Ranks for Errors

When anomaly detected, check stderr on ALL ranks (not just rank 0):

```bash
for r in $(ls ${ATTEMPT}/ 2>/dev/null); do
  [ -f "${ATTEMPT}/$r/stderr.log" ] || continue
  errors=$(grep -c -iE "error|exception|traceback|oom|killed" "${ATTEMPT}/$r/stderr.log" 2>/dev/null)
  if [ "$errors" -gt 0 ]; then
    echo "=== rank $r ($errors errors) ==="
    tail -10 ${ATTEMPT}/$r/stderr.log
  fi
done
```

### Multi-Node Error Check

```bash
for host_dir in ${EXP_DIR}/logs/details/host_*/; do
  host=$(basename "$host_dir")
  latest=$(ls -d ${host_dir}[0-9]*/ 2>/dev/null | sort | tail -1)
  attempt=$(find "$latest" -type d -name "attempt_*" 2>/dev/null | head -1)
  [ -z "$attempt" ] && continue
  echo "=== $host ==="
  for r in $(ls "$attempt/" 2>/dev/null); do
    [ -f "${attempt}/$r/stderr.log" ] || continue
    errors=$(grep -c -iE "error|exception|traceback" "${attempt}/$r/stderr.log" 2>/dev/null)
    [ "$errors" -gt 0 ] && echo "  rank $r: $errors errors" && tail -3 "${attempt}/$r/stderr.log"
  done
done
```

---

## Step 4: Periodic Monitoring Report

Report the following metrics at user-requested intervals:

| Metric | Source | Command |
|--------|--------|---------|
| Iteration / total | stdout.log last rank | `grep "iteration" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -1` |
| LM loss | stdout.log last rank | `grep "lm loss" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -5` |
| Elapsed time per iteration | stdout.log last rank | `grep "elapsed time per iteration" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -1` |
| Grad norm | stdout.log last rank | `grep "grad norm" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -5` |
| GPU utilization + memory | nvidia-smi | `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader` |
| Node health (multi-node) | SSH | `ssh <node> nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` |

### Loss Trend Analysis

```bash
# Extract loss values for trend analysis
grep "lm loss" ${ATTEMPT}/${LAST_RANK}/stdout.log | awk '{for(i=1;i<=NF;i++) if($i=="loss:") print $(i+1)}' | tail -50
```

A healthy training run shows:
- Loss decreasing over time (not necessarily monotonically)
- Grad norm stable (not exploding)
- Elapsed time per iteration consistent (no sudden slowdowns)
- All GPUs at high utilization (>90%)

### Alert Conditions

Alert immediately if ANY of these occur:
- No new loss output for > 5 minutes (training may have stopped)
- Loss becomes `nan` or `inf`
- Grad norm exceeds 100x its typical value
- GPU utilization drops to 0% on any GPU
- Any node becomes unreachable (multi-node)
- stderr shows new errors

---

## Checkpoint Monitoring

```bash
# Check latest checkpoint
ls -lt ${EXP_DIR}/checkpoints/ 2>/dev/null | head -5

# Check if checkpoint is being written (file size changing)
ls -lh ${EXP_DIR}/checkpoints/iter_*/mp_rank_00/ 2>/dev/null | tail -3
```

---

## TensorBoard / WandB

If TensorBoard logs exist:
```bash
ls ${EXP_DIR}/tensorboard/ 2>/dev/null
```

If WandB logs exist:
```bash
ls ${EXP_DIR}/wandb/ 2>/dev/null
```

These provide richer visualization but require a browser. For CLI-based monitoring, use the log parsing commands above.

---

## Common Issues During Monitoring

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Log file exists but empty | Training crashed during startup | Check stderr.log for import errors or CUDA issues |
| Loss stuck at same value | Learning rate too low or gradient issue | Check LR schedule, verify grad norm is non-zero |
| Loss oscillating wildly | LR too high or batch size too small | Reduce LR or increase effective batch size |
| `Killed` in stderr | OOM killed by system | Check `dmesg \| tail`, reduce memory usage |
| Log timestamps stop updating | Process hung (NCCL deadlock, data loader stuck) | Check GPU util — if 0%, likely NCCL hang; if >0%, likely data loader |

---

## Related Skills

- `train-run` — launch, stop, and manage training jobs
- `train-config` — generate and validate training configuration
- `precision-alignment` — verify numerical alignment between implementations
- `topo-detect` — detect hardware topology for diagnosing performance issues
