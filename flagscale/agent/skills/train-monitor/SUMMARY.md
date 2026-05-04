# FlagScale Training Monitor — Summary

## Overview

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies (NaN loss, OOM, NCCL timeout, hangs), parse metrics, and report status.

## Log Directory Structure

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   └── details/host_0_<hostname>/
│       └── <timestamp_dir>/default_<hash>/attempt_0/
│           ├── 0/stdout.log  stderr.log      # rank 0
│           ├── 1/stdout.log  stderr.log      # rank 1
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key facts:
- Each launch creates a NEW timestamp directory — always find the LATEST
- Training metrics (loss, iteration) are printed by the **last rank**
- Errors can appear on **any rank**'s stderr

## Quick Monitoring Steps

### 1. Locate Latest Logs
Use `find_latest_log` tool, or manually:
```bash
LATEST=$(ls -td ${EXP_DIR}/logs/details/host_0_*/*/default_*/attempt_0 2>/dev/null | head -1)
```

### 2. Check for Errors (stderr first)
```bash
for f in ${LATEST}/*/stderr.log; do [ -s "$f" ] && echo "=== $f ===" && tail -20 "$f"; done
```

### 3. Parse Training Metrics
```bash
# Loss trend (last rank)
LAST_RANK=$(ls -d ${LATEST}/*/ | sort -t/ -k$(echo ${LATEST}/*/ | tr '/' '\n' | wc -l) -n | tail -1)
grep -oP 'lm loss[^|]*\|\s*lm loss value:\s*\K[\d.e+-]+' ${LAST_RANK}/stdout.log | tail -20
```

### 4. Check GPU Health
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
```

## Anomaly Detection

| Anomaly | Detection | Severity |
|---------|-----------|----------|
| **NaN/Inf loss** | `grep -i 'nan\|inf' stdout.log` | CRITICAL — stop training |
| **OOM** | `grep -i 'out of memory\|CUDA OOM' stderr.log` | CRITICAL — reduce memory |
| **NCCL timeout** | `grep -i 'nccl\|timeout\|watchdog' stderr.log` | CRITICAL — check network |
| **Loss spike** | Current loss > 5× recent average | WARNING — monitor closely |
| **Zero grad norm** | `grad norm` = 0.0 for multiple steps | WARNING — learning stopped |
| **Throughput drop** | Current TPS < 50% of initial | WARNING — check GPU util |

## Health Check Summary Format

```
Training Status: HEALTHY / WARNING / CRITICAL
Iteration: 1500 / 10000 (15.0%)
Loss: 2.34 (↓ trending down)
Grad Norm: 0.85
Throughput: 12,500 tokens/sec/GPU
GPU Memory: 65.2 / 80.0 GB (81.5%)
Elapsed: 2h 15m | ETA: 12h 45m
```

## Common Issues

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Log file empty | Crashed during startup | Check stderr for import/CUDA errors |
| Loss stuck | LR too low or gradient issue | Check LR schedule, verify grad norm > 0 |
| Loss oscillating | LR too high or batch too small | Reduce LR or increase batch size |
| `Killed` in stderr | OOM killed by system | Check `dmesg | tail`, reduce memory |
| Timestamps stop | Process hung | GPU util 0% → NCCL hang; >0% → data loader stuck |

## When to Load Full Skill

Load the full `train-monitor` skill when:
- Setting up continuous monitoring for long training runs
- Debugging complex multi-node anomalies
- Understanding detailed metric parsing patterns
- Configuring checkpoint and TensorBoard monitoring

This summary covers the essentials. For detailed parsing commands and multi-node monitoring, load the full skill.
