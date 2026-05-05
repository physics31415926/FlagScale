# Train-Run — Summary

Launch and manage FlagScale distributed training runs with proper preflight checks.

## Workflow

1. **Connect to Server** — Use `conda run --prefix <env_path> <command>` for non-interactive shells
2. **Check Environment** — Verify GPU availability (`nvidia-smi`), conda env, multi-node connectivity
3. **Preflight Check** — checks before launching:
   - 3a. Core dependencies (torch, megatron, transformer_engine)
   - 3b. GPU availability
   - 3c. Data path validation
   - 3d. Topology freshness (optional)
   - 3e. **Dry run — HARD GATE** (2-step run at target parallelism)
   - 3f. **Launch script validation** — read source code (argument parser, launcher, existing examples) to validate config
   - 3g. Config arithmetic (TP × PP × DP = world_size)
   - 3h. Checkpoint compatibility
   - 3i. Memory budget estimation
   - 3j. **Data pipeline standalone test — HARD GATE**
   - 3k. Checkpoint loading verification (step-0 loss check)
4. **Start/Stop Training** — `flagscale train` CLI or legacy `python pretrain_*.py`
5. **Monitor** — Log tailing, loss tracking, GPU utilization

## Key Commands

```bash
# Start training
flagscale train --config <config.yaml>
# Dry run (validate config)
flagscale train --config <config.yaml> --train-iters 2
# Stop training
flagscale stop --config <config.yaml>
```

## Critical Checks

- Step-0 loss should be ≈ ln(vocab_size) for random init, much lower for pretrained checkpoint
- If loss is ln(vocab_size) with checkpoint loading, the checkpoint was NOT loaded correctly
- Config arithmetic: `TP × PP × EP × CP` must divide `world_size` evenly
- Memory: model memory ≈ 2 × params_B / TP / PP bytes (bf16)

## Related Skills

- parallel-strategy: Choose TP/PP/DP/EP dimensions
- model-porter: Full porting workflow (training is Step 5)
- train-monitor: Detailed monitoring and debugging
