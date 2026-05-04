# Training Configuration for FlagScale — Summary

## Overview

Generate and manage FlagScale training configuration files. Two-level Hydra YAML structure:
- **Level 1 (Experiment)**: `train.yaml` — controls runner, environment, checkpoint paths
- **Level 2 (Task)**: `train/<size>.yaml` — contains system/model/data sections

## Two-Level YAML Structure

### Experiment Config (`train.yaml`)
Controls the experiment runner and environment:
- `experiment.exp_name`: Experiment name (must be meaningful, e.g., `qwen3_0_6b_train`)
- `experiment.exp_dir`: Output directory (NEVER use generic names like `xxx` or `test`)
- `experiment.load`: Checkpoint path to resume from (null for new training)
- `defaults.train`: References task config filename (e.g., `0_6b` → `train/0_6b.yaml`)
- `cmds.before_start`: Shell commands before training (typically activates conda env)
- `runner.hostfile`: Path to hostfile for multi-node (null for single-node)

### Task Config (`train/<size>.yaml`)
Three major sections:

```yaml
system:    # parallelism, precision, logging, checkpoint
model:     # architecture, training hyperparameters, optimizer
data:      # data path, tokenizer, data loading
```

**Key mapping**: YAML keys map to Megatron CLI args with underscores replacing hyphens:
- Megatron: `--tensor-model-parallel-size 4`
- FlagScale: `system.tensor_model_parallel_size: 4`

## Parallelism Strategy

| Dimension | YAML Key | What It Splits |
|-----------|----------|---------------|
| TP (Tensor Parallel) | `system.tensor_model_parallel_size` | Weight matrices across GPUs |
| PP (Pipeline Parallel) | `system.pipeline_model_parallel_size` | Layers across GPU groups |
| DP (Data Parallel) | Implicit: total_GPUs / (TP × PP × CP × EP) | Replicates model, splits data |
| EP (Expert Parallel) | `system.expert_model_parallel_size` | MoE experts across GPUs |
| CP (Context Parallel) | `system.context_parallel_size` | Sequence length across GPUs |
| VPP (Virtual Pipeline) | `system.num_layers_per_virtual_pipeline_stage` | Reduces PP bubble |

**General considerations** (not rigid rules):
- TP communication is intensive — works well with NVLink/NVSwitch
- PP introduces pipeline bubbles but enables larger models
- DP scales linearly and is simplest
- EP is MoE-specific, CP is for very long sequences
- VPP reduces bubbles when PP ≥ 4

**Topology-aware defaults**: Check memory for `topo_compute`, `topo_comm`, `topo_storage` keys (written by topo-detect). Use as context for parallelism decisions, not deterministic rules.

## Mixed Precision

| Mode | Config Keys | When to Use |
|------|------------|-------------|
| BF16 | `model.bf16: true` | Default for A100/H100/A800 (compute capability >= 8.0) |
| FP16 | `model.fp16: true` | Older GPUs (V100) without BF16 support |
| FP8 | `system.fp8: true` | H100/H800 only, requires TransformerEngine |

## TransformerEngine Integration

```yaml
model:
  transformer_impl: transformer_engine   # use TE (default, faster, supports FP8)
  # transformer_impl: local              # pure PyTorch (no TE dependency)
```

Use `local` when:
- TransformerEngine-FL not installed
- Debugging numerical issues
- Model architecture not supported by TE

## Common Pitfalls

1. **data_path with suffix**: `data_path: ./data/file.bin` is WRONG. Use `data_path: ./data/file` (no extension)
2. **global_batch_size not divisible**: Must be divisible by `DP × micro_batch_size`
3. **transformer_impl mismatch**: If `transformer_impl: transformer_engine` but TE not installed → crash
4. **vocab_size mismatch**: Training config vocab_size MUST match tokenizer's actual vocab
5. **Hydra caching**: Config changes don't take effect → `rm -rf ${experiment.exp_dir}/hydra/`
6. **Modifying wrong YAML**: Changes to `train.yaml` don't affect model/data params (those are in `train/<size>.yaml`)
7. **hostfile null vs missing**: For single-node, explicitly set `hostfile: null`

## Config Validation Checklist

Before EVERY training launch, verify:

### Arithmetic Constraints
```python
assert global_batch_size % (micro_batch_size * DP) == 0
assert num_attention_heads % TP == 0
assert num_key_value_heads % TP == 0  # for GQA
if PP > 1:
    assert num_layers % PP == 0
```

### Path Validation
- All paths exist: `data_path`, `tokenizer_path`, `load` (if resuming)
- No placeholders: `/path/to/`, `FIXME`, `TODO`
- For checkpoint resume: `latest_checkpointed_iteration.txt` exists

### Cross-Config Consistency
- `vocab_size` matches tokenizer's actual vocab size
- `seq_length` matches data preprocessing
- `ckpt_format` matches checkpoint's actual format
- Architecture params (`num_layers`, `hidden_size`, etc.) match model weights being loaded

## Quick Test vs Real Training

**Quick test** (environment validation):
- `train_iters`: 3-5
- `micro_batch_size`: 1
- `global_batch_size`: smallest valid (= DP × micro_batch_size)
- `eval_iters`: 0, `eval_interval`: 999999
- `save_interval`: 999999, `log_interval`: 1

**Real training**:
- Use values from model's reference config or paper
- Maximize `micro_batch_size` within GPU memory
- Enable checkpointing, evaluation, logging at appropriate intervals

## Multi-Node Configuration

### Hostfile Format
```
# Format: ip slots=<num_gpus>
10.0.0.1 slots=8
10.0.0.2 slots=8
```

### Pre-Launch Verification
1. Passwordless SSH between all nodes (both directions)
2. Firewall allows NCCL ports (default: 29500 + dynamic)
3. All nodes can resolve each other's hostnames
4. NCCL environment variables consistent across nodes

## When to Load Full Skill

Load the full `train-config` skill when:
- Writing a new training config from scratch
- Debugging config-related training failures
- Setting up complex parallelism strategies (VPP, EP, CP)
- Configuring FP8 training or TransformerEngine
- Setting up multi-node training
- Understanding detailed YAML-to-Megatron argument mappings

This summary covers the essentials. For detailed config templates, tokenizer type mappings, checkpoint resume structure, and troubleshooting, load the full skill.
