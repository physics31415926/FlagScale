# Parallel Strategy for FlagScale — Summary

## Overview

Guide for selecting and configuring parallelism strategies (TP/PP/DP/EP/CP/SP) in Megatron-LM-FL. Covers data pipeline handling, attention strategy selection, and memory estimation.

## Core Principle

FlagScale's value is parallelism-powered speedup. When porting a model, the goal is to USE the parallel infrastructure, not bypass it. If something doesn't work with TP/PP/EP, fix the integration — don't fall back to single-GPU or wrapper hacks.

## Parallelism Dimensions

| Dimension | Config Key | Splits | When to Use |
|-----------|-----------|--------|-------------|
| **TP** (Tensor Parallel) | `tensor_model_parallel_size` | Weight matrices across GPUs | Model too large for one GPU. Always try first. |
| **PP** (Pipeline Parallel) | `pipeline_model_parallel_size` | Layers across GPU groups | Model still OOM after max TP, or very deep models. |
| **DP** (Data Parallel) | Implicit: `world_size / (TP × PP × EP)` | Batch across GPU groups | Always present. More DP = higher throughput. |
| **EP** (Expert Parallel) | `expert_model_parallel_size` | MoE experts across GPUs | MoE models only. EP ≤ num_experts. |
| **CP** (Context Parallel) | `context_parallel_size` | Sequence length across GPUs | Very long sequences (>8K). Rarely needed. |
| **SP** (Sequence Parallel) | `sequence_parallel: true` | Activations along sequence dim | Always enable with TP. Reduces activation memory. |

**Constraint**: `TP × PP × EP × CP` must divide `world_size` evenly.

## Strategy Selection — Decision Tree

```
START: Estimate model memory
  │
  ├─ Fits on 1 GPU? → TP=1, PP=1, maximize DP
  │
  ├─ Fits with TP? → Set TP to minimum that fits (2, 4, or 8)
  │   └─ Enable sequence_parallel: true (always with TP>1)
  │
  ├─ Still OOM with TP=8? → Add PP
  │   └─ PP = ceil(model_layers / layers_per_stage)
  │
  ├─ MoE model? → Add EP
  │   └─ EP ≤ num_experts, EP should divide num_experts evenly
  │
  └─ Long sequences (>8K)? → Consider CP=2
```

## Quick Reference from Real Configs

| Model | Size | TP | PP | EP | Notes |
|-------|------|----|----|-----|-------|
| Qwen3 | 0.6B | 1 | 1 | - | Small model, DP only |
| Qwen3 | 32B | 8 | 1 | - | Full TP on 8 GPUs |
| Qwen3 | 235B-A22B (MoE) | 2 | 2 | 2 | TP+PP+EP for MoE |
| DeepSeek-V3 | 16B-A3B (MoE) | 1 | 2 | 4 | PP+EP, MLA attention |

## Data Pipeline Under Parallelism

This is where most porting failures happen. Megatron's data pipeline is tightly coupled to parallelism.

### Key Rules

1. **DP**: Each DP rank gets a different data shard. Dataset must be shardable — no global shuffling that differs across ranks.

2. **TP**: Only TP rank 0 loads data, then broadcasts to other TP ranks. Return the same tensor shapes on all TP ranks.

3. **PP**: Only the first pipeline stage needs input data. Other stages receive activations from the previous stage.

4. **EP**: Data pipeline is unaffected. EP only splits expert weights, not data.

5. **CP**: Sequence is split across CP ranks. Requires `reset_position_ids: true` and `reset_attention_mask: true`.

### Custom Dataset Integration Checklist

- [ ] Dataset `__len__` returns a consistent value across all ranks
- [ ] Dataset `__getitem__` returns tensors with shapes independent of rank
- [ ] If using packing: `reset_position_ids: true`, `reset_attention_mask: true`
- [ ] Test with `--train-iters 2` at target parallelism BEFORE long runs
- [ ] Watch for infinite loops: verify custom repeat/cycling terminates correctly

### Common Data Pipeline Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hang at first iteration | TP ranks have different data shapes | Ensure uniform shapes, check broadcast |
| NCCL timeout | PP stages waiting for data that never comes | Check `get_batch` returns None for non-first stages |
| Loss is NaN from step 1 | Attention mask wrong under packing | Enable `reset_position_ids` + `reset_attention_mask` |
| Infinite loop, no progress | Custom dataset repeat logic conflicts with Megatron sampler | Use Megatron's built-in data cycling |

## Attention Strategy

### Decision Order (NEVER skip to custom implementation)

```
Step 1: Use TransformerEngine-FL built-in attention
        ├─ Set transformer_impl: transformer_engine
        └─ Works for: standard MHA, GQA, RoPE

Step 2: If model has non-standard attention (MLA, sliding window)
        ├─ Check if Megatron-LM-FL already supports it
        └─ If supported: use the existing config flags

Step 3: Adapt existing attention — DO NOT write from scratch
        ├─ Subclass or modify the existing SelfAttention
        ├─ Keep TP-aware linear layers (ColumnParallelLinear, RowParallelLinear)
        └─ Test: verify output matches HF model at TP=1 first, then test TP>1

Step 4: ONLY if Step 3 is impossible
        └─ Write custom attention with full TP support
```

### Attention Variants in FlagScale

| Variant | Config Flag | Example Model |
|---------|------------|---------------|
| Standard MHA | (default) | LLaMA 2 |
| GQA | `num_query_groups: N` | Qwen3, LLaMA 3 |
| MLA (Multi-Latent) | `multi_latent_attention: true` + `kv_lora_rank`, `qk_head_dim`, `v_head_dim` | DeepSeek-V3 |
| QK LayerNorm | `qk_layernorm: true` | Qwen3 (large), DeepSeek-V3 |

## Memory Estimation

Quick formula for transformer models (bf16):

```
Per-GPU model memory ≈ (2 × params_B × 1e9 / TP / PP) bytes
Per-GPU optimizer memory ≈ (12 × params_B × 1e9 / TP / PP / DP) bytes
```

**OOM debugging order** (root cause first, not parallelism first):

1. Check if gradients are in fp32 unnecessarily (`accumulate_allreduce_grads_in_fp32`)
2. Check activation checkpointing (`recompute_granularity: selective` or `full`)
3. Check `use_distributed_optimizer: true` (shards optimizer states across DP)
4. THEN increase TP/PP if still OOM

## MoE-Specific Parallelism

MoE models add Expert Parallelism (EP) on top of TP/PP/DP.

### Key Config Fields

```yaml
system:
  expert_model_parallel_size: 4    # split experts across 4 GPUs
model:
  num_experts: 128                 # total experts
  moe_router_topk: 8              # experts activated per token
  moe_grouped_gemm: true          # fuse expert computation
  moe_token_dispatcher_type: "alltoall"
```

### EP Sizing

- `local_experts = num_experts / EP` — must be integer
- EP is orthogonal to TP: you can have TP=2, EP=4 on 8 GPUs (DP=1)
- Memory per GPU: only `local_experts` expert weights

## Verification Checklist

Before committing to a long training run:

```bash
# 1. Dry run: 2 iterations at target parallelism
python -m flagscale.train --config ... --train-iters 2

# 2. Check GPU memory is balanced
nvidia-smi  # during the 2-iteration run

# 3. Verify loss is finite and decreasing
grep "lm loss" <log_file> | head -5

# 4. Check throughput (tokens/sec/GPU) is reasonable
```

## Troubleshooting Quick Reference

| Problem | First Check | Second Check |
|---------|------------|--------------|
| OOM | `nvidia-smi` — which memory type? | Try `recompute_granularity: selective` before adding TP |
| NCCL timeout | Are all ranks reaching the same collective? | Check PP stage assignment, data pipeline |
| Loss NaN | Attention mask correct? | Gradient clipping enabled? (`clip_grad: 1.0`) |
| Throughput too low | TP communication overhead? | Try reducing TP, increasing DP |

## When to Load Full Skill

Load the full `parallel-strategy` skill when:
- Debugging data pipeline hangs or NCCL timeouts
- Implementing custom attention for non-standard architectures
- Porting MoE models with complex expert routing
- Optimizing memory usage for very large models
- Understanding detailed TP/PP/EP interaction rules

This summary covers the essentials. For detailed data pipeline integration, attention adaptation patterns, and MoE configuration, load the full skill.
