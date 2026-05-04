# Reproduce Training Results — Summary

## Overview

Reproduce training results from open-source implementations to establish a verified baseline before migrating to FlagScale. Covers the IMMUTABLE vs ADAPTABLE parameter framework, original artifact reuse, quick baseline validation, and experiment isolation.

## Why Reproduction Matters

Reproduction is the BASELINE for FlagScale migration. If the baseline is wrong, everything built on top is meaningless. Treat with highest rigor.

## Core Principle: IMMUTABLE vs ADAPTABLE

### IMMUTABLE Parameters (changing = different experiment)
- Model architecture (layers, hidden size, heads, FFN)
- Tokenizer and vocabulary
- Optimizer type and LR schedule
- Loss function
- Data processing pipeline
- Weight initialization, dropout, regularization

### ADAPTABLE Parameters (hardware mapping, preserves experiment)
- `num_nodes`, `num_gpus`
- `batch_size` + `gradient_accumulation_steps` (must maintain same effective batch)
- Data parallelism strategy (DP, DDP, FSDP)
- `num_workers`, logging/checkpoint intervals

**Rule**: If unsure → treat as immutable and ask user.

## Original Artifact Reuse

Use artifacts from the original release, NOT regenerated:
- Exact tokenizer from original model release
- Exact config file from original repo
- Official pretrained weights if needed
- Original data processing scripts

## Quick Baseline Validation (3 Steps)

### Step 1: Minimal Run (10-100 steps)
- Verify training starts and loss decreases
- Check loss magnitude matches expectations (random init ≈ ln(vocab_size))
- Verify gradient norms are reasonable (0.1-10.0 typical)

### Step 2: Short Run (100-1000 steps)
- Compare loss curve shape with published results
- Verify learning rate warmup behaves correctly
- Check memory usage is stable (no leaks)

### Step 3: Checkpoint Validation
- Save and reload checkpoint, verify training continues correctly
- Compare loss before/after reload — should be identical

## Per-Step Logging

Record at EVERY step: `step, loss, grad_norm, lr, throughput, timestamp`

Save to: `<output_dir>/metrics.jsonl` (one JSON object per line)

## Experiment Isolation

- Each reproduction gets its own directory
- Never mix reproduction outputs with FlagScale outputs
- Naming: `reproduce_<model>_<framework>_<date>/`
- Record in workspace_state Experiments table

## Verification Checklist

- [ ] All IMMUTABLE parameters match original exactly
- [ ] Effective batch size preserved (batch × accum × GPUs)
- [ ] Original tokenizer/vocab used (not regenerated)
- [ ] Per-step metrics logged
- [ ] Loss at step 100 within expected range
- [ ] Experiment directory isolated and recorded

## Related Skills

- `model-porter` — port model to FlagScale after baseline established
- `precision-alignment` — align FlagScale implementation against reproduction baseline
- `env-setup` — set up environment for original implementation
- `data-prep` — prepare data for reproduction

## When to Load Full Skill

Load the full `reproduce` skill when:
- Starting a new reproduction from scratch
- Debugging loss mismatches with published results
- Understanding framework-specific reproduction commands (HuggingFace, DeepSpeed, Fairseq)
- Setting up per-step logging for custom training loops

This summary covers the essentials. For detailed framework commands and troubleshooting, load the full skill.
