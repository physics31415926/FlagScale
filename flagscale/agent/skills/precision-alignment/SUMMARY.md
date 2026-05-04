# Precision Alignment — Summary

Align training numerics between source framework and FlagScale to ensure identical or near-identical training behavior.

## Three Scenarios

- **A: Model Migration** — Native (HF/custom) → FlagScale on NVIDIA. Baseline = source framework output.
- **B: Internal Iteration** — FlagScale code change on same hardware. Baseline = previous FlagScale version.
- **C: Hardware Migration** — FlagScale on NVIDIA → FlagScale on new hardware. Baseline = NVIDIA run.

## 8 Alignment Levels

| Level | What | Pass Criteria |
|-------|------|---------------|
| 1. Model Structure | Parameter count, layer mapping | Exact match of param count and shapes |
| 2. Hyperparameters | LR, optimizer, schedule, loss | Config diff is empty or justified |
| 3. Data Pipeline | Tokenization, batching, masking | Identical token sequences at step 0 |
| 4. Init + Checkpoint | Weight loading, init distribution | State dict key/shape/norm match |
| 5. Forward (1 step) | Loss, logits at step 0 | Relative error < 1e-5 (bf16) or 1e-7 (fp32) |
| 6. Backward (1 step) | Gradient norms, param updates | Gradient norm relative error < 1e-4 |
| 7. Multi-step | Loss curve over 100+ steps | Loss curves overlap, no divergence |
| 8. Convergence | Final metric (perplexity, accuracy) | Within acceptable tolerance of baseline |

## Core Principles

- Align against a REPRODUCED baseline, not reported numbers
- Understand BOTH sides before aligning — read source and target code
- DEBUG-first: add diagnostic prints BEFORE launching, not after failure
- Constraint elimination: change ONE variable at a time
- Experiment log isolation: each alignment attempt gets its own directory

## Alignment Modes

- **Strict**: bit-exact match (same hardware, same code path). For Scenario B.
- **Relaxed**: statistical match (different framework or hardware). For Scenarios A and C.

## Common Pitfalls

- bf16 vs fp32 accumulation differences
- Different attention implementations (FlashAttention versions)
- Tokenizer differences (BOS/EOS handling)
- Data pipeline ordering (shuffle seed, shard assignment)
- Fused vs unfused operations (fused LayerNorm, fused SwiGLU)

## Related Skills

- model-porter: Full porting workflow (alignment is Step 6)
- train-run: Training launch and monitoring
- parallel-strategy: Parallelism affects numeric behavior
