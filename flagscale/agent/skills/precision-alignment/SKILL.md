---
name: precision-alignment
description: Systematically align training precision across three scenarios — (A) model migration from native implementation to FlagScale on NVIDIA, (B) FlagScale internal iteration with self-regression validation, (C) hardware migration from FlagScale on NVIDIA to new hardware (DCU/TPU). Progressively eliminates divergence through 6 levels — model structure, hyperparameters, data pipeline, weight initialization, loss/evaluation, and forward/backward/optimizer pass. Includes cross-hardware methodology, determinism controls, and systematic divergence diagnosis.
keywords:
  - alignment
  - precision
  - accuracy
  - loss
  - divergence
  - 精度对齐
  - 对齐
  - loss对比
  - 前向对齐
  - 反向对齐
  - 复现
  - reproduce
  - cross-framework
  - cross-hardware
  - 框架对比
  - 训练对比
  - eval
  - benchmark
  - mmlu
  - initialization
  - 初始化
  - checkpoint
  - 权重转换
  - DCU
  - GPU
  - 算子
  - operator
  - flash attention
  - deterministic
  - 确定性
  - spike
  - gradient
parameters:
  - name: source_framework
    description: "The reference framework/hardware (ground truth side)"
  - name: target_framework
    description: "The framework/hardware being aligned"
  - name: model_name
    description: "Model name being aligned"
  - name: work_dir
    description: "Working directory for alignment artifacts (dumps, logs, reports). Should be on shared storage for multi-node scenarios."
requires: []
suggests: [train-run, train-monitor]
---

# Training Precision Alignment

Systematically verify and align training precision. This is a progressive, level-by-level process — each level eliminates one category of variables. Never skip levels.

## Three Alignment Scenarios

FlagScale precision alignment happens in three distinct scenarios, each with a different reference and different scope:

### Scenario A: Model Migration (Native → FlagScale on NVIDIA)

A model's native implementation (HuggingFace, DeepSpeed, etc.) runs on NVIDIA GPUs. FlagScale ports it and also runs on NVIDIA GPUs.

- **Reference**: Native implementation on NVIDIA
- **Target**: FlagScale on NVIDIA
- **Same hardware, different framework** → strict numerical alignment is possible
- **One-time**: Once FlagScale baseline aligns with native, the native implementation is done. All further work happens in Scenario B.
- **Levels 1-6 apply**: Full alignment from structure to forward/backward/optimizer

### Scenario B: FlagScale Internal Iteration (FlagScale on same hardware)

FlagScale baseline is already aligned (from Scenario A). Now adding parallelism, optimizations, TE-FL, FP8, etc.

- **Reference**: FlagScale's own aligned baseline (from Scenario A)
- **Target**: FlagScale with new changes
- **Same hardware, same framework, different config** → loss curve must not regress
- **Ongoing**: Every new feature or optimization goes through this
- **Primarily Level 5**: Compare loss curves. If loss regresses, use Level 6 to diagnose. Levels 1-4 are usually unchanged.

### Scenario C: Hardware Migration (FlagScale on NVIDIA → FlagScale on new hardware)

FlagScale already works on NVIDIA. Now porting to DCU, TPU, or other hardware.

- **Reference**: FlagScale on NVIDIA
- **Target**: FlagScale on new hardware
- **Same framework, different hardware** → strict numerical match may be impossible due to hardware floating-point differences
- **One-time per hardware**: Once aligned on the new hardware, further iteration on that hardware follows Scenario B
- **All levels apply**: But Level 6 focuses on operator-level differences (vendor kernels, compiler-induced divergence)

### Scenario Flow

```
Native impl (HF/DeepSpeed)
    │
    │  Scenario A: model migration (one-time)
    ▼
FlagScale @ NVIDIA (aligned baseline)
    │                    │
    │  Scenario B:       │  Scenario C: hardware migration (one-time per hardware)
    │  internal iteration│
    ▼                    ▼
FlagScale @ NVIDIA    FlagScale @ new hardware
(+TP/PP/TE-FL/FP8)      │
                         │  Scenario B: internal iteration on new hardware
                         ▼
                      FlagScale @ new hardware (+optimizations)
```

## Core Principles

### Align Against the Reproduced Baseline

Precision alignment MUST compare against a **reproduced baseline** — the reference implementation running on the same data, producing verified outputs. Never align against "expected" values from papers or documentation.

The alignment is fundamentally about **data flow and tensor-level equivalence**:

1. **Data flow alignment**: Trace the exact path of data through both systems. At each stage (tokenization → embedding → attention → FFN → loss), the tensors entering and leaving must match. If they diverge at stage N, all stages after N are meaningless to compare.

2. **Tensor-level verification**: Don't just compare final loss. Compare intermediate tensors — embeddings, attention outputs, MLP outputs, logits. A matching loss can hide compensating errors; matching intermediate tensors cannot.

3. **Reproduce first, then align**: Before any alignment work, run the reference implementation end-to-end and capture its outputs. This is your ground truth. If you can't reproduce the reference, you have nothing to align against.

The `reproduce` skill establishes this baseline. Use it before starting alignment.

### Understand Both Sides Before Aligning

Before writing any alignment code or running comparison experiments, you MUST understand:

**1. The native (baseline) implementation in depth:**
- Read the complete model forward pass — not fragments, the entire flow
- Understand the attention mechanism: what mask patterns, what kernel, what precision
- Understand the loss computation: what reduction, what normalization, any auxiliary losses
- Understand the data pipeline: how data is batched, padded, shuffled, distributed across ranks
- Run it and capture actual tensor values at key points — don't assume behavior from reading code alone

**2. What FlagScale/Megatron supports and what it doesn't:**
- Check Megatron-LM-FL's actual source code for supported attention types, activation functions, normalization layers
- Check TransformerEngine-FL's actual capabilities — what mask types it supports, what falls back to unfused paths
- Check what parallelism strategies are compatible with the model's architecture
- Identify the gaps: components the native implementation uses that FlagScale doesn't natively support

**3. The gap analysis determines your alignment strategy:**
- If FlagScale supports everything → alignment should be strict, differences are bugs
- If FlagScale lacks a component → you need an adaptation, and the adaptation's impact on precision must be quantified
- If FlagScale has an equivalent but different implementation (e.g., different FA version) → numerical differences are expected, define tolerance

Without this understanding, you'll waste cycles: "fixing" things that aren't bugs, missing things that are, or choosing approaches that the framework can't support.

### DEBUG-First Principle

Add diagnostic prints and tensor captures BEFORE launching training, not after failure. Each failed launch wastes minutes (model loading, data loading, NCCL init). Adding debug instrumentation proactively saves full launch cycles.

**Before every alignment training run:**
1. Add `print()` or `torch.save()` at key checkpoints in the forward pass (embedding output, attention output, loss input)
2. Add shape/dtype assertions at layer boundaries
3. Enable `TORCH_DISTRIBUTED_DEBUG=DETAIL` for distributed issues
4. For attention mechanisms: print `attn_mask_type`, mask shape, and a few mask values to confirm the mask is actually being applied

**Example proactive instrumentation:**
```python
# Add BEFORE launching, not after failure
# In the forward pass of the model being aligned:
if step < 3 and rank == 0:
    print(f"[DEBUG] step={step} embed_out: shape={h.shape}, sum={h.float().sum():.6f}, norm={h.float().norm():.6f}")
    print(f"[DEBUG] step={step} attn_out: shape={attn_out.shape}, sum={attn_out.float().sum():.6f}")
    print(f"[DEBUG] step={step} logits: shape={logits.shape}, sum={logits.float().sum():.6f}")
    print(f"[DEBUG] step={step} loss: {loss.item():.6f}")
```

This costs nothing (only runs for first 3 steps on rank 0) but immediately reveals where data flow diverges. Without it, a failed alignment requires: stop training → add prints → relaunch → wait for init → finally see the problem. That's 5-10 minutes wasted per cycle.

### Use Shared Storage for Multi-Node

All paths (data, checkpoints, logs, experiment outputs, captured tensors) must be on shared storage accessible from all nodes. Local paths like `/tmp/` or `./` will break in multi-node scenarios — node1 cannot see node0's local filesystem. If shared storage is not available, ask the user where to place artifacts before starting experiments.

### Leverage Existing Experiments

**Do NOT write custom alignment scripts from scratch.** Both the reference and target systems should already have runnable training code. Use them directly:

1. Run the reference experiment (native implementation or FlagScale baseline)
2. Run the target experiment (FlagScale new version or new hardware)
3. To capture intermediate values (activations, gradients, optimizer states), add print/hook statements in the existing training code — don't write standalone scripts

**Capturing tensor values:**
- Small tensors: `print(f"tensor_name: {tensor}")` or `torch.save(tensor, path)`
- Large tensors: use statistical summaries — `tensor.sum()`, `tensor.mean()`, `tensor.std()`, `tensor.norm()`, `tensor.min()`, `tensor.max()`
- Be careful with overflow: use `.float()` before `.sum()` if the tensor is bf16/fp16 and large
- Use hooks for intermediate layers: `module.register_forward_hook()` or `module.register_full_backward_hook()`

**Why this matters:** Writing custom scripts leads to:
- Reimplementing model logic (error-prone)
- Long debug cycles (5-10 min model loads per attempt)
- Mismatches between the custom script and actual training (different data pipeline, different initialization)

The existing training code is the ground truth. Instrument it, don't rewrite it.

### Constraint Elimination

Precision alignment is a constraint elimination problem. The final training accuracy is determined by: model structure × hyperparameters × data × weight initialization × numerical computation. If two systems produce different results, the cause MUST be in one of these categories. Isolate and verify each one in order, from cheapest to most expensive.

### Experiment Log Isolation

Every experiment MUST have isolated, self-contained logs. Without this, results are unreproducible and debugging is impossible.

```
{work_dir}/
├── experiments/
│   ├── exp001_structure_check/
│   │   ├── README.md              # what, why, config, conclusion
│   │   ├── ref_params.txt
│   │   └── tgt_params.txt
│   ├── exp002_hyperparam_align/
│   │   ├── README.md
│   │   ├── ref_config.yaml
│   │   └── tgt_config.yaml
│   ├── exp003_data_capture/
│   │   ├── README.md
│   │   ├── ref_rank0_step0.pt ... ref_rank7_step9.pt
│   │   └── tgt_rank0_step0.pt ... tgt_rank7_step9.pt
│   ├── exp004_small_deterministic/
│   │   ├── README.md
│   │   ├── ref_loss.csv
│   │   ├── tgt_loss.csv
│   │   └── error_report.txt
│   ├── exp005_fa_switch/          # controlled variable experiment
│   │   ├── README.md
│   │   └── ...
│   └── ...
├── checkpoints/                   # converted checkpoints
├── eval_results/                  # downstream evaluation outputs
└── summary.md                     # overall alignment status
```

Each experiment directory MUST contain a README.md with:
- Experiment ID and date
- Hypothesis being tested
- Exact configs/commands used (copy, not reference)
- Hardware and software environment
- Result and conclusion
- What changed vs the previous experiment (exactly ONE variable)

This applies to ALL phases: FlagScale migration verification, production training, reproduction, and cross-hardware alignment. Experiments without isolated logs are experiments that never happened.

## Alignment Modes

| Mode | Scenario | Verification Standard |
|------|----------|----------------------|
| **Strict** | A (model migration, same hardware) or B (internal iteration) | Numerical match: atol + rtol per step |
| **Relaxed** | C (hardware migration, deterministic: dropout=0) | Per-step tolerance + overall error bound |
| **Trend** | C (hardware migration, non-deterministic: dropout on) | Convergence trend + downstream task quality |

Choose the mode based on what can be controlled. Always start with strict/relaxed (deterministic) before moving to trend mode.

## Level Overview

```
Level 1: Model Structure Alignment       — are the models structurally identical?
Level 2: Hyperparameter Alignment         — are the training dynamics identical?
Level 3: Data Pipeline Alignment          — does each rank at each step see the same data?
Level 4: Weight Initialization Alignment  — do both start from identical parameters?
Level 5: Loss Curve & Evaluation          — do the outputs match?
Level 6: Forward/Backward Alignment       — where exactly does computation diverge?
```

---

## Level 1: Model Structure Alignment

### Objective

Verify identical architecture: same layers, same parameter shapes, same dtypes, same total parameter count.

### Method

Print `named_parameters()` from both systems and diff:

```python
for name, param in model.named_parameters():
    print(f"{name:<60} | Shape: {str(param.shape):<25} | Dtype: {param.dtype}")
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total}")
```

For FlagScale, add this in `model_provider()` of the relevant `train_*.py`.

### Parameter Fusion Mapping

Megatron fuses certain parameters. Element count must match after accounting for fusion:

**QKV Fusion**: `q_proj + k_proj + v_proj → linear_qkv`
- GQA: interleaved per KV group [Q_heads, K_head, V_head]
- MHA: per-head interleaved [q0,k0,v0, q1,k1,v1, ...]

**MLP gate+up Fusion (SwiGLU)**: `gate_proj + up_proj → linear_fc1`

Bias fusion follows the same pattern.

### Pass Criteria

- All parameters have a documented 1:1 mapping (build explicit table)
- Total element count identical
- No unexplained missing/extra keys

### Common Issues

- Padded vocab size (Megatron pads to multiple of 64/128)
- Tied weights (embedding/output)
- Bias presence flags (`add_qkv_bias`, `disable_bias_linear`)

---

## Level 2: Hyperparameter Alignment

### Objective

Unify ALL hyperparameters that affect training dynamics.

### Hyperparameter Checklist

```
Category              Parameter                    Reference       Target          Match?
──────────────────────────────────────────────────────────────────────────────────────────
Optimizer             type                         ____________    ____________    [ ]
                      beta1                        ____________    ____________    [ ]
                      beta2                        ____________    ____________    [ ]
                      epsilon                      ____________    ____________    [ ]
                      weight_decay                 ____________    ____________    [ ]
                      distributed optimizer        ____________    ____________    [ ]

Learning Rate         scheduler type               ____________    ____________    [ ]
                      peak lr                      ____________    ____________    [ ]
                      min lr                       ____________    ____________    [ ]
                      warmup steps                 ____________    ____________    [ ]
                      decay steps                  ____________    ____________    [ ]

Batch Size            micro_batch_size             ____________    ____________    [ ]
                      gradient_accumulation        ____________    ____________    [ ]
                      global_batch_size            ____________    ____________    [ ]
                      sequence_length              ____________    ____________    [ ]

Regularization        gradient clipping            ____________    ____________    [ ]
                      attention dropout            ____________    ____________    [ ]
                      hidden dropout               ____________    ____________    [ ]

Parallelism           DP                           ____________    ____________    [ ]
                      TP                           ____________    ____________    [ ]
                      PP                           ____________    ____________    [ ]
                      ZeRO stage                   ____________    ____________    [ ]

Precision             training dtype               ____________    ____________    [ ]
                      loss scaling                 ____________    ____________    [ ]
                      grad accumulation dtype      ____________    ____________    [ ]
                      softmax dtype                ____________    ____________    [ ]

Duration              total steps/samples          ____________    ____________    [ ]
                      seed                         ____________    ____________    [ ]

Implementation        Flash Attention              ____________    ____________    [ ]
                      Transformer Engine           ____________    ____________    [ ]
                      fused kernels                ____________    ____________    [ ]
```

### Critical Formula

```
global_batch_size = micro_batch_size × DP × gradient_accumulation_steps
```

Verify this equation holds for BOTH systems.

### Difference Classification

| Type | Impact | Action |
|------|--------|--------|
| **Mathematically equivalent** | None | Document only |
| **Numerically different** | Micro-level divergence, same convergence | Document, acceptable |
| **Semantically different** | Different training dynamics | Must fix |

Examples:
- Distributed optimizer (ZeRO-1) vs AllReduce → mathematically equivalent
- Flash Attention vs native attention → numerically different, acceptable
- epsilon 1e-8 vs 1e-18 → semantically different, must align
- bf16 with GradScaler vs bf16 without → document and verify

### Pass Criteria

- All "semantically different" items resolved
- Global batch size equation verified
- Training duration identical

---

## Level 3: Data Pipeline Alignment

### Objective

Ensure each DP rank at each step receives exactly the same input data (content + order) in both systems.

### Method

Insert data capture code in both training loops:

```python
if step < N:
    save_path = f"{work_dir}/experiments/exp_data/rank{rank}_step{step}.pt"
    torch.save(input_ids, save_path)
```

Compare:
```python
for step in range(N):
    for rank in range(world_size):
        ref = torch.load(f"ref_rank{rank}_step{step}.pt")
        tgt = torch.load(f"tgt_rank{rank}_step{step}.pt")
        assert torch.equal(ref, tgt), f"Mismatch at step {step}, rank {rank}"
```

### Common Divergence Sources

1. **Document boundary semantics**: continuous byte stream vs document collection
2. **Shuffle/randomness**: DistributedSampler stride vs shard-based distribution vs pseudo-random offset
3. **Multi-GPU distribution**: verify actual sample IDs per rank
4. **Tokenization**: verify tokenizer produces identical output

### Alignment vs Production

Data alignment tools (custom loaders, disabled shuffle) are verification-only. Document what to revert.

### Pass Criteria

- `torch.equal()` passes for ALL ranks across ALL captured steps

---

## Level 4: Weight Initialization Alignment

### Objective

Both systems start from numerically identical parameters (max_diff < 1e-6).

### Why Direct Seed Matching Fails

1. **RNG type**: Reference on CPU (CPU RNG), Megatron on GPU (CUDA RNG) → different sequences
2. **Seed offset**: Megatron applies `seed + 2718 + tp_rank` internally
3. **Consumption order**: Fused parameters consume RNG differently than separate ones

Due to point 3, identical random initialization from the same seed is generally impossible across frameworks.

### Method: Initialize → Convert → Load

```
Step 1: Initialize in reference framework → dump weights
Step 2: Convert to target format (handling fusion mappings)
Step 3: Load in target → dump weights
Step 4: Verify all parameter pairs match (max_diff < 1e-6)
```

### Pass Criteria

- ALL parameter pairs: max_diff < 1e-6
- Total parameter count matches

---

## Level 5: Loss Curve & Evaluation

### Objective

With Levels 1-4 aligned, compare actual training outputs.

### Quantitative Verification Standard

Define acceptance criteria BEFORE running experiments:

**Per-step tolerance** (first N steps):
```
pass_condition: |loss_target - loss_ref| <= atol + rtol × |loss_ref|
```

Recommended defaults:
- Strict mode (same hardware): atol=1e-4, rtol=1e-4
- Relaxed mode (cross-hardware): atol=1e-2, rtol=2e-3

**Overall error bound**:
```
avg_relative_error = mean(|loss_target - loss_ref| / |loss_ref|) over all steps
```
Recommended: < 2% for relaxed mode.

**Report format**:
```
Iter | Ref Loss  | Target Loss | Diff      | AbsErr    | RelErr (%)
-----+-----------+-------------+-----------+-----------+-----------
1    | 12.759    | 12.759      | 0.000030  | 0.000030  | 0.0002
...
Failing steps: [3, 5, 8]  (N out of 30)
Overall avg relative error: 0.0150 (1.50%)
```

### Verification Strategy Matrix

Run experiments in this order (cheapest first):

| Phase | Experiment | Controls | What it proves |
|-------|-----------|----------|---------------|
| 1 | Small-scale deterministic | Same init, same data, dropout=0, small cluster | Numerical computation matches |
| 2 | Large-scale deterministic | Same init, same data, dropout=0, full cluster | Scaling does not introduce errors |
| 3 | End-to-end with non-determinism | Production config (dropout on) | Convergence trend matches |
| 4 | Downstream task evaluation | Generate/classify from checkpoint | End-to-end quality matches |

### Checkpoint Warm-Start Test

For cases where cold-start is too short to be meaningful:

1. Load pretrained checkpoint in target system
2. Continue training for N steps
3. Verify: no loss spike, no grad norm explosion, normal convergence
4. Run downstream evaluation, compare with reference

Note: If optimizer states cannot transfer (different parallelism), only model weights are warm-started. First few steps may show transient loss increase.

### When Strict Alignment Is Impossible

When non-deterministic operations differ across platforms (dropout, NCCL reduce order):

1. **Visual convergence**: Same curve shape, similar descent rate, similar final value
2. **Statistical bounds**: Overall avg relative error within tolerance
3. **Downstream quality**: The definitive test — comparable output quality = aligned

### Divergence Pattern Diagnosis

| Pattern | Likely Cause | Action |
|---------|-------------|--------|
| Divergence from step 0 | Initialization mismatch | Re-verify Level 4 |
| Linear growth | Small numerical difference | Acceptable if within tolerance |
| Exponential growth | Bug in computation | Proceed to Level 6 |
| One curve flat, other descends | Broken gradient or wrong operator | Proceed to Level 6 |
| Sudden spike then recovery | Numerical instability in specific op | Check loss scaling, grad clipping |
| Periodic large errors | Data pipeline or LR schedule mismatch | Re-verify Levels 2-3 |
| First 2 steps match, step 3+ diverge | Likely attention or complex operator | Proceed to Level 6 |

---

## Level 6: Forward/Backward Alignment

### Objective

Locate the exact layer and operator where computation diverges.

### When to Use

- Loss curves diverge beyond tolerance
- One system fails to learn while the other converges
- Specific steps show anomalously large errors

### Method: Controlled Variable Experiments

Before hooking every layer (expensive), run targeted experiments. Each changes exactly ONE variable:

| # | Experiment | What it isolates |
|---|-----------|-----------------|
| 1 | Reference switches attention kernel (e.g., fused_attn → FA) | Attention implementation |
| 2 | Target loads reference-initialized weights | Init as cause |
| 3 | Target switches to high-precision variant of suspected op | Operator precision |
| 4 | Forward/backward hooks on suspected layers | Exact divergence point |

For cross-hardware: operator implementations are the primary suspect. Test by switching the reference to use the same kernel type as the target.

### Method: Layer-by-Layer Hook Comparison

**Add hooks to the existing training code** on both sides — don't write standalone forward scripts. Both systems already have working training loops; instrument them:

```python
# Add this to the existing training script (both reference and target)
hook_outputs = {}
def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # For large tensors, save statistics instead of full tensor
        t = output.detach().float()
        hook_outputs[name] = {
            "sum": t.sum().item(),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "max": t.abs().max().item(),
            "norm": t.norm().item(),
        }
    return hook_fn

for name, module in model.named_modules():
    if "layers." in name and name.count(".") == 1:  # top-level layers only
        module.register_forward_hook(make_hook(name))
```

Then compare the statistics from both runs. If a layer shows divergence, add more detailed hooks to that specific layer's submodules.

### Divergence Metrics

Use multiple metrics — a single metric can be misleading:

| Metric | What it catches | Blind spot |
|--------|----------------|------------|
| Max absolute diff | Outlier errors | Misses systematic small bias |
| Mean absolute diff | Average error level | Hides outliers |
| Relative error | Scale-independent comparison | Unstable near zero |
| Cosine similarity | Direction alignment | Misses magnitude differences |
| Gradient ratio | Backward pass bugs (2x = likely bug) | N/A |

### Divergence Localization

1. Find the first layer with significant divergence — all layers before should have near-zero diff
2. Examine that layer's operator — custom CUDA kernel? Vendor-specific implementation?
3. Compare gradients at the same layer — 2x difference or sign flip = bug, not noise

### Common Root Causes

| Root Cause | Symptom | Fix |
|-----------|---------|-----|
| Different FA implementations | Divergence at attention; gradient 2x diff | Use same FA version or high-precision variant |
| Custom CUDA kernel mismatch | Divergence at specific op | Ensure identical kernel source |
| Vendor operator precision | Forward diff small, gradient diff large | Use vendor's high-precision variant |
| fused_attn vs FA | Divergence in attention, not MLP | Standardize attention implementation |
| Loss scaling difference | Gradient magnitudes differ by constant | Align GradScaler settings |
| Softmax precision | Consistent small divergence in attention | Ensure fp32 softmax on both sides |
| NCCL/RCCL reduce order | Non-deterministic allreduce | Accept or use deterministic collectives |

### Backward Pass Alignment

If forward matches but training diverges:

```python
def make_backward_hook(name):
    def hook_fn(module, grad_input, grad_output):
        hook_outputs[f"{name}.grad_out"] = grad_output[0].detach().cpu()
        if grad_input[0] is not None:
            hook_outputs[f"{name}.grad_in"] = grad_input[0].detach().cpu()
    return hook_fn
```

Key signals:
- Gradient 2x difference → bug in backward kernel
- Gradient sign flip → definitely a bug
- Gradient sparsity mismatch → different masking behavior

### Optimizer State Alignment

If forward and backward match but training diverges over multiple steps, the issue may be in optimizer behavior. After 10+ steps, compare optimizer states:

```python
ref_state = ref_optimizer.state_dict()["state"]
tgt_state = tgt_optimizer.state_dict()["state"]

for param_id in ref_state:
    for key in ["exp_avg", "exp_avg_sq"]:  # Adam m/v
        if key in ref_state[param_id]:
            ref_val = ref_state[param_id][key]
            tgt_val = tgt_state[param_id][key]
            diff = (ref_val - tgt_val).abs().max().item()
            print(f"Param {param_id} {key}: max diff = {diff:.2e}")
```

Common causes of optimizer divergence:
- Different Adam epsilon values
- Different bias correction schedules
- Weight decay applied differently (AdamW vs Adam + L2)
- Gradient clipping threshold or method mismatch
- Loss scaling differences in mixed precision

### After Identifying Root Cause

1. Fix the specific operator (or switch implementation)
2. Re-run the SAME experiment that revealed the issue
3. Return to Level 5 and re-run full loss comparison

---

## Determinism Controls

To maximize reproducibility, apply these settings before alignment experiments:

### PyTorch Determinism

```python
import torch
import os

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Note: `torch.use_deterministic_algorithms(True)` will raise errors for operations without deterministic implementations (e.g., some scatter/gather ops). In that case, use `torch.use_deterministic_algorithms(True, warn_only=True)` and document which ops are non-deterministic.

### NCCL Determinism

AllReduce with floating-point is not bitwise deterministic by default (reduction tree order may vary). For strict alignment:
- Use same number of GPUs on both sides
- Use same NCCL version if possible
- For NVIDIA CCCL 3.1+: use deterministic reduction mode

### Dropout and Stochastic Ops

For deterministic comparison, disable ALL stochastic operations:
- `attention_dropout = 0.0`
- `hidden_dropout = 0.0`
- Any other dropout in the model

Only re-enable for trend-mode verification after deterministic alignment passes.

### Cross-Hardware Determinism Limits

Even with all controls, cross-hardware (GPU vs DCU) will have micro-level numerical differences due to:
- Different floating-point unit implementations
- Different operator fusion strategies
- Different memory access patterns affecting accumulation order
- Compiler-induced differences (NVCC vs HIP compiler)

These are inherent and not bugs. The question is whether they compound into meaningful divergence.

---

## Known Sources of Numerical Divergence

### Flash Attention Numerical Properties

Research shows Flash Attention has roughly an order of magnitude more numeric deviation compared to baseline attention at BF16 (forward pass). This is because FA recomputes attention in tiles, and the online softmax accumulation introduces different rounding patterns.

Implications for alignment:
- If reference uses fused_attn and target uses FA (or different FA version), attention outputs will differ
- The difference is typically small in forward but can amplify in backward
- Cross-vendor FA implementations (NVIDIA FA vs AMD/DCU FA) may have larger differences
- High-precision FA variants exist for some vendors — use them for alignment verification

### Compiler-Induced Differences

NVIDIA (NVCC/PTX) and AMD (HIP/ROCm) compilers may generate different instruction sequences for the same mathematical operation, leading to different floating-point rounding. This is a fundamental source of cross-hardware divergence that cannot be eliminated.

### AllReduce Non-Determinism

In distributed training, AllReduce operations may sum gradients in different orders across runs, producing different results due to floating-point non-associativity. This affects:
- Run-to-run reproducibility on the same hardware
- Cross-hardware comparison (different collective implementations)

### Loss Spike Diagnosis

If alignment experiments show unexpected loss spikes:

| Symptom | Diagnosis | Action |
|---------|----------|--------|
| Spike at step 0 only | Initialization issue | Check Level 4 |
| Spike at specific step, reproducible | Data issue at that step | Check Level 3 |
| Random spikes, not reproducible | Non-deterministic op | Enable determinism controls |
| Grad norm explosion before spike | Numerical instability | Check grad clipping, loss scaling |
| Spike only on target, not reference | Target-specific bug | Proceed to Level 6 |
| Spike on both sides at same step | Data or LR schedule artifact | Not an alignment issue |

---

## Verification Progression

Recommended order for a complete alignment project:

```
Phase 0: Setup
  │  Create {work_dir} with experiment directory structure
  │  Document hardware, software, environment on both sides
  │  Choose alignment mode (strict / relaxed / trend)
  │
Phase 1: Static Alignment (Levels 1-4)
  │  Each level = one experiment directory
  │  All must pass before any training runs
  │
Phase 2: Small-Scale Deterministic (Level 5, Phase 1)
  │  Small cluster (e.g., 8 GPUs), dropout=0, fixed seeds
  │  Compare per-step loss for 30-100 steps
  │  PASS → Phase 3
  │  FAIL → Level 6 (diagnose with controlled experiments)
  │
Phase 3: Large-Scale Deterministic (Level 5, Phase 2)
  │  Full cluster, dropout=0, fixed seeds
  │  Compare per-step loss
  │  PASS → Phase 4
  │  FAIL → Scaling-specific issue (NCCL, TP/PP, etc.)
  │
Phase 4: End-to-End (Level 5, Phase 3-4)
  │  Production config (dropout on)
  │  Compare convergence trend + downstream task quality
  │  PASS → DONE
  │  FAIL → Investigate non-deterministic ops
  │
Phase 5: Warm-Start Validation (optional but recommended)
  │  Load pretrained ckpt → continue training → evaluate
  │  Confirms the full pipeline works end-to-end
```

---

## Decision Flowchart

```
Level 1: Model Structure
  │  Print named_parameters() on both sides
  │  Build fusion mapping table
  │  Verify total element count matches
  │  PASS → Level 2
  │  FAIL → Fix model definition

Level 2: Hyperparameters
  │  Fill comparison checklist
  │  Classify differences (equivalent / numerical / semantic)
  │  Fix all semantic differences
  │  PASS → Level 3
  │  FAIL → Fix config, do NOT proceed

Level 3: Data Pipeline
  │  Capture input tensors for N steps on ALL ranks
  │  torch.equal() on each (step, rank) pair
  │  PASS → Level 4
  │  FAIL → Debug data loading (boundaries, shuffle, distribution)

Level 4: Weight Initialization
  │  Initialize in reference → convert → load in target
  │  Verify max_diff < 1e-6 for all parameter pairs
  │  PASS → Level 5
  │  FAIL → Fix conversion script

Level 5: Loss & Evaluation
  │  Phase 1: Small-scale deterministic comparison
  │  Phase 2: Large-scale deterministic comparison
  │  Phase 3: End-to-end trend comparison
  │  Phase 4: Downstream task evaluation
  │  PASS → DONE
  │  FAIL → Level 6

Level 6: Forward/Backward
  │  Run controlled variable experiments (cheapest first)
  │  Narrow to suspect operator
  │  Hook layers, compare outputs with fixed input
  │  Identify divergent operator
  │  Fix root cause (or switch implementation)
  │  Re-verify → Return to Level 5
```

---

## Rules

1. **Isolate every experiment.** Each experiment gets its own directory with README, configs, results, and conclusion. No isolation = no experiment.
2. **Never skip levels.** Level 6 is expensive. Levels 1-4 are cheap and eliminate most issues.
3. **Verify before proceeding.** Each level has explicit pass criteria. Do not assume — check.
4. **Do not modify reference code** to make alignment easier. The reference is ground truth.
5. **Minimize target modifications.** Only modify target code when a genuine bug is found. Document every change.
6. **Save all artifacts.** Parameter dumps, captured tensors, loss logs, evaluation results — everything to `{work_dir}`.
7. **One variable at a time.** When diagnosing, change only one thing per experiment.
8. **Classify differences.** Not every difference is a bug. Mathematically equivalent implementations produce micro-level numerical differences that do not affect convergence.
9. **Warm start is the definitive test.** Cold start on small models produces near-random results.
10. **Alignment tools are temporary.** Custom loaders, disabled shuffle, fixed seeds — document what to revert for production.
11. **Environment matters.** Document hardware, driver version, library versions, and environment variables on both sides.
12. **Cross-hardware has inherent limits.** Different hardware has different floating-point behavior. When strict match is impossible, convergence trend + downstream quality is the acceptance standard.
13. **Check vendor high-precision variants.** When cross-hardware divergence traces to a specific operator, check if the vendor provides a high-precision variant before concluding it's a bug.
14. **Use multiple divergence metrics.** No single metric (max diff, mean diff, cosine similarity) tells the full story. Always report at least max diff + mean diff + cosine similarity.
15. **Downstream task is the ultimate judge.** If per-step loss doesn't match but downstream quality (generation, classification, etc.) is comparable, the alignment is acceptable for production.

## References

- [Is Flash Attention Stable?](https://arxiv.org/abs/2405.02803) — FA has ~10x more numeric deviation than baseline attention at BF16
- [Finding Numerical Differences Between NVIDIA and AMD GPUs](https://arxiv.org/abs/2410.09172) — compiler-induced numerical differences across GPU vendors
- [Revealing Inconsistencies Across Heterogeneous AI Accelerators](https://arxiv.org/abs/2511.11601) — non-NVIDIA platforms show >5% operator output discrepancies
- [Joint Training on AMD and NVIDIA GPUs](https://arxiv.org/abs/2602.18007) — cross-vendor training achieving 98% throughput with preserved accuracy
- [NVIDIA Framework Reproducibility](https://github.com/NVIDIA/framework-reproducibility) — determinism controls for PyTorch/TensorFlow
- [Controlling Floating-Point Determinism in NVIDIA CCCL](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) — three determinism levels for parallel reductions
- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html) — official determinism settings
- [Stabilizing the Pre-training of LLMs](https://arxiv.org/abs/2312.16903) — loss spike analysis and gradient norm relationship

---

## Related Skills

- `reproduce` — establish verified baseline before alignment
- `model-porter` — port model architecture (alignment verifies the port)
- `train-run` — launch training runs for alignment comparison
- `train-monitor` — monitor training metrics during alignment experiments
