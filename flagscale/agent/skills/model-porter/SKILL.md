---
name: model-porter
description: Port models from papers, HuggingFace, or other frameworks to Megatron-LM-FL for distributed training on FlagScale. Analyzes model architecture, matches to Megatron components, generates checkpoint conversion code, and verifies forward alignment. For training config generation, use the train-config skill.
keywords:
  - model
  - port
  - porting
  - migrate
  - migration
  - convert
  - checkpoint
  - huggingface
  - megatron
  - architecture
  - alignment
  - 模型迁移
  - 模型转换
  - 权重转换
  - 模型适配
  - 检查点
parameters:
  - name: source
    description: "Source type: paper, huggingface, or code"
    default: huggingface
  - name: model_name
    description: "Name for the ported model (used in directory names)"
---

# Model Porter

Port models to Megatron-LM-FL / FlagScale for distributed pre-training.

**Scope**: All model types — decoder-only LLM, VL, robotics, etc.

**Outputs**:
1. Checkpoint conversion code (`tools/checkpoint/<model>/`)
2. Forward alignment verification script
3. Architecture analysis and porting path recommendation

## Important Notes

- **Environment isolation**: Create a dedicated conda environment for FlagScale porting to avoid conflicts with the source model's dependencies. Use the `env-setup` skill to determine the correct Python version and dependencies — it will auto-detect CUDA driver, resolve FlagScale/Megatron-LM-FL/TransformerEngine-FL version constraints, and install accordingly. This keeps the source model's environment intact for side-by-side alignment verification.
- **Shared storage for multi-node**: All paths (data, checkpoints, logs, experiment outputs) should be on shared storage accessible from all nodes. Avoid local paths like `/tmp/` or `./` unless you're certain the task is single-node only. If shared storage is not available, ask the user where to place artifacts before proceeding.
- **Quantized models (GPTQ, AWQ, GGUF)**: These are inference-optimized and NOT suitable for training. If the user provides a quantized checkpoint, inform them that the original full-precision (FP16/BF16) weights are needed for pre-training or fine-tuning.
- **Tokenizer handling**: Always copy or reference the tokenizer from the source model. Verify `vocab_size` in the training config matches the tokenizer's vocabulary size. For models with added special tokens, check `added_tokens.json`.
- **Megatron-LM-FL path**: Checkpoint conversion code imports from `megatron`. Ensure Megatron-LM-FL (not upstream Megatron-LM) is installed — it contains FL-specific modules like `megatron.plugin.platform`.
- **Auto-fetch FL dependencies**: When you need to analyze or compile Megatron-LM-FL or TransformerEngine-FL source code and it's not available locally, pull the latest automatically — don't ask the user. Use the repos from env-setup skill (github.com/flagos-ai/Megatron-LM-FL, github.com/flagos-ai/TransformerEngine-FL).
- **Model size selection**: When a model family has multiple size variants (e.g., 0.6B/1.8B/7B/70B) or multiple training configs (e.g., e6_d6_size256 vs e18_d18_size1024) and the user did not specify a size, list the available options with their parameter counts and ask the user to choose. Recommend the smallest variant for initial porting/verification, but let the user decide.

## Overview

FlagScale supports models in two modes:

**Mode A — Config-driven** (most LLMs): Architecture expressed entirely through YAML parameters. Uses `train_gpt.py` → `gpt_builder()` → `GPTModel`. No model code needed.

**Mode B — Custom entrypoint** (non-standard architectures): Requires a dedicated `train_*.py`. Needed for RWKV, VL models, robotics models, or any architecture with components not in Megatron-LM-FL.

The skill determines which mode applies, then generates all artifacts.

---

## Step 1: Source Model Analysis

Determine the input source type and extract architecture parameters.

### 1a. Paper Input

Extract from the paper:
- Basic: num_layers, hidden_size, num_attention_heads, num_kv_heads, ffn_hidden_size, vocab_size, max_seq_length
- Attention type: MHA / GQA / MLA
- FFN type: standard / SwiGLU / GeGLU
- Normalization: LayerNorm / RMSNorm (pre-norm or post-norm)
- Position encoding: learned / RoPE / ALiBi / YaRN
- Special: MoE config, MTP, sliding window, etc.

### 1b. HuggingFace Model Input

```bash
# Read config.json
cat <hf_model_path>/config.json

# List weight names and shapes
python3 -c "
import json, os
idx_file = os.path.join('<hf_model_path>', 'model.safetensors.index.json')
if not os.path.exists(idx_file):
    idx_file = os.path.join('<hf_model_path>', 'pytorch_model.bin.index.json')
if os.path.exists(idx_file):
    with open(idx_file) as f:
        idx = json.load(f)
    for name in sorted(idx.get('weight_map', {}).keys())[:50]:
        print(name)
else:
    print('No index file found, model may be single-file')
"

# Read modeling code to identify non-standard components
find <hf_model_path> -name "modeling_*.py" -exec head -200 {} \;
```

### 1c. Other Source Code Input

Analyze the model definition code:
- Find the model class (usually inherits `nn.Module` or `PreTrainedModel`)
- Extract layer structure: attention, FFN, norm implementations
- Map weight names from `state_dict()`

### Output of Step 1

Present a structured parameter table to the user for confirmation:

```
=== Model Architecture Summary ===
Model name: <name>
Source: <paper/huggingface/code>

Basic Parameters:
  num_layers:            <value>
  hidden_size:           <value>
  num_attention_heads:   <value>
  num_kv_heads:          <value>
  ffn_hidden_size:       <value>
  vocab_size:            <value>
  max_position_embeddings: <value>

Architecture Features:
  Attention:    <MHA/GQA/MLA>
  FFN:          <Standard/SwiGLU/GeGLU>
  Norm:         <LayerNorm/RMSNorm>
  Position:     <learned/RoPE/ALiBi>
  Bias:         <with bias / no bias>
  Tied embeddings: <yes/no>

Special Components:
  MoE:          <yes/no> (experts=X, topk=Y, shared=Z)
  MTP:          <yes/no> (layers=X)
  Sliding window: <yes/no> (size=X)
```

**Ask user to confirm before proceeding.**

---

**Note**: If the source model has runnable training code, run a baseline validation first using the `reproduce` skill before proceeding with porting.

---

## Step 2: Architecture Matching

Compare extracted parameters against Megatron-LM-FL supported components.

### Supported Components Checklist

| Component | Megatron Support | Config Flag |
|-----------|-----------------|-------------|
| MHA (Multi-Head Attention) | Yes | default |
| GQA (Grouped Query Attention) | Yes | `group_query_attention: true`, `num_query_groups: N` |
| MLA (Multi-Latent Attention) | Yes | `multi_latent_attention: true`, `kv_lora_rank`, `qk_head_dim`, `qk_pos_emb_head_dim`, `v_head_dim` |
| Standard FFN | Yes | default |
| SwiGLU | Yes | `swiglu: true` |
| GeGLU | Yes | `gated_linear_unit: true` with gelu activation |
| LayerNorm | Yes | `normalization: LayerNorm` |
| RMSNorm | Yes | `normalization: RMSNorm` |
| Learned position embedding | Yes | `position_embedding_type: learned_absolute` |
| RoPE | Yes | `position_embedding_type: rope`, `rotary_base` |
| YaRN (RoPE scaling) | Yes | `rope_scaling: true` |
| MoE (Mixture of Experts) | Yes | `num_experts`, `moe_router_topk`, `moe_ffn_hidden_size`, etc. |
| Shared experts in MoE | Yes | `moe_shared_expert_intermediate_size` |
| MTP (Multi-Token Prediction) | Yes | `mtp_num_layers` |
| Sliding window attention | Yes | `window_size` |
| QK LayerNorm | Yes | `qk_layernorm: true` |

### Decision Logic

```
IF all components are in the supported list above:
  → Mode A (config-driven)
  → entrypoint = flagscale/train/megatron/train_gpt.py
ELSE:
  → Mode B (custom entrypoint needed)
  → List unsupported components
  → Find closest existing custom entrypoint as reference
```

### Finding the Closest Baseline

Check existing examples to find the most similar model:

```bash
ls examples/
# For each candidate, compare architecture features
```

Existing models and their key features:

| Model | Attention | FFN | MoE | MLA | MTP | Entrypoint |
|-------|-----------|-----|-----|-----|-----|------------|
| qwen3 | GQA + QK-norm | SwiGLU | Optional | No | No | train_gpt.py |
| llama3 | GQA | SwiGLU | No | No | No | train_gpt.py |
| deepseek_v3 | MLA | SwiGLU | Yes (shared) | Yes | Yes | train_gpt.py |
| aquila | GQA | SwiGLU | No | No | No | train_gpt.py |
| mixtral | GQA | SwiGLU | Yes | No | No | train_gpt.py |
| qwq | GQA | SwiGLU | No | No | No | train_gpt.py |
| rwkv | Custom (linear) | Custom | No | No | No | train_rwkv.py |
| llava | GQA + vision | SwiGLU | No | No | No | train_llava.py |
| qwen2_5_vl | GQA + vision | SwiGLU | No | No | No | train_qwen2_5_vl.py |

Select the baseline with the most matching features. Present the comparison to the user.

---

## Step 3: Determine Porting Path

### Mode A: Config-Driven

For models where all components are supported:

1. Select baseline model from the table above
2. List parameter differences between source model and baseline
3. Confirm entrypoint is `train_gpt.py`
4. Proceed to Step 4 (checkpoint conversion)

### Mode B: Custom Entrypoint

For models with unsupported components:

1. Identify which components need custom implementation
2. Find the closest existing `train_*.py` as reference
3. List files that need to be created/modified:
   - `flagscale/train/megatron/train_<model>.py` — training entrypoint
   - Possibly new modules in `Megatron-LM-FL/megatron/core/` or `megatron/plugin/`
4. Generate code scaffolding with clear TODO markers
5. **Warn the user**: custom model code requires manual verification and testing

**Important**: For Mode B, the agent generates scaffolding and guidance but cannot guarantee correctness. The user must review and test the custom code.

### Mode B: Phased Migration Plan (Complex Models)

For models with significant architectural gaps (multiple unsupported components, custom attention patterns, multi-modal generation, etc.), a phased migration is required. The agent MUST present a complete plan upfront, organized by the phases below, before implementing any code.

**Communicate to the user**: FlagScale Agent has the capability to achieve full distributed training optimization for any architecture. The work is phased because training infrastructure is inherently complex — each phase builds on the previous one, and attempting everything at once increases risk without saving time.

#### Phase 1: Model Structure (Functional Correctness)

Goal: The model runs correctly on a single GPU with the same forward/backward behavior as the source.

Work items:
1. **Component inventory**: List every architectural component in the source model. For each, classify as:
   - `SUPPORTED`: exists in Megatron-LM-FL (use config flag)
   - `ADAPTABLE`: close to an existing component (inherit and modify)
   - `CUSTOM`: must be implemented from scratch
2. **Custom entrypoint**: Create `train_<model>.py` with model builder, forward step, data pipeline
3. **Custom layers/modules**: Implement CUSTOM components as Megatron `TransformerLayer` submodules or standalone modules
4. **Checkpoint conversion**: `tools/checkpoint/<model>/` (args.py, ckpt.py, convert.py)
5. **Forward alignment**: Verify logits match source model on identical input (Step 5)
6. **Loss alignment**: Ensure all loss terms (lm_loss + auxiliary losses) match source (Step 5.3)
7. **Single-GPU training**: Run a few iterations with DP=1, TP=1, PP=1, verify loss curve matches source

Deliverables: Working single-GPU training with verified forward/loss alignment.

#### Phase 2: Basic Parallelism (Scale-Out)

Goal: The model trains correctly across multiple GPUs with basic data parallelism and tensor parallelism.

Work items:
1. **Data Parallelism (DP)**: Usually works out of the box with Megatron. Verify gradient synchronization correctness.
2. **Tensor Parallelism (TP)**: For each custom layer, implement `sharded_state_dict()` and verify the layer produces identical output when split across GPUs.
   - Standard attention/MLP: Megatron handles TP automatically via `ColumnParallelLinear` / `RowParallelLinear`
   - Custom components (e.g., MoT dual-path, special routing): Must explicitly define how weights are partitioned
3. **Checkpoint save/load with parallelism**: Verify checkpoint written with TP=N can be loaded correctly
4. **Multi-GPU forward alignment**: Compare outputs between TP=1 and TP=N on same input

Deliverables: Training runs correctly with DP + TP. Loss curve matches single-GPU baseline.

#### Phase 3: Advanced Parallelism (Full Distributed)

Goal: Support all parallelism dimensions needed for the target scale.

Work items (add as needed based on model size):
1. **Pipeline Parallelism (PP)**: Define layer partitioning strategy. For heterogeneous layers (e.g., vision encoder + LLM + VAE), define stage boundaries.
2. **Expert Parallelism (EP)**: For MoE/MoT models, distribute experts across GPUs
3. **Context Parallelism (CP)**: For long-sequence models, partition along the sequence dimension
4. **Virtual Pipeline Parallelism (VPP)**: Interleave pipeline stages for better efficiency
5. **Sequence Parallelism (SP)**: Partition LayerNorm and dropout along the sequence dimension
6. **Activation checkpointing**: Configure which layers to recompute vs. store

Deliverables: Full parallel training at target scale. Throughput and memory within expected range.

#### Phase 4: Performance Optimization

Goal: Maximize training throughput (MFU) and minimize memory usage.

Work items:
1. **Communication overlap**: Overlap gradient all-reduce with backward computation
2. **Flash Attention integration**: Ensure custom attention patterns work with FlashAttention
3. **TransformerEngine / FP8**: Enable mixed-precision training for supported components
4. **Kernel fusion**: Custom CUDA kernels for performance-critical operations
5. **Memory optimization**: Gradient accumulation, selective activation recomputation, CPU offloading
6. **Profiling**: Use Nsight Systems / PyTorch Profiler to identify bottlenecks
7. **Throughput benchmarking**: Compare MFU against theoretical peak

Deliverables: Optimized training at target throughput. MFU report.

#### Plan Output Format

When presenting the phased plan, output a structured table:

```
=== Phased Migration Plan: <model_name> ===

Closest baseline: <existing FlagScale model>
Estimated total phases: 4

Phase 1: Model Structure
  ┌─────────────────────────────────────────────────────────┐
  │ Component              │ Status    │ Work Required       │
  ├────────────────────────┼───────────┼─────────────────────┤
  │ LLM backbone (Qwen2)   │ SUPPORTED │ Config only          │
  │ GQA Attention           │ SUPPORTED │ Config only          │
  │ SwiGLU MLP              │ SUPPORTED │ Config only          │
  │ MoT dual-path layer     │ CUSTOM    │ New TransformerLayer │
  │ SigLIP ViT              │ ADAPTABLE │ Modify existing ViT  │
  │ Flow matching / VAE     │ CUSTOM    │ New module           │
  │ NaViT packed sequence   │ CUSTOM    │ Custom attention mask│
  └─────────────────────────────────────────────────────────┘

  Files to create/modify:
    - flagscale/train/megatron/train_bagel.py
    - megatron/core/models/bagel/mot_layer.py
    - megatron/core/models/bagel/siglip_vit.py
    - megatron/core/models/bagel/flow_matching.py
    - tools/checkpoint/bagel/{args,ckpt,convert}.py

Phase 2: Basic Parallelism
  - DP: standard (no custom work)
  - TP: MoT layer needs custom sharding logic
  - Estimated: 1 week after Phase 1

Phase 3: Advanced Parallelism
  - PP: Define stage boundaries (ViT | LLM layers 0-N | LLM layers N-M + VAE)
  - EP: If MoE layers present
  - Estimated: 1-2 weeks after Phase 2

Phase 4: Performance Optimization
  - FlashAttention for NaViT packed sequences
  - FP8 for eligible linear layers
  - Estimated: 1-2 weeks after Phase 3

FlagScale Agent can execute each phase incrementally.
Each phase produces a working, testable checkpoint before proceeding.
```

This plan gives the user full visibility into the work ahead. The agent should proceed to implement Phase 1 immediately (or the phase the user requests), and revisit the plan as each phase completes.

**Note**: Use the `train-config` skill to generate FlagScale training configuration files (experiment YAML + task YAML). The HF config.json to Megatron YAML parameter mapping table is available in that skill.

---

## Step 4: Generate Checkpoint Conversion Code

Generate `tools/checkpoint/<model>/` with 3 files.

### 4.1 args.py

Two functions:

**`load_args_hf2mg(args)`**: Read HF config.json, set Megatron args.

```python
import json, os

def load_args_hf2mg(args):
    hf_args_path = os.path.join(args.load, "config.json")
    with open(hf_args_path) as f:
        hf_args = json.load(f)

    # Map HF config.json fields to Megatron args
    args.hidden_size = hf_args["hidden_size"]
    args.ffn_hidden_size = hf_args["intermediate_size"]
    args.num_layers = hf_args["num_hidden_layers"]
    args.num_attention_heads = hf_args["num_attention_heads"]
    args.num_query_groups = hf_args["num_key_value_heads"]
    # ... (all parameters from the model's config.json)

    # Fixed Megatron defaults
    args.seq_length = 2048
    args.global_batch_size = 1024
    args.iteration = 1
    args.add_position_embedding = False
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.make_vocab_size_divisible_by = 64

    return args
```

**`save_args_mg2hf(args)`**: Create HF config from Megatron args.

```python
def save_args_mg2hf(args):
    # Import or create the HF config class
    # Map Megatron args back to HF config
    # Save config to args.save directory
    pass
```

### 4.2 ckpt.py

Weight mapping between HF and Megatron. **Prefer inheriting from existing models**.

Inheritance decision tree:
```
IF model uses MLA + MoE (like DeepSeek-V3):
  -> inherit from deepseek_v3/ckpt.py or write custom
IF model uses GQA + SwiGLU + no MoE (like LLaMA/Qwen):
  -> inherit from mistral/ckpt.py (attention) + override MLP if needed
IF model uses MoE + GQA (like Mixtral/Qwen3-MoE):
  -> inherit from mixtral/ckpt.py
ELSE:
  -> write from scratch
```

Required functions for HF -> Megatron direction:

```python
def get_hf_attn_ckpt(message, model, layer_id, args):
    """Extract attention weights from HF model into message dict."""
    tf_layer = model.model.layers[layer_id]
    # Standard GQA pattern:
    message["qkv weight"] = ...  # or separate q/k/v
    message["o weight"] = tf_layer.self_attn.o_proj.weight.data
    message["input norm weight"] = tf_layer.input_layernorm.weight.data
    message["post norm weight"] = tf_layer.post_attention_layernorm.weight.data

def get_hf_mlp_ckpt(message, model, layer_id, args):
    """Extract MLP weights from HF model into message dict."""
    tf_layer = model.model.layers[layer_id]
    # SwiGLU pattern:
    message["mlp l0 weight W"] = tf_layer.mlp.gate_proj.weight.data
    message["mlp l0 weight V"] = tf_layer.mlp.up_proj.weight.data
    message["mlp l1 weight"] = tf_layer.mlp.down_proj.weight.data
```

Required functions for Megatron -> HF direction:

```python
def set_hf_attn_ckpt(message, model, layer_id, args):
    """Write attention weights from message dict into HF model."""
    tf_layer = model.model.layers[layer_id]
    tf_layer.self_attn.o_proj.weight.data.copy_(message["o weight"])
    # Handle QKV: may need to split from fused format
    # ...

def set_hf_mlp_ckpt(message, model, layer_id, args):
    """Write MLP weights from message dict into HF model."""
    tf_layer = model.model.layers[layer_id]
    tf_layer.mlp.gate_proj.weight.data.copy_(message["mlp l0 weight W"])
    tf_layer.mlp.up_proj.weight.data.copy_(message["mlp l0 weight V"])
    tf_layer.mlp.down_proj.weight.data.copy_(message["mlp l1 weight"])
```

### QKV Weight Handling

Megatron fuses Q, K, V into a single tensor. The fusion format depends on attention type:

**GQA (num_kv_heads < num_attention_heads)**:
```python
# Interleaved format: [q_head_0, k_group_0, v_group_0, q_head_1, ...]
# Each group: group_size Q heads + 1 K head + 1 V head
group_size = num_attention_heads // num_kv_heads
qkv = torch.cat([
    q.view(num_kv_heads, group_size, head_dim, hidden_size),
    k.view(num_kv_heads, 1, head_dim, hidden_size),
    v.view(num_kv_heads, 1, head_dim, hidden_size),
], dim=1).reshape(-1, hidden_size)
```

**MHA (num_kv_heads == num_attention_heads)**:
```python
# Per-head interleaved: [q_0, k_0, v_0, q_1, k_1, v_1, ...]
qkv = torch.stack([q_heads, k_heads, v_heads], dim=1).reshape(-1, hidden_size)
```

### 4.3 Embedding and Output Layer

```python
def get_hf_embedding_ckpt(message, model, args):
    message["word embeddings"] = model.model.embed_tokens.weight.data

def get_hf_output_layer_ckpt(message, model, args):
    if hasattr(model, "lm_head"):
        message["output layer weight"] = model.lm_head.weight.data
    # Some models tie embeddings and output layer

def get_hf_final_norm_ckpt(message, model, args):
    message["weight"] = model.model.norm.weight.data
```

### 4.4 Full Conversion Script

```python
# tools/checkpoint/<model>/convert.py
# Typically inherits from the closest existing model's converter
# Override only the functions that differ
```

### 4.5 Conversion Commands

```bash
# HF -> Megatron (for loading pre-trained weights)
cd tools/checkpoint
python convert.py \
  --model-type <model> \
  --loader transformers --saver mcore \
  --load-dir <hf_model_path> \
  --save-dir <megatron_ckpt_path> \
  --target-tensor-parallel-size <tp> \
  --target-pipeline-parallel-size <pp>

# Megatron -> HF (after training, for evaluation/release)
python convert.py \
  --model-type <model> \
  --loader mcore --saver transformers \
  --load-dir <megatron_ckpt_path> \
  --save-dir <hf_output_path>
```

---

## Step 5: Get Training Running (Minimal Viable Training)

**Goal**: Get the model running in FlagScale with distributed training and producing loss values. The whole point of migrating to FlagScale is distributed training, so the baseline should already include parallelism. "Minimal" means small batch, short sequence, few steps — NOT single GPU.

**Flash attention strategy**: Megatron-LM-FL supports multiple flash attention backends. In the baseline, use the native backend for correctness and ease of debugging. TransformerEngine-FL is the ultimate target for peak performance, but it's introduced later in the iterative refinement (Step 6.4) after baseline precision is confirmed.

### 5.1 Baseline Feature Parity

The first training run must achieve **feature parity** with the source implementation, plus basic parallelism. Analyze the source training code/config to identify which features are enabled, then mirror them in FlagScale.

**Principle: "If the source has it, FlagScale must have it."** Plus parallelism — because that's why we're here.

Common features to check for parity:
- Mixed precision (BF16/FP16/AMP)
- Flash attention or other fused attention kernels
- Gradient checkpointing / activation recomputation
- Fused kernels (bias-geglu, fused-rmsnorm, etc.)

Parallelism baseline: use as many GPUs as currently available. If the full node is free, use the full node; if some GPUs are occupied by other tasks, use the remaining ones. Use topo-detect skill to check hardware topology and GPU availability, then choose TP/PP accordingly.

### 5.2 Minimal Configuration

Create a minimal-scale config with baseline features and initial parallelism:

```yaml
system:
  # Use available GPUs: full node if free, or 2/4/8 GPUs if others are occupied
  # Check with topo-detect skill, then set TP/PP accordingly
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  micro_batch_size: 1
  global_batch_size: 4

model:
  # Precision: match source (bf16/fp16/fp32)
  bf16: true
  # Flash attention: enable if source uses it (native backend for baseline)
  # use_flash_attn: true
  # Gradient checkpointing: match source granularity (selective/full/none)
  # recompute_granularity: selective

data:
  seq_length: 128
  train_samples: 100
```

### 5.3 First Training Run

```bash
# Use train-run skill to launch
python flagscale/train/megatron/train_<model>.py \
  --config <minimal_config.yaml> \
  --experiment-dir <shared_storage>/<model>_minimal
```

**Expected outcome**: Training completes 10 steps and prints loss values. Since baseline features (mixed precision, flash attention, gradient checkpointing) are enabled, the loss should be directly comparable to the source implementation at the same scale.

**Common issues at this stage:**
- Config parameter mismatch (vocab_size, hidden_size, etc.)
- Data pipeline errors (tokenizer mismatch, wrong data format)
- Shape mismatches in forward pass (usually caught by PyTorch)
- OOM even with minimal config (reduce sequence length further)
- Flash attention version mismatch (check FA2 vs FA1)
- TransformerEngine version incompatibility

### 5.4 Sanity Checks

Before proceeding to precision alignment, verify:
1. Loss is finite (not NaN or Inf)
2. Loss decreases over 10-20 steps (even if slowly)
3. All ranks are active and training (check multi-GPU utilization)
4. No Python exceptions or CUDA errors
5. Checkpoint saving works
6. Mixed precision is actually active (check log for "bf16" or "fp16" enabled)
7. Flash attention is actually being used (check log for flash attention kernel)

If any of these fail, fix them before moving to Step 6.

---

## Step 6: Precision Alignment

This step corresponds to **Scenario A** in the precision-alignment skill: aligning FlagScale against the native implementation, both on NVIDIA hardware.

Precision alignment has two phases:

1. **Align with native implementation** (6.1–6.3): Compare FlagScale baseline against the original native implementation on the same NVIDIA hardware. This is done ONCE. Once the baseline loss curve, forward output, backward gradients, and optimizer states match the source within acceptable tolerance, the native implementation is no longer needed.

2. **Self-regression within FlagScale** (6.4): This is **Scenario B** — all subsequent changes (more parallelism, TE-FL, FP8, scale-up) are validated against the FlagScale baseline itself. The goal shifts from "match the native" to "don't regress from our own aligned baseline."

For hardware migration (Scenario C), use the precision-alignment skill directly.

### 6.1 Alignment Method

**Do NOT write custom alignment scripts.** Both the native implementation and FlagScale should already be runnable from Step 5. Use them directly:

1. Run the native training for N steps, capture loss values
2. Run FlagScale training for N steps with the same data/config, capture loss values
3. Compare

When you need intermediate values (activations, gradients, optimizer states), instrument the existing training code with print statements or hooks — don't build standalone forward scripts. The training code IS the ground truth.

**Capturing values from existing training code:**
- Loss: already printed in training logs
- Logits/hidden states: add `register_forward_hook()` to the model in the training script
- Gradients: add `register_full_backward_hook()` or print `param.grad` after `loss.backward()`
- Optimizer states: print from `optimizer.state_dict()` after N steps
- Large tensors: use `.float().sum()`, `.norm()`, `.mean()`, `.max()` to avoid overflow in bf16

### 6.2 Alignment Hierarchy (Coarse to Fine)

Compare in this order, from coarsest to finest. Stop as soon as alignment passes at a given level:

**Level 1: Loss Curve** — Run both for 100-1000 steps, compare loss curves. Acceptable: ±1-5% after 100 steps.

**Level 2: Single-Step Loss** — Run one step, compare loss value. Acceptable: <1e-3 (fp32), <1e-2 (bf16).

**Level 3: Logits** — Add a hook to print/save final logits, compare. Acceptable: <1e-4 (fp32), <1e-2 (bf16).

**Level 4: Layer-by-Layer** — Add hooks to each transformer layer, compare hidden states to find where divergence starts.

**Level 5: Weights** — Compare loaded weights between the two systems to check checkpoint conversion correctness.

**Level 6: Operator** — If weights match but outputs don't, the issue is in operator implementation (RoPE, attention mask, norm epsilon, activation function variant).

### 6.3 Backward and Optimizer Alignment

If forward aligns but training diverges over steps:

**Backward**: Add hooks or print gradients for key parameters after `loss.backward()`. Look for 2x differences (bug in backward kernel) or sign flips.

**Optimizer**: After 10+ steps, compare optimizer states (Adam m/v). Common causes: different epsilon, different bias correction, AdamW vs Adam+L2.

### 6.4 Iterative Refinement (Scenario B: Self-Regression)

Once baseline precision is aligned with the native implementation (6.1–6.3), all further work happens within FlagScale. Each change is validated against the FlagScale baseline — the native implementation is no longer needed.

**Iteration 1: Scale to Multi-Node / Larger Parallelism**
- Expand from single-node to multi-node, or increase TP/PP for larger models
- Compare loss curve against FlagScale baseline (same data, same config except parallelism)
- TP/PP/DP should NOT change loss (mathematical equivalence)
- If loss changes, check sharded_state_dict(), pipeline schedule, or NCCL communication

**Iteration 2: Add Sequence Parallelism / Context Parallelism**
- Enable SP or CP
- Compare against FlagScale baseline
- Should NOT change loss

**Iteration 3: Add Expert Parallelism (MoE models only)**
- Enable EP
- Compare against FlagScale baseline
- Should NOT change loss

**Iteration 4: Switch Flash Attention to TransformerEngine-FL**
- The baseline uses Megatron-LM-FL's native flash attention backend
- For peak performance, switch to TransformerEngine-FL (`transformer_impl: transformer_engine`)
- TE-FL provides fused attention kernels, FP8 support, and other hardware-level optimizations
- Compare against FlagScale baseline — small numerical differences are expected (<1e-3)
- If TE-FL doesn't yet support a specific operator for this model, contribute the implementation or fall back to native backend for that operator

**Iteration 5: Add FP8 (if needed)**
- Enable FP8 for eligible layers via TransformerEngine-FL
- Compare against FlagScale baseline
- FP8 will introduce larger numerical differences (1e-2 to 1e-1)
- Verify loss still converges to similar final value

**Iteration 6: Scale Up**
- Increase batch size, sequence length, number of GPUs
- Compare against FlagScale baseline
- Verify throughput and memory usage are reasonable

At each iteration, if alignment breaks, roll back the change and debug before proceeding.

### 6.5 Multi-Loss Alignment

Many source models use auxiliary losses beyond `lm_loss`. When forward alignment on `lm_loss` alone is insufficient — or when the original training recipe relies on auxiliary losses for convergence — you may need to add those losses to FlagScale/Megatron-LM-FL.

**Step 1: Identify all losses in the source model**

Analyze the source training code to catalog every loss term:

```bash
# Search for loss computation in source code
grep -rn "loss" <source_model_path>/train*.py <source_model_path>/model*.py | grep -v "test\|#\|log"
# Look for loss aggregation (total_loss = lm_loss + alpha * aux_loss + ...)
grep -rn "total_loss\|loss =" <source_model_path>/ --include="*.py" | head -20
```

**Common auxiliary losses by model type:**

| Model Type | Auxiliary Losses | Purpose |
|-----------|-----------------|---------|
| MoE (DeepSeek, Mixtral, Qwen3-MoE) | Load balancing loss, z-loss | Expert utilization balance |
| MTP (DeepSeek-V3) | MTP loss (per speculative head) | Multi-token prediction alignment |
| VLM (QwenVL, LLaVA) | Vision-language contrastive loss | Cross-modal alignment |
| VLA (robotics) | Action loss (MSE/L1 on predicted actions) | Action prediction accuracy |
| RLHF/DPO models | Reward/preference loss | Alignment objective |
| Models with regularization | KL divergence, entropy bonus | Prevent distribution collapse |

**Step 2: Check what Megatron-LM-FL already supports**

Megatron-LM-FL has built-in support for several auxiliary losses:

| Loss | Config Parameter | Location |
|------|-----------------|----------|
| MoE load balancing | `moe_aux_loss_coeff` | `megatron/core/transformer/moe/moe_utils.py` |
| MoE z-loss | `moe_z_loss_coeff` | `megatron/core/transformer/moe/moe_utils.py` |
| MTP loss | `mtp_loss_scaling_factor` | `megatron/core/models/gpt/gpt_model.py` |

If the source model's auxiliary loss is already supported, just enable the corresponding config parameter with the coefficient from the source training recipe.

**Step 3: Implement missing losses**

If the source model has losses NOT supported in Megatron-LM-FL, implement them. There are two integration points:

**Option A: In `loss_func` (train_*.py) — for losses computed from model output**

Modify the `loss_func` in the model's training script (e.g., `pretrain_gpt.py` or `train_<model>.py`):

```python
def loss_func(loss_mask, output_tensor, model=None):
    # Standard lm_loss
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    lm_loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    # Add custom auxiliary loss
    aux_loss = compute_custom_aux_loss(model)  # your implementation
    aux_coeff = 0.1  # from source training recipe
    total_loss = lm_loss + aux_coeff * aux_loss

    report = {
        'lm loss': torch.cat([lm_loss.clone().detach().view(1), num_tokens.view(1)]),
        'aux loss': torch.cat([aux_loss.clone().detach().view(1), num_tokens.view(1)]),
    }
    return total_loss, num_tokens, report
```

**Option B: In model forward (GPTModel subclass) — for losses computed from intermediate activations**

When the loss requires access to intermediate states (e.g., hidden states from specific layers):

```python
class CustomModel(GPTModel):
    def post_process(self, hidden_states, labels, loss_mask=None):
        # Compute standard lm_loss via parent
        lm_loss = self.compute_language_model_loss(labels, logits)

        # Compute custom loss from hidden states
        custom_loss = self.compute_custom_loss(hidden_states)

        return lm_loss + custom_loss_coeff * custom_loss
```

**Step 4: Verify multi-loss alignment**

Run the same input through both implementations and compare each loss term individually:

```
=== Multi-Loss Alignment ===
Loss Term           Source      FlagScale   Diff        Status
lm_loss             2.3451      2.3453      0.0002      PASS (< 1e-3)
aux_loss (balance)  0.0012      0.0012      0.0000      PASS
mtp_loss            0.4521      0.4519      0.0002      PASS
total_loss          2.7984      2.7984      0.0000      PASS
```

**Guidelines:**
- Match loss coefficients exactly from the source training recipe (these are IMMUTABLE parameters)
- Log each loss term separately in the training report for monitoring
- If a loss term cannot be exactly reproduced (e.g., due to parallelism differences), document the expected divergence
- When adding a new loss, ensure gradient flow is correct — use `torch.autograd.grad` to verify gradients reach the expected parameters

### 6.6 Pre-check: State Dict Key and Shape Validation

Before loading the full model (which can take minutes), validate that the checkpoint keys and shapes match the model definition. This catches weight mapping errors in seconds instead of after a full model load.

```bash
python -c "
from safetensors import safe_open
import torch, json, os

ckpt_path = '<model_path>/model.safetensors'  # or ema.safetensors
# For sharded checkpoints, use model.safetensors.index.json

# 1. List all keys and shapes in the checkpoint
with safe_open(ckpt_path, framework='pt', device='cpu') as f:
    ckpt_keys = {k: f.get_tensor(k).shape for k in f.keys()}

print(f'Checkpoint: {len(ckpt_keys)} tensors')
for k, s in sorted(ckpt_keys.items())[:20]:
    print(f'  {k}: {s}')

# 2. Build model WITHOUT loading weights, check expected keys
# (import model class, instantiate with config, call state_dict())
# model = MyModel(config)
# model_keys = {k: v.shape for k, v in model.state_dict().items()}
# missing = set(model_keys) - set(ckpt_keys)
# unexpected = set(ckpt_keys) - set(model_keys)
# shape_mismatch = {k for k in model_keys & ckpt_keys if model_keys[k] != ckpt_keys[k]}
# print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}, Shape mismatch: {len(shape_mismatch)}')
# for k in shape_mismatch:
#     print(f'  MISMATCH {k}: model={model_keys[k]} ckpt={ckpt_keys[k]}')
"
```

**Common shape mismatch causes:**
- Conv2d patch embedding vs Linear patch embedding: `(out, in, kH, kW)` vs `(out, in*kH*kW)` — call `convert_conv2d_to_linear()` before `load_state_dict()`
- Transposed weight matrices: some frameworks store weights transposed
- Fused vs unfused QKV: checkpoint has separate q/k/v, model expects fused (or vice versa)
- Different vocab sizes: model has padded vocab, checkpoint does not

Fix all shape mismatches before proceeding to full model load.

---

## Step 7: Summary and Memory

After completing all steps, output a summary:

```
=== Model Porting Summary ===
Model: <name>
Source: <paper/huggingface/code>
Mode: <A (config-driven) / B (custom entrypoint)>
Baseline: <closest existing model>

Generated files:
  Checkpoint: tools/checkpoint/<model>/args.py
              tools/checkpoint/<model>/ckpt.py
              tools/checkpoint/<model>/model.py
  Verify:     tools/checkpoint/<model>/verify_alignment.py

Next steps:
  1. Review generated code
  2. Run checkpoint conversion (command above)
  3. Run forward alignment verification
  4. Generate training config (use train-config skill)
  5. Prepare training data (use data-prep skill)
  6. Start training (use train-run skill)
```

Write a memory entry with key findings:
- Model name and architecture type
- Which mode (A/B) was used
- Any non-standard components or known issues
- Checkpoint conversion status

---

## Reference: Existing Model Examples

Models already supported in FlagScale with their key characteristics:

| Model | Size variants | Attention | FFN | Norm | Position | MoE | MLA | MTP | Entrypoint |
|-------|--------------|-----------|-----|------|----------|-----|-----|-----|------------|
| aquila | 1.8B/3B/7B/34B/70B | GQA | SwiGLU | RMSNorm | RoPE | No | No | No | train_gpt.py |
| llama2 | 7B/13B/70B | GQA | SwiGLU | RMSNorm | RoPE | No | No | No | train_gpt.py |
| llama3 | 8B/70B | GQA | SwiGLU | RMSNorm | RoPE | No | No | No | train_gpt.py |
| qwen2_5 | various | GQA | SwiGLU | RMSNorm | RoPE | No | No | No | train_gpt.py |
| qwen3 | 0.6B-235B | GQA+QKnorm | SwiGLU | RMSNorm | RoPE | Optional | No | No | train_gpt.py |
| qwq | 32B | GQA | SwiGLU | RMSNorm | RoPE | No | No | No | train_gpt.py |
| mixtral | 8x7B | GQA | SwiGLU | RMSNorm | RoPE | Yes | No | No | train_gpt.py |
| deepseek_v3 | 11B-671B | MLA | SwiGLU | RMSNorm | RoPE | Yes(shared) | Yes | Yes | train_gpt.py |
| rwkv | various | Linear(custom) | Custom | LayerNorm | None | No | No | No | train_rwkv.py |
| llava1_5 | 7B/13B | GQA+vision | SwiGLU | RMSNorm | RoPE | No | No | No | train_llava.py |
| llava_onevision | various | GQA+vision | SwiGLU | RMSNorm | RoPE | No | No | No | train_llava_onevision.py |
| qwen2_5_vl | various | GQA+vision | SwiGLU | RMSNorm | RoPE | No | No | No | train_qwen2_5_vl.py |
| qwen3_vl | various | GQA+vision | SwiGLU | RMSNorm | RoPE | No | No | No | train_qwen3_vl.py |

### Checkpoint Conversion Inheritance

```
mixtral/ckpt.py          ← base: embedding, attention (GQA), MLP, norm, output layer
  ├── mistral/ckpt.py    ← inherits attention from mixtral, overrides MLP (no MoE)
  │   └── qwen3/ckpt.py  ← inherits from mistral
  └── deepseek_v3/ckpt.py ← custom: MLA attention, MoE MLP, MTP
llama/ckpt.py            ← standalone
aquila/ckpt.py           ← standalone
```

When creating a new model's ckpt.py, find the closest match in this tree and inherit.

---

## Related Skills

- `reproduce` — establish a verified baseline before porting
- `train-config` — generate FlagScale training configuration after porting
- `train-run` — launch training with the ported model
- `precision-alignment` — verify numerical alignment between original and ported model
