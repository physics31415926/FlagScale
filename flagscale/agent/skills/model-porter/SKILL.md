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
- **Source code provenance**: When reading source code to understand Megatron-LM-FL or any dependency, verify you're reading the ACTUALLY INSTALLED code, not a stale copy from another directory. Run `conda run -n <env> python -c "import megatron; print(megatron.__file__)"` to find the real location. If the installed package is an editable install from a different workspace (e.g., installed from `/workspace/A/` but you're working in `/workspace/B/`), this is a critical mismatch — the code you read for API understanding won't match runtime behavior. Resolve by reinstalling from the correct source tree.
- **Auto-fetch FL dependencies**: When you need to analyze or compile Megatron-LM-FL or TransformerEngine-FL source code and it's not available locally, pull the latest automatically — don't ask the user. Use the repos from env-setup skill (github.com/flagos-ai/Megatron-LM-FL, github.com/flagos-ai/TransformerEngine-FL).
- **Model size selection**: When a model family has multiple size variants (e.g., 0.6B/1.8B/7B/70B) or multiple training configs (e.g., e6_d6_size256 vs e18_d18_size1024) and the user did not specify a size, list the available options with their parameter counts and ask the user to choose. Recommend the smallest variant for initial porting/verification, but let the user decide.

## Porting Discipline — lessons from real failures

### Read COMPLETELY before writing ANY code

Model porting is the task most prone to "understand 20%, implement, then debug for 80%". This wastes enormous effort.

**Before writing a single line of porting code:**
1. Read the COMPLETE source model code (modeling_*.py, config.json, tokenizer_config.json)
2. Read the COMPLETE target Megatron model code (the relevant model_provider, builder, and spec)
3. **Read the FULL `__init__` and `forward` signatures of every Megatron base class you will subclass or call** — TransformerLayer, SelfAttention, TransformerBlock, MLP, etc. These APIs evolve (e.g., `pg_collection` was added to TransformerLayer). If your subclass doesn't accept a parameter the base class passes, you get a TypeError at runtime. Read the CURRENT signatures, not what you remember from a previous session.
4. **Read the IMPLEMENTATION (not just signature) of every base class method you plan to call from your custom code.** Megatron methods often do more than their name suggests — e.g., `get_query_key_value_tensors()` internally calls `self.linear_qkv()` again, so passing already-projected QKV to it causes double projection. You cannot safely call a method you haven't read the body of.
5. **Understand existing implementations before writing new code.** For every component you need to port, first search the FlagScale ecosystem (flagscale/models/, Megatron-LM-FL, TransformerEngine-FL, tools/checkpoint/) for similar implementations. Understand how they handle TP, pipeline integration, dtype, and checkpoint conversion. Reuse or adapt when possible; when writing from scratch, base your patterns on what already works in the codebase.
6. Build a complete mapping table: source layer name → target layer name, with shape transformations
7. Identify ALL non-standard components (custom attention, custom FFN, MoE routing, etc.)
8. **Extract ALL config parameters from the source model's config.json** — not just the obvious ones (hidden_size, num_layers). Every parameter that affects tensor shapes or model behavior must be captured and mapped to the corresponding Megatron config field. This includes activation function type and gating, positional encoding parameters, vocabulary size, bias settings per layer type, normalization constants, and any model-specific dimensions. Missing a single parameter causes shape mismatches that waste hours of debugging.
9. Save this analysis to workspace_state before proceeding

**Read whole files, not fragments.** Don't use `sed -n '100,130p'` to read 30-line snippets — you'll miss context. Use `read_file` to read the complete file or at least the complete class/method. Piecemeal reading leads to piecemeal understanding.

### Pre-coding analysis gate (MANDATORY for models >10B params or multimodal)

Before writing ANY porting code, complete and present these three analyses. These are not optional — skipping them caused real failures (wrong parallelism, missing components, broken checkpoints).

**Analysis 1: Component-by-component diff table**

For every architectural component in the source model, document how it maps to Megatron-LM-FL:

```
| Source Component       | HF Implementation         | Megatron-LM-FL Equivalent    | Existing Reference             | Gap / Action Required          |
|------------------------|---------------------------|------------------------------|--------------------------------|--------------------------------|
| LLM backbone           | Qwen2ForCausalLM          | GPTModel (config-driven)     | qwen2 model in flagscale       | Config mapping only            |
| Vision encoder (ViT)   | SigLIPVisionModel         | CLIPViTModel (adaptable)     | CLIPViTModel in mcore          | Modify image preprocessing     |
| MoT routing            | custom MoTLayer           | NONE                         | none found                     | Implement from scratch         |
| Flow matching / VAE    | custom FlowMatching       | NONE                         | none found                     | Implement from scratch         |
```

The "Existing Reference" column forces you to look before writing. Filling it in means you've studied what's already there — even when the answer is "none found".
| Checkpoint TP split    | N/A (single-GPU weights)  | sharded_state_dict needed    | Implement for each custom layer|
```

Every row must have an explicit action. "TBD" or blank is not acceptable.

**Analysis 2: Memory budget calculation**

Calculate BEFORE choosing parallelism strategy:

```
Model parameters:        <N>B (total), <M>B (active if MoE/MoT)
Bytes per param (bf16):  2 bytes × <N>B = <X> GB (weights only)
Adam optimizer states:   12 bytes × <N>B = <Y> GB (fp32 weights + fp32 momentum + fp32 variance)
Gradients (bf16):        2 bytes × <N>B = <Z> GB
Activation memory:       estimate based on batch_size × seq_len × hidden_size × num_layers
─────────────────────────────────────────────
Total per-GPU (no parallelism): <W> GB
Available GPU memory:    <V> GB per GPU × <K> GPUs
```

**Analysis 3: Parallelism strategy (derived from memory budget)**

Based on the memory budget above, choose parallelism:
- Total memory fits in 1 GPU → DP only (single-GPU verification OK)
- Total memory fits in 1 GPU with activation checkpointing → DP + activation checkpointing
- Total memory > 1 GPU → MUST use TP, PP, or FSDP. **Never use bare DDP for models >10B params** — DDP replicates the full model on every GPU, so a 14B model needs ~196GB per GPU with Adam, which exceeds any single GPU.
- For verification scripts: use the MINIMUM parallelism that fits in available hardware. If the model doesn't fit on one GPU, the verification script must use TP or model sharding — not DDP.

Save all three analyses to workspace_state. Present to user for confirmation before writing code.

**Analysis 4: Data pipeline compatibility**

Before writing any training script, analyze the source model's data pipeline and its compatibility with Megatron's `pretrain()`:

```
| Aspect                  | Source Model              | Megatron pretrain()                  | Gap / Action                    |
|-------------------------|---------------------------|--------------------------------------|---------------------------------|
| Dataset type            | (IterableDataset? Map?)   | Map-style by default, cyclic loader  | Adapter if iterable             |
| Batch format            | (what fields?)            | Determined by get_batch/forward_step | Custom get_batch if different   |
| Sequence handling       | (packed? padded? variable?)| Fixed-length, optional packing      | Packing adapter if needed       |
| Multimodal inputs       | (how are they batched?)   | Text-only by default                 | Custom data provider            |
```

Key questions to answer:
- Does the source use `IterableDataset` or `Dataset`? Megatron's standard pipeline expects indexed datasets.
- Does the source use packed sequences? If so, how are sequence boundaries marked?
- What is the exact batch dict format? Map every field to what `get_batch()` / `forward_step()` expects.
- For multimodal: how are image/video tokens interleaved with text? How are they batched?

If the data pipeline is incompatible, plan the adapter BEFORE writing the training script — not after discovering the mismatch during implementation.

### Verify runtime state, not config declarations

Config objects and code structure can be misleading about what actually happens at runtime. Before writing any conversion or splitting code, verify these by instantiation (on meta device or with tiny shapes):

1. **Target model's actual state_dict keys**: Instantiate the target model and print `model.state_dict().keys()`. Do NOT guess key names from class hierarchy — nested modules, shared parameters, and framework-injected keys (like TransformerEngine's `_extra_state`) make guessing unreliable.

2. **Actual parallelism behavior**: A config field like `tensor_model_parallel_size=1` on a subnetwork does NOT mean that subnetwork runs with TP=1. Framework layers (ColumnParallelLinear, RowParallelLinear, TE layers) typically query the global process group, not the config. Verify by checking the actual tensor shapes after model construction — if a layer's weight shape is divided by global TP, it IS TP-sharded regardless of what the config says.

3. **Vocab size transformation chain**: Tokenizers may add special tokens (e.g., NullTokenizer adds +1 for EOD). Then `make_vocab_size_divisible_by × TP` rounds up further. Trace the full chain: `raw_vocab → tokenizer adjustment → padding → per-rank size` and verify the final per-rank size matches your checkpoint BEFORE launching training.

4. **Activation function and gating**: A CLI flag like `--swiglu` may set multiple config fields (e.g., `gated_linear_unit=True` which doubles fc1 output size, plus `activation_func`). When constructing configs manually in `model_provider`, these implicit settings are NOT auto-propagated — you must set them explicitly. Verify by checking the actual fc1 weight shape after construction.

5. **dtype at component boundaries**: In multimodal or multi-component models, different subnetworks (ViT, connector, LLM, VAE) may operate in different dtypes. Position embeddings, normalization layers, and custom modules often default to fp32 even when the rest of the model is bf16. The forward pass will crash at the first `nn.Linear` or `matmul` where input dtype doesn't match weight dtype. Before writing the forward pass, trace the dtype through every component boundary (encoder output → connector input, connector output → LLM input, etc.) and add explicit `.to(dtype)` casts at each boundary. Do NOT wait for runtime crashes to discover these — add the casts proactively during implementation, and verify them in Tier 2 (random-weight forward).

The pattern: **instantiate first, then write conversion code to match what you observed** — not the other way around.

### No approach flip-flopping

When porting, you'll face choices (e.g., use TE attention vs custom attention, THD format vs standard padding). The rule:
1. List ALL constraints before choosing
2. Pick one approach and commit
3. If it fails, record WHY it failed before trying the next approach
4. Never flip between approaches more than twice — if A→B→A happens, stop and ask the user

### Design before writing — especially for custom components

When implementing a custom component from scratch (MoT layer, MoE router, flow matching, etc.):

1. **Write a design sketch first** (in your response, not a file): class hierarchy, key methods, data flow (what tensors flow in/out, what shapes). Keep it to 10-20 lines of pseudocode.
2. **Validate the design against the source model**: trace the source forward pass to confirm your component receives the right inputs and produces the right outputs. Pay special attention to shared vs separate submodules (e.g., does MoT share attention weights between paths or use separate modules?).
3. **Only then write the full implementation.**

This prevents the "write 300 lines → realize the design is wrong → rewrite 300 lines" pattern. A 10-line pseudocode sketch is cheap to throw away; a full implementation is not. The most common design mistake is misunderstanding which submodules are shared vs duplicated — always verify this from the source code before committing to a class hierarchy.

### Add diagnostic prints during development

When writing new model code, checkpoint conversion, or training scripts, add print statements at key points BEFORE the first run:
- **Forward pass**: print shape and dtype at every component boundary (encoder output, connector output, LLM input, etc.)
- **Checkpoint loading**: print key counts (loaded, missing, unexpected), sample key names and shapes
- **Config resolution**: print final values of critical parameters after all transformations (vocab_size after padding, ffn_hidden_size after gating, per-rank shapes after TP split)

The goal is to get a complete diagnostic picture on the FIRST run, whether it succeeds or fails. Without these prints, a crash gives you only a traceback — with them, you see exactly where shapes, dtypes, or keys diverged from expectations. This eliminates the crash → guess → add print → rerun → crash cycle that wastes multiple GPU-minutes per iteration.

Remove or downgrade to logging.debug after the component is verified working.

**Distributed training visibility**: In multi-GPU training, plain `print()` from worker processes is often buffered or lost. Always use `print(..., flush=True)` or `print(..., file=sys.stderr)`. When using FlagScale Launcher, each rank's output is captured in separate log files — check per-rank logs for debug output. If debug prints don't appear in the main log, check per-rank files before concluding the code path wasn't reached.

### Verify fundamentals first

After the first successful training launch, check these IN ORDER before celebrating:
1. **Loss sanity**: is `ce_loss` close to `ln(vocab_size)`? If yes → model outputs random, something is fundamentally broken
2. **Gradient flow**: is `num_zeros` / total_params < 50%? If not → gradients not flowing
3. **Weight loading**: did `params_norm` start at a reasonable value (not 0, not identical to random init)?
4. **Loss trend**: does loss decrease over 50 iterations?

Only after ALL four pass should you proceed to precision alignment or scaling.

### Verify ALL model components are integrated

For multimodal or multi-component models (VL, MoT, robotics), verify EACH component individually before end-to-end testing:
1. **Vision encoder (ViT/SigLIP)**: Does it produce non-zero embeddings? Are image tokens reaching the LLM?
2. **Generation components (VAE/flow matching)**: Are they connected in the forward pass? Do they receive gradients?
3. **Routing layers (MoE/MoT)**: Are experts being activated? Is the routing loss included?
4. **Auxiliary losses**: Are ALL loss terms from the source model present? Compare loss breakdown (lm_loss, router_loss, etc.) against source.

A model that "runs without errors" but silently ignores the ViT or VAE is a broken port.

### Verification scripts must match the actual API

Before writing a Tier 2 verification script (forward+backward test), read the `forward()` signatures of every module you'll call — especially the top-level model, `TransformerBlock.forward()`, and any custom layer's `forward()`. Common failures:
- Passing kwargs that the module doesn't accept (e.g., custom kwargs to `TransformerBlock.forward()`)
- Missing required initialization (CUDA RNG tracker, process groups, parallel state)
- Using config fields that don't exist on the config class you're using

A verification script that takes 6 rounds of fix-run-fix to pass is a sign you didn't read the APIs before writing it. Read first, write once.

### TransformerEngine attention mask gotcha

TE's `DotProductAttention` with `attn_mask_type="causal"` IGNORES any custom attention mask you pass. If your model needs a non-standard mask (e.g., per-sample causal for packed sequences), you must either:
- Use `attn_mask_type="arbitrary"` (slower but respects your mask)
- Use THD format with `cu_seqlens` (efficient, but requires careful setup)
- Verify the mask is actually being used by checking intermediate attention outputs, not just loss

### get_batch is a critical porting surface — treat it with the same rigor as model code

`get_batch` (or `get_batch_on_this_tp_rank` / the dataloader collate function) is where raw data becomes model input tensors. Under parallelism, this is deceptively complex:

**What get_batch must handle correctly:**
- **TP**: All TP ranks must receive identical input tensors (same tokens, same masks, same labels). If get_batch produces different data on different TP ranks, the model silently computes on inconsistent inputs — loss looks normal but the model learns garbage. Use `broadcast_data` or ensure the dataloader is seeded identically across TP ranks.
- **PP**: Only the first pipeline stage needs input tokens/embeddings; only the last stage needs labels. Intermediate stages need neither. Sending full data to all stages wastes memory. Check `mpu.is_pipeline_first_stage()` / `is_pipeline_last_stage()`.
- **CP (Context Parallelism)**: Sequence is split across CP ranks. get_batch must shard the sequence dimension correctly and provide the right position IDs for each shard. Attention masks must account for cross-shard dependencies.
- **DP**: Each DP rank gets a different micro-batch — this is the easy case, handled by the dataloader sampler.
- **EP (Expert Parallelism)**: Usually transparent to get_batch, but verify that token routing metadata (if any) is consistent across EP ranks.

**Multimodal models add more complexity:**
- Image/video tokens, text tokens, and generation tokens may need different padding, masking, and position encoding
- Packed sequences (multiple samples in one sequence) require `cu_seqlens` or equivalent metadata
- Routing indexes for multi-path architectures (e.g., understanding vs generation paths) must be consistent with the actual token layout in the batch

**Verification checklist for get_batch:**
1. Print tensor shapes and a few values on rank 0 and rank 1 of each parallel group — confirm TP ranks match, DP ranks differ
2. Verify position IDs are correct (especially for packed sequences and CP)
3. Verify attention masks match the expected pattern (causal, bidirectional for vision encoders, custom for multi-path architectures)
4. Verify labels are only computed on the correct pipeline stage
5. Run a few training steps and confirm loss is identical across TP ranks (they should be, since they see the same data)

**Common get_batch bugs:**
- Using `torch.randint` without setting the same seed across TP ranks → inconsistent inputs
- Forgetting to broadcast data from DP rank 0 to TP group → each TP rank generates its own random batch
- Padding/truncation that changes sequence length differently on different ranks → shape mismatch in all-reduce
- Position IDs not accounting for CP sequence offset → wrong RoPE embeddings on non-zero CP ranks

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

**CRITICAL: Convert for the target parallelism.** The checkpoint MUST be converted with the same TP/PP that will be used for training. If the analysis phase determined TP=4 PP=1, the conversion command MUST use `--target-tensor-parallel-size 4 --target-pipeline-parallel-size 1`. Converting to TP=1 and then trying to load with TP=4 will fail — Megatron legacy checkpoints cannot be resharded at load time. The decided parallelism is a binding constraint for all downstream steps (conversion, config, data processing, launch). Do not change it to work around a failure; fix the failing step instead.

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

**TP/PP splitting completeness**: When writing a checkpoint splitter (or using the conversion tool with `--target-tensor-parallel-size`), you MUST handle ALL parameter variants that need splitting, not just the obvious ones. Models with multiple subnetworks (multimodal, mixture-of-experts, mixture-of-transformers) often have parallel sets of parameters with different name suffixes or prefixes that all require the same split logic.

**TP splitting must cover ALL components, not just the LLM backbone.** In multimodal models, vision encoders (ViT/SigLIP), connectors, VAE decoders, and other subnetworks may also contain ColumnParallelLinear or RowParallelLinear layers that are TP-sharded at runtime. If you only split the LLM transformer layers, the non-LLM components will have shape mismatches at load time. To find ALL TP-sharded parameters: instantiate the model on meta device with the target TP size, then compare every key's shape against the single-GPU checkpoint. Any key where `model_shape != ckpt_shape` needs a split rule — regardless of which subnetwork it belongs to.

Before running the splitter, use the 4.5.1 cross-check: instantiate the target model on meta device, compare every key's shape between the checkpoint and the model. Any shape mismatch means the splitter missed a parameter. Do NOT discover these mismatches by launching training — the cross-check takes seconds, a failed training launch takes minutes.

### 4.5.1 Post-Conversion Checkpoint Verification (MANDATORY)

After saving a converted checkpoint, verify it BEFORE proceeding. A 28GB file on disk does not mean a correct checkpoint. Run these checks immediately after conversion:

```python
import torch

# 1. Reload and verify key count
ckpt = torch.load("<saved_ckpt_path>", map_location="cpu")
state_dict = ckpt["model"] if "model" in ckpt else ckpt
print(f"Saved checkpoint: {len(state_dict)} keys")

# 2. Verify total parameter count matches expectation
total_params = sum(v.numel() for v in state_dict.values())
print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
# Compare against known model size — e.g., 7B model should have ~7e9 params

# 3. Spot-check key names and shapes
for k, v in sorted(state_dict.items())[:10]:
    print(f"  {k}: {v.shape} {v.dtype}")

# 4. CRITICAL: Cross-check against model state_dict
# Instantiate the target Megatron model on meta device and compare ALL keys
with torch.device("meta"):
    model = build_model(config)  # same config as training
model_keys = {k: v.shape for k, v in model.state_dict().items()}
ckpt_keys = {k: v.shape for k, v in state_dict.items()}

# Check every model key has a matching checkpoint key with correct shape
missing_from_ckpt = set(model_keys) - set(ckpt_keys)
extra_in_ckpt = set(ckpt_keys) - set(model_keys)
shape_mismatch = {k for k in model_keys.keys() & ckpt_keys.keys()
                  if model_keys[k] != ckpt_keys[k]}

# Filter out expected mismatches (_extra_state from TE, padded vocab)
missing_real = {k for k in missing_from_ckpt if '_extra_state' not in k}
print(f"Missing from ckpt (excluding _extra_state): {len(missing_real)}")
print(f"Extra in ckpt: {len(extra_in_ckpt)}")
print(f"Shape mismatch: {len(shape_mismatch)}")
for k in sorted(missing_real):
    print(f"  MISSING: {k} (model expects {model_keys[k]})")
for k in sorted(shape_mismatch):
    print(f"  SHAPE: {k} ckpt={ckpt_keys[k]} model={model_keys[k]}")
```

**Gate**: Zero missing keys (excluding `_extra_state`), zero shape mismatches. If ANY model component's keys are missing, the conversion is incomplete — fix the conversion script, do not proceed to training. Common failure patterns:
- Subnetwork keys still in source format (e.g., HF naming) instead of target format — conversion script only handled the main network
- Variant/suffix parameters missing — conversion script only handled base parameter names, not model-specific variants
- Keys with wrong nesting depth — double-nested prefix instead of single

This cross-check is the MOST IMPORTANT verification step. Skipping it and discovering key mismatches during training wastes 10-30 minutes per attempt.

**Gate**: Key count is non-zero, total parameter count is within 5% of expected model size, key names follow the expected Megatron naming convention (e.g., `decoder.layers.0.self_attention.linear_qkv.weight`).

### 4.5.2 Missed/Skipped Keys Audit (MANDATORY)

If the conversion script reports missed, skipped, or unexpected keys:

1. **Capture the FULL list** — not just the last few lines of terminal output
2. **Categorize every key by prefix** — group by component (e.g., `decoder.*`, `encoder.*`, `vision_model.*`, `language_model.*`)
3. **Verify that NO model-critical keys were skipped** — if any key with a prefix that should have been converted appears in the missed list, the conversion has a bug
4. **Document the expected skips** — e.g., "N skipped keys, all with prefix `X.` — these are weights for a separately-loaded component"

Do NOT assume missed keys are harmless based on seeing a few keys with an expected prefix. The full list may contain keys from other components that were silently skipped due to a prefix mismatch in the conversion code.

```bash
# Quick audit: group missed keys by top-level prefix
python -c "
keys = [...]  # paste or load the missed keys list
from collections import Counter
prefixes = Counter(k.split('.')[0] for k in keys)
for prefix, count in prefixes.most_common():
    print(f'  {prefix}: {count} keys')
"
```

### 4.6 Three-Tier Pre-flight Verification (MANDATORY before Step 5)

Loading a large checkpoint onto GPUs takes minutes. Most failures (missing packages, shape mismatches, dtype errors, broken forward pass) can be caught in seconds without touching the real checkpoint. Run these tiers in order — only proceed to the next tier when the current one passes completely.

**Tier 1: Zero-cost pre-checks (seconds)**

Run ALL of these before any model instantiation:

```python
# 1a. Dependency imports — catch missing packages immediately
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
from megatron.plugin.platform import get_platform
print(f"Megatron-LM-FL: OK (platform={get_platform()})")
import transformer_engine; print(f"TransformerEngine-FL: {transformer_engine.__version__}")
import apex; print("Apex: OK")
import flash_attn; print(f"Flash-Attention: {flash_attn.__version__}")

# 1b. Config sanity — catch arithmetic errors before they become shape mismatches
config = ...  # load model config
assert config["hidden_size"] % config["num_attention_heads"] == 0, "hidden_size must be divisible by num_heads"
assert config["num_attention_heads"] % config.get("num_key_value_heads", config["num_attention_heads"]) == 0, "num_heads must be divisible by num_kv_heads"
tp = <target_tp_size>
assert config["num_attention_heads"] % tp == 0, f"num_heads ({config['num_attention_heads']}) not divisible by TP ({tp})"
assert config.get("num_key_value_heads", config["num_attention_heads"]) % tp == 0, f"num_kv_heads not divisible by TP ({tp})"

# 1b-2. Training config arithmetic — catch batch size / parallelism mismatches
dp = num_gpus // (tp * pp)
micro_batch = 1  # from config
global_batch = 8  # from config
assert global_batch % (micro_batch * dp) == 0, f"global_batch ({global_batch}) must be divisible by micro_batch ({micro_batch}) × DP ({dp})"
grad_accum = global_batch // (micro_batch * dp)
print(f"DP={dp}, micro_batch={micro_batch}, global_batch={global_batch}, grad_accum_steps={grad_accum}")

# 1c. Checkpoint key/shape audit — read metadata only, no GPU memory used
from safetensors import safe_open
with safe_open("<ckpt_path>", framework="pt", device="meta") as f:
    ckpt_keys = {k: f.get_tensor(k).shape for k in f.keys()}
print(f"Checkpoint: {len(ckpt_keys)} tensors")
# Instantiate model on meta device (zero memory)
with torch.device("meta"):
    model = MyModel(config)
model_keys = {k: v.shape for k, v in model.state_dict().items()}
missing = set(model_keys) - set(ckpt_keys)
unexpected = set(ckpt_keys) - set(model_keys)
shape_mismatch = {k for k in model_keys.keys() & ckpt_keys.keys() if model_keys[k] != ckpt_keys[k]}
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}, Shape mismatch: {len(shape_mismatch)}")
for k in sorted(shape_mismatch):
    print(f"  MISMATCH {k}: model={model_keys[k]} ckpt={ckpt_keys[k]}")
```

**Gate**: ALL imports succeed, ALL config assertions pass, zero shape mismatches (or all mismatches have known transforms in the conversion code). If any fail, fix before proceeding.

**Tier 1b: Checkpoint ↔ parallelism compatibility (seconds)**

Before launching training, verify the checkpoint format is compatible with the target parallelism:

```python
import torch, os

ckpt_dir = "<megatron_ckpt_path>"
target_tp = 4
target_pp = 1

# Check checkpoint TP/PP by examining directory structure
# Legacy format: iter_XXXXXXX/mp_rank_XX/model_optim_rng.pt
# Dist-ckpt format: iter_XXXXXXX/mp_rank_XX_XXX/ (multiple shards)
ckpt_files = []
for root, dirs, files in os.walk(ckpt_dir):
    for f in files:
        if f.endswith('.pt') or f.endswith('.distcp'):
            ckpt_files.append(os.path.join(root, f))

# Count mp_rank directories to determine saved TP×PP
mp_ranks = set()
for f in ckpt_files:
    parts = f.split('/')
    for p in parts:
        if p.startswith('mp_rank_'):
            mp_ranks.add(p)

saved_tp_pp = len(mp_ranks) if mp_ranks else 1
print(f"Checkpoint saved with {saved_tp_pp} model-parallel ranks")
print(f"Target: TP={target_tp}, PP={target_pp} → {target_tp * target_pp} ranks")

if saved_tp_pp != target_tp * target_pp:
    print("WARNING: Checkpoint TP×PP mismatch!")
    print("  Options:")
    print("  1. Re-convert checkpoint with target TP/PP (recommended)")
    print("  2. Use dist_checkpointing with resharding (if supported)")
    print("  3. Match runtime TP×PP to checkpoint (may cause OOM)")
```

**Gate**: Checkpoint TP×PP matches runtime TP×PP, OR you have a verified resharding path. Do NOT launch training with a mismatch — it will either fail to load or silently load incorrect weights.

Also verify memory budget when changing parallelism:
```python
model_params = <N>  # from checkpoint param count
bytes_per_param_model = 2  # bf16
bytes_per_param_optim = 8  # Adam: fp32 copy + momentum + variance
dp_size = <num_gpus> // (target_tp * target_pp)
use_distributed_optim = True

per_gpu_model = model_params * bytes_per_param_model / target_tp / target_pp
per_gpu_grad = per_gpu_model  # same size as model
per_gpu_optim = (model_params * bytes_per_param_optim / dp_size) if use_distributed_optim else (model_params * bytes_per_param_optim / target_tp / target_pp)

total_gb = (per_gpu_model + per_gpu_grad + per_gpu_optim) / 1e9
gpu_mem_gb = <gpu_memory_in_gb>
print(f"Estimated per-GPU memory: {total_gb:.1f} GB (model={per_gpu_model/1e9:.1f} + grad={per_gpu_grad/1e9:.1f} + optim={per_gpu_optim/1e9:.1f})")
print(f"GPU memory available: {gpu_mem_gb} GB")
if total_gb > gpu_mem_gb * 0.9:  # leave 10% headroom for activations
    print("WARNING: Will likely OOM! Increase TP or enable activation checkpointing.")
```

**Tier 2: Random-weight forward/backward (tens of seconds)**

Instantiate the model with random weights (no checkpoint loading) and run one micro-batch through the full compute graph:

```python
import torch
model = MyModel(config).cuda().bfloat16()  # random init, no ckpt

# Build a minimal dummy batch matching the model's expected input format
dummy_input = build_dummy_batch(
    batch_size=1, seq_len=128, vocab_size=config["vocab_size"],
    device="cuda", dtype=torch.bfloat16,
)

# Forward
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(**dummy_input)

# Verify output structure
assert "loss" in output or isinstance(output, torch.Tensor), "Model must return loss or tensor"
loss = output["loss"] if isinstance(output, dict) else output
print(f"Forward OK — loss shape: {loss.shape}, dtype: {loss.dtype}, value: {loss.item():.4f}")

# Backward
loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
zero_grads = [n for n, g in grad_norms.items() if g == 0.0]
print(f"Backward OK — {len(grad_norms)} params with grads, {len(zero_grads)} with zero grad")
if zero_grads:
    print(f"  WARNING zero-grad params: {zero_grads[:5]}")
```

This tier catches: dtype mismatches (fp32/bf16 promotion errors), attention mask shape errors, MoT/MoE routing bugs, missing forward connections (ViT, VAE not wired up), and backward graph breaks. All without waiting for checkpoint loading.

**Gate**: Forward produces finite loss, backward produces non-zero gradients for all trainable parameters. If any component (ViT, VAE, routing) is present in the model but produces zero gradients, it's not connected — fix before proceeding.

**Tier 3: Real checkpoint loading and validation (minutes)**

Only after Tier 1 and Tier 2 pass:

```python
# Load real checkpoint
state_dict = load_checkpoint("<ckpt_path>")  # or converted Megatron ckpt
result = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {result.missing_keys[:5]}")
print(f"Unexpected: {result.unexpected_keys[:5]}")

# Verify weights are loaded (not random)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: norm={param.data.norm().item():.4f}")
        break

# Forward with real weights — compare against source model if possible
with torch.no_grad():
    output = model(**dummy_input)
print(f"Real-weight forward OK — loss: {output['loss'].item():.4f}")
```

**Gate**: No missing keys (or all missing keys are expected, e.g., optimizer states). Parameter norms are non-zero and reasonable (not identical to random init norms from Tier 2).

**Summary**: Tier 1 catches environment and config errors (seconds). Tier 2 catches code-level bugs in the compute graph (tens of seconds). Tier 3 catches weight mapping errors (minutes). Most debugging happens in Tier 1-2 without ever loading a checkpoint.

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

## Checkpoint Verification Protocol

After checkpoint conversion, verify BEFORE proceeding to training:

### Quick Checks (seconds)
```python
import torch
ckpt = torch.load("<converted_checkpoint>/mp_rank_00/model_optim_rng.pt", map_location="cpu")
state = ckpt["model"]
print(f"Total keys: {len(state)}")
print(f"Sample shapes: {[(k, v.shape) for k, v in list(state.items())[:10]]}")
print(f"Dtypes: {set(v.dtype for v in state.values())}")
```

### Key Count Match
Compare converted checkpoint keys against the target model's expected keys (from meta-device instantiation):
```python
expected_keys = set(model.state_dict().keys())
actual_keys = set(state.keys())
missing = expected_keys - actual_keys
unexpected = actual_keys - expected_keys
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing: print(f"Missing samples: {list(missing)[:10]}")
if unexpected: print(f"Unexpected samples: {list(unexpected)[:10]}")
assert len(missing) == 0, "Checkpoint is incomplete — conversion has bugs"
```

### Shape Match
```python
for key in expected_keys:
    if key in actual_keys:
        expected_shape = model.state_dict()[key].shape
        actual_shape = state[key].shape
        if expected_shape != actual_shape:
            print(f"SHAPE MISMATCH: {key}: expected {expected_shape}, got {actual_shape}")
```

### Norm Sanity Check
Converted weights should NOT look like random init:
```python
for key in list(state.keys())[:20]:
    norm = state[key].float().norm().item()
    print(f"{key}: norm={norm:.4f}")
# Random init norms are typically ~0.01-0.1 for small tensors
# Real weights have larger, varied norms
```

### Source Code Provenance
When reading source code to understand Megatron-LM-FL or any dependency, verify you're reading the ACTUALLY INSTALLED code:
```bash
conda run -n <env> python -c "import megatron; print(megatron.__file__)"
```
If the installed package is an editable install from a different workspace, this is a critical mismatch — resolve by reinstalling from the correct source tree.

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
