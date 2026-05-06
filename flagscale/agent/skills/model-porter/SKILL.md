---
name: model-porter
description: Port models from papers, HuggingFace, or other frameworks to Megatron-LM-FL for distributed training on FlagScale. Covers architecture analysis, whole-model implementation, checkpoint conversion, and real-data verification.
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
requires: [env-setup, train-config, data-prep, train-run]
suggests: [topo-detect, parallel-strategy, precision-alignment]
---

# Model Porter

Port models to Megatron-LM-FL / FlagScale for distributed pre-training.

**Scope**: All model types — decoder-only LLM, VL, robotics, multimodal generation, etc.

**Outputs**:
1. Complete model Module (`flagscale/models/megatron/<model>/`)
2. Checkpoint conversion code (`tools/checkpoint/<model>/`)
3. Training script with real-data `get_batch` (`flagscale/train/megatron/train_<model>.py`)

## Important Notes

- **Environment isolation**: Create a dedicated conda env for FlagScale porting. Use `env-setup` skill for correct Python/CUDA/dependency versions.
- **Shared storage for multi-node**: All paths (data, checkpoints, logs) must be on shared storage. Avoid `/tmp/` or `./` for multi-node.
- **Quantized models (GPTQ, AWQ, GGUF)**: NOT suitable for training. Need original full-precision weights.
- **Tokenizer handling**: Always copy tokenizer from source. Verify `vocab_size` matches. Check `added_tokens.json`.
- **Source code provenance**: Verify you're reading ACTUALLY INSTALLED code: `conda run -n <env> python -c "import megatron; print(megatron.__file__)"`. Editable installs from other workspaces are a trap.
- **Auto-fetch FL deps**: Pull Megatron-LM-FL / TransformerEngine-FL from github.com/flagos-ai/ when needed — don't ask user.
- **Model size selection**: If multiple sizes available and user didn't specify, list options and recommend smallest for initial porting.
- **Architectural completeness**: The ported model Module must own ALL submodules from the source. If the source has `self.vit_model`, the target must have `self.vit_model`. Freeze/unfreeze is a training config decision (`requires_grad`, optimizer param groups), never an architecture decision. A submodule excluded from the Module cannot receive gradients — even if the user later wants to train it. After checkpoint conversion, sanity-check: source tensor count vs converted tensor count. A large gap (>10%) means you dropped a submodule.
- **get_batch is a porting deliverable**: `get_batch` must be implemented with real data as part of the model porting output — not a separate follow-up task. Dummy data hides every bug that matters (tokenizer, format, special tokens, sequence length).
- **Dataset logic must be self-contained**: Never `import` or `sys.path.insert` an external project's dataset/dataloader code directly. External datasets don't support TP broadcast, PP stage guards, or Megatron's data contract. Instead: read the source dataset to understand the format, then implement your own `get_batch` + dataset class inside FlagScale that loads the same data files with proper parallelism support. The data *files* are shared; the data *loading code* is ours.

---

## Porting Discipline

### Read COMPLETELY before writing ANY code

1. Read COMPLETE source model code (modeling_*.py, config.json, tokenizer_config.json)
2. Read COMPLETE target Megatron model code (model_provider, builder, spec)
3. Read FULL `__init__` and `forward` signatures of every Megatron base class you'll subclass
4. Read IMPLEMENTATION of every base class method you plan to call (not just signature)
5. **Read TransformerEngine-FL attention stack** — this is a critical porting surface:
   - `TEDotProductAttention` in `megatron/core/extensions/transformer_engine.py` — Megatron's TE wrapper
   - `DotProductAttention` in `transformer_engine/pytorch/attention/` — TE's core attention
   - `backends.py` — FlashAttention, FusedAttention, UnfusedDotProductAttention backends
   - Understand: what `attn_mask_type` options exist, how `qkv_format` maps to memory layout, how CP integrates with attention
   - If source model uses a non-standard attention (flex_attention, custom masks, sliding window, sparse), map it to TE's equivalent backend and mask type
6. Search FlagScale ecosystem for similar implementations — reuse when possible
7. Build complete mapping table: source layer → target layer with shape transforms
8. Extract ALL config parameters from source config.json (not just obvious ones)
9. Save analysis to workspace before proceeding

### Pre-coding analysis (MANDATORY for models >10B or multimodal)

**Analysis 1: Component diff table**

| Source Component | HF Implementation | Megatron-LM-FL Equivalent | Existing Reference | Gap / Action |
|------------------|-------------------|---------------------------|-------------------|--------------|

Every row must have an explicit action. "TBD" is not acceptable.

**Analysis 1b: Attention mechanism diff (MANDATORY)**

| Aspect | Source Model | TE-FL / Megatron | Mapping / Gap |
|--------|-------------|------------------|---------------|
| Backend | (e.g., flex_attention, sdpa, custom) | TEDotProductAttention → FlashAttention / FusedAttention | |
| Mask type | (e.g., causal, bidirectional, block-sparse, sliding window) | AttnMaskType: no_mask / causal / padding / arbitrary | |
| QKV format | (e.g., separate Q/K/V, fused QKV) | qkv_format: sbhd / bshd / thd | |
| GQA/MQA | (num_kv_heads vs num_heads) | num_gqa_groups in TEDotProductAttention | |
| Position encoding | (RoPE variant, ALiBi, absolute) | Applied before/after TE attention? | |
| Sliding window | (window_size if any) | window_size param in DotProductAttention | |
| Special masking | (cross-modal masks, prefix masks) | How to express via attn_mask_type + attention_mask tensor | |

If source uses flex_attention or custom attention kernels: identify what mask/score_mod functions they apply, then determine the equivalent TE configuration (attn_mask_type + window_size + custom mask tensor). This is a common porting gap — TE supports arbitrary masks via `attn_mask_type="arbitrary"` but performance differs from specialized kernels.

**Fallback: use source model's native attention implementation.** If the source attention cannot be cleanly mapped to TE (e.g., complex score_mod in flex_attention, custom sparse patterns, or novel attention variants), keep the original attention code and integrate it as `core_attention` in the Megatron layer spec instead of `TEDotProductAttention`. This trades TE's fused kernels for correctness and faster porting. The rest of the model (linear layers, norms, embeddings) can still use TE modules — only the attention needs to fall back.

**Analysis 2: Memory budget**

Calculate: params × bytes_per_param (weights + optimizer + gradients + activations) → total per-GPU → choose parallelism.

**Analysis 3: Parallelism strategy** (derived from memory budget)

| Model Size | TP | PP | DP | CP | Min GPUs |
|-----------|----|----|----|----|----------|

---

## Porting Modes

### Mode 1: Config-driven (YAML only)

For standard architectures already in Megatron-LM-FL (GPT, LLaMA, Mistral, Qwen).

- No new model code needed — only YAML config + checkpoint conversion
- Verify: architecture params match source exactly (hidden_size, num_layers, num_heads, intermediate_size, vocab_size, norm_eps)
- Checkpoint conversion: weight name mapping + transpose where needed

### Mode 2: Megatron Native (full parallelism)

For custom architectures needing TP/PP/CP support.

- Implement as MegatronModule with TransformerLayer specs
- Use `ColumnParallelLinear`, `RowParallelLinear` for TP
- Implement pipeline stage splits via `pre_process`/`post_process`
- Reference: `flagscale/models/megatron/qwen2_5_vl/`, `flagscale/models/megatron/bagel/`

### Mode 3: HuggingFace Wrapper (FSDP2 fast path)

For rapid prototyping or models that don't need Megatron parallelism.

- Wrap HF model with FSDP2 sharding
- Limited to DP + FSDP (no TP/PP/CP)
- Fastest path to training but limited scalability

---

## Implementation Flow — Whole Model First

**Core principle**: Analysis is per-component. Implementation is whole-model.

Do NOT verify components in isolation. Do NOT use dummy/synthetic data for verification. Build the complete model as a single nested Module and verify with real data.

### Step 1: Build complete Module structure

Create ONE top-level Module that nests all components:

```
class MyMultimodalModel(MegatronModule):
    def __init__(self, ...):
        self.vision_encoder = VisionTransformer(...)   # ViT
        self.vision_projection = MLP(...)              # bridge
        self.language_model = TransformerDecoder(...)   # LLM
        self.generation_head = DiffusionHead(...)      # VAE/gen (if applicable)

    def forward(self, ...):
        # Wire all components in one forward pass
        vision_features = self.vision_encoder(images)
        projected = self.vision_projection(vision_features)
        output = self.language_model(tokens, visual_embeds=projected)
        return output
```

Reference implementations:
- `flagscale/models/megatron/qwen2_5_vl/qwen2_5_vl_model.py` — ViT + projection + language
- `flagscale/models/megatron/bagel/` — ViT + LLM + VAE generation

### Step 2: Checkpoint conversion (all weights at once)

Convert the ENTIRE checkpoint into the nested structure in one pass:
- Map source weight names → target weight names for ALL components
- Handle shape transforms (transpose, reshape, split/merge heads)
- Verify: `model.load_state_dict()` with `strict=True` — zero missing, zero unexpected keys
- **Completeness sanity check**: Compare source vs converted tensor counts. If you loaded 1200 tensors but only converted 500, you dropped a submodule. Go back and include it. Common mistake: excluding vision encoders or VAE because "they'll be frozen" — wrong, they must still be in the model.

### Step 3: Real data adaptation

Implement `get_batch` with actual dataset as part of the porting deliverable — not a separate follow-up task. This is the primary verification mechanism:
- Tokenizer mismatches → vocab index errors (caught instantly)
- Preprocessing differences → shape mismatches in forward pass
- Sequence length issues → OOM or padding bugs
- Missing special tokens → silent training degradation (caught by loss comparison)

Use the real dataset the model will train on. If the full dataset isn't ready, use a representative subset.

**No dummy data.** `get_batch` must NEVER use `torch.rand`/`torch.zeros`/`torch.randn` or any synthetic tensors — not during development, not for "shape debugging", not as a placeholder. Always load real data from the start. If the data pipeline isn't working, fix it before proceeding. Dummy data hides every bug that matters (tokenizer, format, special tokens, sequence length).

**Own your dataset code.** Do NOT import the source project's dataset/dataloader classes (e.g., `from data.dataset_base import PackedDataset`, `sys.path.insert(0, BAGEL_DIR)`). External dataset code is unaware of Megatron's parallelism contract — it won't call `broadcast_data` for TP, won't guard inputs by PP stage, and may break under multi-node. Instead: read the source dataset code to understand the data format and preprocessing, then write your own dataset + `get_batch` inside FlagScale that reads the same data files with correct parallelism handling. Reuse data *files*, never data *code*.

**Parallelism-aware design.** When the target is distributed training, `get_batch` must handle:
- **TP**: All TP ranks receive identical input — use `broadcast_data` from `megatron.training.utils`
- **PP**: Only first stage needs tokens, only last needs labels — guard with `pre_process`/`post_process`
- **DP**: Different micro-batch per rank — handled by sampler, don't break it with global indexing

Read an existing `get_batch` (e.g., `train_gpt.py`, `train_qwen2_5_vl.py`) before writing yours. Copy the broadcast pattern.

### Step 4: First forward pass = verification

The first successful forward pass with real data that produces a finite loss IS the structural verification. If loss is produced:
- All weight shapes are correct
- All component connections work
- Data pipeline feeds correct formats
- Tokenizer is compatible

---

## Verification Standard

| Level | Criterion | What it proves |
|-------|-----------|----------------|
| 1 | `load_state_dict(strict=True)` passes | All weights mapped correctly |
| 2 | Forward pass with real data → finite loss | Model structure is correct, data pipeline works |
| 3 | Loss decreases over 50-100 steps | Model is learning, gradients flow through all components |
| 4 | Loss curve matches reference within tolerance | Numerical equivalence with source implementation |

Level 2 is the minimum bar before declaring "porting works". Levels 3-4 confirm correctness.

---

## Multimodal Module Nesting

For models with multiple modalities (vision + language + generation):

**Architecture pattern**:
- Top-level Module owns ALL sub-modules
- Pipeline parallelism splits at the top level via `pre_process`/`post_process`/`add_encoder`/`add_decoder`
- Each sub-module is a standard MegatronModule — no standalone verification needed
- Cross-component connections (vision→projection→language) are wired in the top-level `forward()`

**Pipeline stage assignment** (typical VL model):
- Stage 0: vision encoder + projection (when `add_encoder=True`)
- Stages 1..N-1: language model transformer layers
- Stage N: output head (when `post_process=True`)

**Critical wiring points**:
- Vision encoder output → projection layer → language model input embeddings
- Position IDs must account for visual tokens (image token positions)
- Attention mask must handle mixed visual/text sequences
- Loss computation must mask visual token positions appropriately

**Common pitfalls**:
- Vision encoder produces non-zero embeddings but wrong dtype (fp32 vs bf16)
- Projection output shape doesn't match language model hidden_size
- Rotary embeddings applied to visual token positions incorrectly
- Generation components (VAE) not receiving gradients due to detach

---

## Failure Pivot Discipline

**2-strike rule**: Same error category twice → STOP. Don't attempt 3rd fix.

1. Pause execution
2. Root cause audit — re-read relevant source end-to-end
3. Identify systemic gap (wrong assumption upstream, not local bug)
4. Report to user with new approach
5. Only proceed after confirmation

Error categories: shape/dimension, import/module, parallelism, data pipeline, config.

---

## get_batch Under Parallelism

`get_batch` is a critical porting surface:
- **TP**: All TP ranks must receive identical input (use `broadcast_data`)
- **PP**: Only first stage needs tokens, only last needs labels
- **CP**: Sequence split across ranks — correct position IDs and masks
- **DP**: Different micro-batch per rank (handled by sampler)

Verify: print shapes on rank 0 and rank 1, confirm TP ranks match, DP ranks differ.

---

## Related Skills

- `reproduce` — establish verified baseline before porting
- `train-config` — generate FlagScale training configuration
- `train-run` — launch training with ported model
- `precision-alignment` — verify numerical alignment
- `data-prep` — prepare real dataset for verification
