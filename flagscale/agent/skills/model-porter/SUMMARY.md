# Model Porter — Summary

Port HuggingFace/custom models to FlagScale (Megatron-LM-FL).

## Three Porting Modes

| Mode | When | Parallelism | Effort | Artifacts |
|------|------|-------------|--------|-----------|
| **A — Config-driven** | All components in Megatron supported list | Full (TP/PP/DP/EP/CP/SP) | Low | YAML config only |
| **B — Megatron Native** | Custom components, production scale | Full (TP/PP/DP/EP/CP/SP) | High | `train_<model>.py` + native model code |
| **C — HF Wrapper** | Custom components, fast validation | FSDP2 only | Low | `train_<model>_hf_fsdp2.py` + wrapper |

**MANDATORY**: When Mode A doesn't apply, present Mode B vs Mode C trade-offs to user and get explicit confirmation before writing code. A porting path gate enforces this.

## 7-Step Process

1. **Source Model Analysis** — Read config.json, modeling code, identify non-standard components. Output: architecture mapping document.
2. **Architecture Matching** — Compare with Megatron's supported components. Find closest existing FlagScale model.
3. **Determine Porting Path** — Mode A if all components match. Otherwise present Mode B vs Mode C to user with model-specific analysis.
4. **Checkpoint Conversion** — Generate weight mapping, handle TP/PP sharding, verify key counts + shapes + norms.
4.7. **Pre-Launch Validation** — Read FlagScale/Megatron source code to validate config against actual argument parsers, launcher code, and existing examples. Source code is ground truth, not static checklists.
5. **Get Training Running** — 2-step dry run at target parallelism, verify loss is finite and decreasing.
6. **Precision Alignment** — Compare loss/logits with HF reference.
7. **Summary and Memory** — Record findings, decisions, and gotchas.

## Critical Execution Order (Phase 1)

**Data pipeline MUST come first**: get_batch → dataset → model code → training script.

Rationale: If get_batch is wrong, every subsequent step debugs phantom errors. Read existing `examples/<similar_model>/` implementations to understand FlagScale patterns before writing your own.

**2-Strike Pivot Rule**: After 2nd consecutive failure in same error category (shape/import/parallelism/data/config), STOP. Do root-cause audit instead of trying a 3rd incremental fix.

## Mode C Reference Implementation

Files in Megatron-LM-FL:
- `megatron/core/models/huggingface/module.py` — `HuggingFaceModule` base class
- `megatron/core/models/huggingface/qwen_model.py` — Qwen2 LLM wrapper
- `megatron/core/models/huggingface/clip_model.py` — SigLIP ViT wrapper
- `megatron/core/distributed/torch_fully_sharded_data_parallel.py` — FSDP2 integration

Pattern: subclass `HuggingFaceModule`, set `_fsdp_modules`, implement `forward()`.

## Artifact Naming

Mode C uses `_hf_fsdp2` suffix: `train_<model>_hf_fsdp2.py`, `<model>_model_hf_fsdp2.py`, `7b_hf_fsdp2.yaml`. Native (Mode B) has no suffix.

## Key Gates (Engineering Enforced)

- **Porting Path Gate**: Must confirm Mode B/C with user before writing porting code
- **Data Pipeline Gate**: Must implement and verify get_batch BEFORE writing training scripts
- **Reading Depth**: Must read ≥8 files before writing porting code
- **Reading Quality**: Must cover 3/4 categories (source_model, megatron_base, existing_impl, checkpoint)
- **Analysis Persistence**: Must persist analysis to workspace before coding
- **Verification Ladder**: none → analysis → init_ok → forward_aligned → backward_ok → distributed_ok → full_training
- **Pivot Rule**: 2 consecutive failures of same category → mandatory stop and root-cause audit

## Existing Model Examples

| Model | Type | Key Feature |
|-------|------|-------------|
| Qwen3 | Dense + MoE | GQA, SwiGLU, RoPE, QK LayerNorm |
| DeepSeek-V3 | MoE | MLA attention, shared experts, aux-loss-free routing |
| LLaVA-OneVision | Multimodal | HF wrapper + FSDP2 for Qwen LLM + SigLIP ViT |

## Phased Migration Strategy

Phase 1 execution order: component inventory → **data pipeline** → custom layers → entrypoint → checkpoint → verification → training.

Mode C as Phase 0 → Mode B as Phase 1 → Scale up as Phase 2. De-risks native port by providing reference baseline.

**Core principle**: Deep-read FlagScale/Megatron-LM-FL/TransformerEngine-FL source code to understand patterns before implementing. Source code is ground truth — not static checklists or remembered patterns.

## Related Skills

- data-prep: Data pipeline understanding and preparation
- parallel-strategy: Parallelism dimension selection
- train-run: Training launch and monitoring
- precision-alignment: Detailed alignment methodology
