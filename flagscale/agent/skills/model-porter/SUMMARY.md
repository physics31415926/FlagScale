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
5. **Get Training Running** — 2-step dry run at target parallelism, verify loss is finite and decreasing.
6. **Precision Alignment** — Compare loss/logits with HF reference.
7. **Summary and Memory** — Record findings, decisions, and gotchas.

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
- **Reading Depth**: Must read ≥8 files before writing porting code
- **Reading Quality**: Must cover 3/4 categories (source_model, megatron_base, existing_impl, checkpoint)
- **Analysis Persistence**: Must persist analysis to workspace before coding
- **Verification Ladder**: none → analysis → init_ok → forward_aligned → backward_ok → distributed_ok → full_training

## Existing Model Examples

| Model | Type | Key Feature |
|-------|------|-------------|
| Qwen3 | Dense + MoE | GQA, SwiGLU, RoPE, QK LayerNorm |
| DeepSeek-V3 | MoE | MLA attention, shared experts, aux-loss-free routing |
| LLaVA-OneVision | Multimodal | HF wrapper + FSDP2 for Qwen LLM + SigLIP ViT |

## Phased Migration Strategy

Mode C as Phase 0 → Mode B as Phase 1 → Scale up as Phase 2. De-risks native port by providing reference baseline.

## Related Skills

- data-prep: Data pipeline understanding and preparation
- parallel-strategy: Parallelism dimension selection
- train-run: Training launch and monitoring
- precision-alignment: Detailed alignment methodology
