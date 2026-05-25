# Infer-HW-Adapt — Summary

Adapt and fix vllm-plugin-FL for specific hardware backends after plugin version upgrades.

**Load when**: running adaptation tests on non-NVIDIA backends, diagnosing Triton compilation failures after vLLM upgrade, creating patches for hardware-specific issues, or preparing hardware adaptation PRs.

**Requires**: `infer-env-setup` (environment must be set up first — SSH, container, vLLM/plugin/FlagGems installed).

**Full cycle**: progressive testing (unit → functional → offline → serving) → test-diagnose-patch loop → patch review → PR preparation.

**Key principles**:
- Never modify vLLM source — all adaptations go through plugin patches
- All test output persisted to `/workspace/adapt-logs/` for efficient re-reading
- One patch per failure, gated by platform check
- Minimum diff — remove anything unnecessary before PR

**Constraints**: 8 hard constraints covering testing (progression order, log persistence, model path verification), code (no vLLM source mods, Triton diagnosis before patch, patch review, logical commits, sensitive content check). 1 soft constraint for code sync before test.
