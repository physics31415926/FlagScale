# Gate System

## Overview

The gate system is a pre/post-execution safety layer that intercepts tool calls before they run. Gates can **hard-block** (prevent execution entirely) or **soft-warn** (inject advisory messages without blocking).

## Design

Gates are methods on `GatesMixin` with signature `(self, tool_name: str, arguments: dict) -> str`. They return:
- Empty string `""`: no action (gate passes)
- Non-empty string: the gate message (block or warning text)

Gates are organized into two lists:
- `hard_gates`: Return includes `"TOOL NOT EXECUTED"` — the tool call is skipped
- `soft_gates`: Return is injected as advisory text but execution proceeds

## Gate Inventory (31 methods)

### Hard Gates

| Gate | Purpose |
|------|---------|
| `_check_training_hang` | Blocks when training appears stalled (scoring-based, needs ≥2 signals) |
| `_check_distributed_prerequisite_gate` | Ensures distributed setup before multi-node operations |
| `_check_checkpoint_verified_gate` | Requires `inspect_checkpoint` deep verification before training launch |
| `_check_pipeline_comprehension_gate` | Ensures pipeline code is understood before modification |
| `_check_structure_completeness_gate` | Blocks checkpoint conversion until model structure is enumerated |
| `_check_phase_ordering_gate` | Enforces 6-phase strict ordering (analysis→training) |
| `_check_data_parallelism_gate` | Requires parallelism strategy documentation before data pipeline code |

### Soft Gates (Warnings)

| Gate | Purpose |
|------|---------|
| `_check_context_pressure` | Warns when context usage is high, triggers force-compact |
| `_check_reading_quality` | Warns about shallow file reads |
| `_check_error_escalation` | Escalates repeated failures to structured diagnosis |
| `_check_source_reading_gate` | Reminds to read framework source before fixing (after 2+ failures) |
| `_check_analysis_persistence` | Reminds to persist analysis results |
| `_check_plan_maintenance_gate` | Warns when plan steps are stale (8+ turns without update) |
| `_check_config_validation_hint` | Suggests `validate_config` after writing YAML to conf/ |
| `_check_diagnostic_print_hint` | Suggests diagnostic prints when writing forward/init code |

## Key Behaviors

### Auto-Release Mechanisms
- **Monitor gate**: Auto-clears after 5 consecutive blocks (prevents permanent stuck state)
- **Pipeline comprehension**: Phase 3 auto-passes after 3 blocks (prevents marker deadlock)
- **Training hang**: Scoring-based (needs ≥2 independent signals), not keyword-only

### Error Escalation (Layered Diagnosis)
After 2+ failures, forces this order:
1. Environment check (Python path, CUDA, packages)
2. Dependency verification (versions, compatibility)
3. Source code reading (framework internals)
4. Code fix (only after understanding root cause)

### Context Pressure Management
- Soft warning at 80% context usage
- Force-compact trigger at 90% with target 50%
- Warning count limited to 3 per session (prevents token waste)

## Integration

Gates run in `_run_pre_execution_gates(tool_name, arguments)` called from the main agent loop after parsing tool calls but before execution. Hard gates short-circuit (first block wins). All soft gates run and their messages are concatenated.

### Gate Override Mechanism

Hard gates can be overridden by the LLM when it is certain the gate does not apply to its situation. This prevents gates from being too rigid for edge cases (e.g., custom models that don't follow standard Megatron patterns).

**How it works:**
1. Gate blocks a tool call and includes the override key in its message (e.g., `PIPELINE_COMPREHENSION`)
2. LLM includes in its response: `[GATE_OVERRIDE: PIPELINE_COMPREHENSION] Reason: <detailed justification>`
3. Agent parses the override declaration from the response text
4. On the NEXT tool call, if the same gate fires again, the override is consumed and the gate passes (one-shot)

**Design principles:**
- Override is one-shot: it passes the gate exactly once, then expires
- Override requires a clear reason — the LLM must explain WHY the gate doesn't apply
- Override key is derived from gate method name: `_check_xxx_gate` → `XXX`
- The override hint is shown after 1st block (subtle), becomes more prominent at 2nd and 3rd blocks
- This prevents the LLM from being permanently stuck when a gate's heuristic doesn't match the actual situation

### Data→Model Interface Gate

Added to prevent the #1 cause of porting rework: writing model.forward() without knowing what the data pipeline produces.

- Triggers when writing model code (files matching model/forward/backbone/head patterns)
- Checks if data→model interface is documented in session memory
- Passes if the code itself shows awareness of real data keys (input_ids, attention_mask, etc.)
- Override key: `DATA_MODEL_INTERFACE`

### No Dummy Data Gate

Strictly forbids using torch.rand/zeros/ones as model input during porting verification.

- Triggers when writing/running code that invokes model forward with synthetic tensors
- Does NOT trigger for model definition files (class with def forward)
- Override key: `NO_DUMMY_DATA`
