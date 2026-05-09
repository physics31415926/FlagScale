# Prompt Construction

## Overview

`PromptMixin` builds the system prompt dynamically based on current context, active skills, and situational needs.

## Prompt Layers

```
SYSTEM_PROMPT_CORE          (always present — identity, capabilities, rules)
    + SYSTEM_PROMPT_OPTIONAL (tool descriptions, skill context)
    + Situational Sections   (context-dependent additions)
    = Final system prompt
```

## Situational Sections

Sections are conditionally included based on the agent's current state:

| Section Key | Trigger | Content |
|-------------|---------|---------|
| `training` | Active training task | Training workflow guidance, log analysis priority |
| `porting` | Model porting skill loaded | Porting methodology, diagnostic print strategy |
| `decision` | Always (when skills active) | Decision-making rules, layered error recovery |
| `config_schema` | Training context active | FlagScale YAML config structure reference |

### Training Context
- Log analysis priority: OOM/CUDA > NCCL timeout > loss anomaly > slow iteration > warnings
- Tool recommendations per phase
- Monitoring best practices

### Porting Context
- Model structure enumeration requirement
- Diagnostic print strategy (shape/dtype at module boundaries)
- Phase ordering reminder

### Config Schema
- Two-level Hydra config structure (top-level + model-level)
- Valid key locations
- Common misplacement errors

## Memory Context

`_build_memory_context()`:
- Injects relevant memory entries into system prompt
- When `_consecutive_train_failures >= 1`: queries error-related memories
- Marks injected entries as `[RELEVANT:key]`

## Turn Context

`_build_turn_context()`:
- Injects ExperimentManager's failed attempts (last 5)
- Format: `FAILED attempts (DO NOT REPEAT):`
- Ensures LLM sees history of what didn't work before generating new approach

## Tool Phase Detection

`_detect_tool_phase(tool_name, arguments)`:
- Classifies current operation into phases for gate system
- Used by phase ordering gate to enforce sequence
- Based on tool name + argument patterns (not keyword matching on output)
