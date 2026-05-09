# Skill Lifecycle

## Overview

The skill lifecycle system manages how the agent progresses through complex multi-phase tasks (especially model porting). It enforces strict phase ordering, tracks error patterns, and prevents repeated failed approaches.

## 6-Phase Strict Ordering

```
analysis → structure_implementation → structure_verification
    → data_pipeline → checkpoint_conversion → training_verification
```

Each phase has:
- **Entry conditions**: What must be true before entering
- **Allowed operations**: What tools/actions are permitted
- **Exit conditions**: What must be achieved to advance

The `_check_phase_ordering_gate` hard-blocks operations attempted in the wrong phase (e.g., checkpoint conversion before structure verification).

## Error Pattern Tracking

### Structured History
`_error_pattern_history` stores `(pattern, fix_fingerprint)` tuples:
- `pattern`: The error signature (e.g., "shape mismatch in layer3.weight")
- `fix_fingerprint`: Description of what was tried (extracted from recent tool calls)

### Fix Fingerprinting
`_extract_fix_fingerprint()` examines recent tool calls to build a description of the attempted fix:
- File edits: captures the file path and nature of change
- Shell commands: captures the command
- Config changes: captures the parameter modified

### Semantic Deduplication
`_llm_judge_is_same_fix()` uses the LLM to determine if a proposed fix is semantically equivalent to a previously failed one. This catches cases where the same approach is tried with minor syntactic differences.

## Error Escalation

`_record_and_escalate_failure()`:
1. Records the error pattern + fix fingerprint
2. Builds `failed_attempts_summary` from all prior attempts for this pattern
3. Injects the summary into escalation messages
4. Resets `_source_reads_since_last_failure` counter
5. Emphasizes "read framework source code" as first step

### Escalation Levels
- **1st failure**: Normal retry allowed
- **2nd failure**: Soft warning — must read framework source
- **3rd+ failure**: Hard escalation — forced diagnosis order (env → deps → source → fix)

## Phase Transition Guidance

`_get_current_phase_guidance()` provides:
- What operations are forbidden in the current phase
- What must be completed before advancing
- Suggested next actions

## Integration Points

- `agent.py`: Injects failed attempts into turn context (`_build_turn_context`)
- `compact.py`: Syncs attempted_fixes to ExperimentManager during compaction
- `gates.py`: Phase ordering gate, source reading gate, error escalation gate
- `prompt.py`: Decision section includes layered recovery order
