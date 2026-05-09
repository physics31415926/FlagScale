# Plan Management

## Overview

`TaskPlan` provides step-based planning with automatic state synchronization, consistency checking, and rebuild suggestions.

## Data Model

```python
Step:
    title: str
    status: "todo" | "doing" | "done" | "failed" | "skipped"
    notes: str
    started_turn: int
    last_active_turn: int
    failure_count: int
```

## Core Features

### Auto-Sync (`auto_sync_step`)
Called after every tool execution:
- `write_file` success → marks relevant step as `done`
- Tool failure → increments `failure_count`, adds error to `notes`
- Keeps plan state aligned with actual execution without requiring LLM to explicitly update

### Consistency Check (`check_consistency`)
Runs every 5 turns:
- **Stale detection**: Steps in `doing` status for >10 turns without activity → flagged
- **Repeated failure**: Steps with `failure_count ≥ 3` → flagged for approach change
- Returns list of issues for the agent to address

### Rebuild Suggestion (`should_rebuild`)
Triggers when:
- 3+ consecutive step failures
- Overall plan progress stalled (no step completed in 15+ turns)
- Returns recommendation to create a new plan with different approach

## Plan Maintenance Gate

`_check_plan_maintenance_gate` (soft warning):
- Fires when a `doing` step hasn't been updated in 8+ turns
- Injects reminder: "Step X has been in progress for N turns — update or mark done/failed"
- Prevents plans from silently drifting from reality

## Plan Tools

| Tool | Purpose |
|------|---------|
| `plan_create` | Create a new plan with steps |
| `plan_update` | Update step status, add notes |
| `plan_status` | Show current plan state |

## Compaction Integration

Plan state is preserved as a compaction anchor:
- Current step title and status survive context compression
- Uses `get('title')` field (previously bugged as `get('text')`)
- Ensures agent knows where it left off after compaction
