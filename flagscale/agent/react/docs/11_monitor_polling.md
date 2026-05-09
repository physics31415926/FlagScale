# Monitor & Polling

## Overview

The monitor and polling systems handle long-running operations without consuming LLM tokens. They poll locally and only return to the LLM when something meaningful happens.

## MonitorTool

### Trigger Conditions (returns immediately on any)
1. **Error pattern**: Matches `_INTERESTING_RE` (ERROR, FATAL, Traceback, OOM, NCCL, etc.)
2. **Success pattern**: User-specified regex match
3. **Fail pattern**: User-specified regex match
4. **Target step**: Training step number reached
5. **Process death**: All training processes exited (`pgrep` check)
6. **Timeout**: Duration exceeded

### Process Death Detection
- Checks `pgrep -f <pattern>` for training processes
- Default pattern: `torchrun|python.*train|flagscale|megatron|deepspeed`
- Immediate check on first poll — if process already dead, returns immediately
- Result header: `TRAINING DEAD — all processes exited`

### Stderr Scanning
- For FlagScale output directories, auto-discovers `stderr.log` files
- Tracks file sizes to detect only NEW errors (not pre-existing content)
- Filters out known non-errors: wandb warnings, FutureWarning, DeprecationWarning
- Result header: `TRAINING CRASHED — fatal error detected in stderr`

### Output Format
```
TRAINING DEAD — all processes exited (3 polls, 15s)
Events:
  [process DEAD at start, no training running]
Last output before death:
  <last 20 lines of log>
```

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | str | Log file path to watch |
| `command` | str | Shell command to poll (alternative to file) |
| `output_dir` | str | FlagScale output dir (auto-scans stderr) |
| `duration` | int | Max seconds to poll (default: 300) |
| `interval` | int | Seconds between polls (default: 10) |
| `target_step` | int | Stop when this training step is reached |
| `success_pattern` | str | Regex — stop on match |
| `fail_pattern` | str | Regex — stop on match |
| `process_pattern` | str | Custom pgrep pattern for liveness check |

## PollMixin

Provides the agent-level polling interface:
- Wraps MonitorTool for use within the agent loop
- Manages display updates during polling (progress indicators)
- Integrates with gate system (monitor gate auto-release after 5 blocks)

## Integration with Gates

- `_check_training_hang`: Uses scoring (≥2 signals needed), not just keyword match
- Monitor gate auto-clears when:
  - 5 consecutive blocks (prevents permanent stuck)
  - Diagnostic commands (pgrep/ps) return empty (process gone)
