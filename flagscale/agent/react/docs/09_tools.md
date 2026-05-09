# Tool System

## Overview

The tool system provides the agent with concrete actions: file I/O, shell execution, monitoring, validation, and memory operations. Tools are registered in a `ToolRegistry` and exposed to the LLM via OpenAI/Anthropic function-calling schemas.

## Tool Registry

All tools inherit from `Tool` base class and implement:
- `name`: Tool identifier
- `description`: LLM-facing description
- `parameters`: JSON Schema for arguments
- `execute(**kwargs) -> str`: Execution logic

### Registered Tools (18)

| Tool | Category | Purpose |
|------|----------|---------|
| `shell` | Execution | Run shell commands |
| `read_file` | File I/O | Read file contents (with FileCache) |
| `write_file` | File I/O | Write file (invalidates cache) |
| `edit_file` | File I/O | Surgical text replacement (invalidates cache) |
| `find_latest_log` | Diagnostics | Find and filter log files (errors/progress/all) |
| `monitor` | Diagnostics | Long-running file/process watch without LLM calls |
| `parse_training_metrics` | Diagnostics | Extract loss/throughput/MFU from logs |
| `inspect_checkpoint` | Validation | Deep checkpoint verification (shape/dtype/anomalies) |
| `validate_config` | Validation | YAML config structure validation |
| `web_fetch` | Research | Fetch and extract web page content |
| `memory_write` | Memory | Store key-value pair |
| `memory_read` | Memory | Retrieve stored value |
| `memory_list` | Memory | List all memory keys |
| `workspace_experiment` | Experiments | Manage experiment records |
| `plan_create` | Planning | Create task plan |
| `plan_update` | Planning | Update plan step status |
| `plan_status` | Planning | Show plan state |
| `load_skill` | Skills | Load a skill definition |

## Parallel Execution

Tools are executed in parallel via `ThreadPoolExecutor` when:
- Multiple tool calls are returned in a single LLM response
- No dependencies between the calls (determined by the LLM)

## FileCache (read_file)

- **TTL**: 30 seconds
- **mtime validation**: Invalidates if file modified externally
- **LRU eviction**: Max 50 entries
- **Write-through invalidation**: `write_file` and `edit_file` explicitly invalidate

## Monitor Tool

Special long-running tool that polls without LLM calls:
- Watches file or command output
- Returns on: error pattern, success pattern, target step, process death, timeout
- Process death detection: checks `pgrep` for training processes
- Auto-scans stderr.log files in FlagScale output directories
- Formats results with clear headers: `TRAINING CRASHED`, `TRAINING DEAD`, etc.

## Inspect Checkpoint Tool

Deep checkpoint verification:
- Loads .pt/.bin/.safetensors files
- Reports: key count, shapes, dtypes, parameter count
- Detects anomalies: NaN, Inf, all-zero tensors
- Sample statistics: mean, std, min, max for sampled tensors
- Expected key matching (regex patterns)
- Reference comparison: shape/dtype diff against a known-good checkpoint

## Validate Config Tool

FlagScale YAML config validation:
- Auto-detects config type (top-level vs model-level)
- Checks required keys and valid structure
- Detects common misplacements (bf16 under model, tp under model, etc.)
- Returns: OK, WARNINGS, or ERRORS with specific guidance
