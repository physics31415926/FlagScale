# Loop Detection & Tool-Call Caching

## Overview

Prevents the agent from repeating identical tool calls in a loop and caches read results to avoid redundant disk I/O.

## Loop Detection

### Mechanism
- Maintains a sliding window of recent tool call keys (`_recent_tool_calls`, size=10)
- Each tool call is converted to a canonical key via `_get_tool_call_key()`
- If the same key appears ≥3 times in the window, the 4th attempt triggers a `LOOP DETECTION` warning

### Key Generation

| Tool | Key Format |
|------|-----------|
| `shell` | `("shell", command_string)` |
| `read_file` | `("read_file", path, str(start_line), str(end_line))` |
| `write_file` | `("write_file", path, content)` |
| `edit_file` | `("edit_file", file_path, old_string, new_string)` |
| `load_skill` | `("load_skill", name)` |
| `memory_write` | `("memory_write", key)` |
| Other | `(tool_name, "key1=val1;key2=val2;...")` |

### Window Eviction
- Window size is 10 entries (FIFO)
- Old entries are evicted as new ones arrive
- This means a tool call repeated after 10 different calls won't trigger detection

## Tool-Call Caching

### Cacheable Tools
- `read_file`: Cached by `(path, start_line, end_line)`
- `memory_write`: Cached by `(key)`

### Cache Behavior
- `_check_duplicate_read()`: Returns cached result if key exists in `_tool_call_cache`
- `_cache_tool_result()`: Stores result after successful execution
- **Not cached on error**: Results containing "ERROR" are never cached
- **Not cached for empty paths/keys**: Prevents garbage entries

### Cache Invalidation
- `write_file` execution: Invalidates cache for the written path
- `edit_file` execution: Invalidates cache for the edited path
- No TTL-based expiration (session-scoped)

## File-Level Cache (ReadFileTool)

Separate from the tool-call cache, `ReadFileTool` has its own `FileCache`:
- **TTL**: 30 seconds
- **mtime validation**: Cache entry invalidated if file mtime changes
- **LRU eviction**: Maximum 50 entries
- **Explicit invalidation**: Called by `WriteFileTool` and `EditFileTool` after modifications

## Constants

```python
_LOOP_DETECTION_WINDOW = 10
_LOOP_DETECTION_THRESHOLD = 3
```
