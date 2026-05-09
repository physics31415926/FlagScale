# Memory System

## Overview

Two complementary memory systems provide cross-session and within-session persistence:

1. **SessionMemory**: Key-value store with TTL, priority, and relevance search
2. **ExperimentManager**: Structured experiment/attempt tracking

## SessionMemory

### Storage
- YAML files on disk, one per key
- Fields: `key`, `value`, `type` (finding/decision/todo/context), `priority`, `created`, `accessed`, `access_count`

### Priority & TTL

| Priority | TTL | Use Case |
|----------|-----|----------|
| `high` | Never expires | Critical findings, permanent decisions |
| `critical` | 7 days | Compaction checkpoints (auto-cleanup) |
| `normal` | 30 days | Standard findings and context |
| `low` | 7 days | Ephemeral context |

### Auto-Promotion
- Entries accessed ≥3 times are auto-promoted from `normal` → `high`
- Prevents important but initially low-priority items from expiring

### Relevance Query
`query_relevant(keywords, max_tokens)`:
- Scores each entry by keyword match against key + content
- Returns top matches within token budget
- Used for proactive memory recall on errors

### Deduplication
- Confidence threshold: 0.7
- Before writing, checks if semantically similar entry exists
- Prevents memory bloat from repeated similar findings

## ExperimentManager

### Schema
```yaml
Experiment:
  name: str
  purpose: str
  hypothesis: str
  status: running | completed | failed
  created: timestamp
  attempts: List[Attempt]
  root_cause: str
  learnings: List[str]

Attempt:
  timestamp: str
  change: str
  hardware: {gpus, gpu_type, driver, cuda}
  config: {model, tp, dp, pp, global_batch_size, ...}
  output_dir: str
  result: str
```

### Integration
- `compact.py`: Syncs `attempted_fixes` to `learnings` during compaction
- `agent.py`: Injects recent failed attempts (last 5) into turn context
- `checkpoint.py`: Proactive recall queries experiment learnings on error

## Proactive Memory Recall (CheckpointMixin)

`_proactive_memory_recall(error_text)`:
1. Extracts keywords from error (error type, module name, config params)
2. Queries SessionMemory for relevant entries
3. Injects matched memories into tool result context
4. Triggered automatically on training failures and shell errors

### Keyword Extraction
`_extract_recall_keywords(text)`:
- Error patterns: OOM, NCCL, CUDA, shape mismatch
- Module names: Layer, Module, Attention, MLP
- Config keywords: tp, pp, dp, seq_length, batch_size

## Memory Tools

| Tool | Purpose |
|------|---------|
| `memory_write` | Store a key-value pair with optional priority/type |
| `memory_read` | Retrieve a specific key |
| `memory_list` | List all keys with metadata |
