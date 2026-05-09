# History & Compaction

## Overview

The history system manages the conversation context window — tracking token usage, triggering compaction when budget is exceeded, and preserving critical information through compression.

## HistoryManager

### Token Budget
- Tracks per-message token counts (estimated via tiktoken)
- Maintains running total against `max_context_tokens`
- Triggers compaction when usage exceeds threshold

### Compaction Triggers
- **Intra-turn**: Threshold 0.70 of max context, target 0.50
- **Force compact**: Threshold 0.90, target 0.50
- **Ratios**: `[0.60, 0.50, 0.40, 0.35]` — progressively more aggressive

### Summary Management
- Summary upper limit: `max_context_tokens * 0.05` (e.g., 10K tokens for 200K context)
- Old summary sections are fused (merged) rather than discarded
- Prevents summary from growing unboundedly across many compactions

### Inflation Ratio
- Tracks how much compacted messages expand when re-summarized
- Uses EMA: `0.7 * old + 0.3 * new`
- Anomalous values (>3.0) are discarded to prevent runaway estimates

### Hard Ceiling
- When messages exceed budget even after compaction, truncates by heuristic score:
  - High-score messages: 1200 character budget
  - Medium-score: 600 characters
  - Low-score: 200 characters

## CompactionMixin

### Scorer
- Rates each message's importance (0.0–1.0) for retention decisions
- Tolerance: if scorer returns ±10% different count from messages, truncates/pads to align
- Previously: any mismatch caused full fallback (lost all scoring)

### Pre-Compaction Memory Dump
- Before discarding messages, scans them for:
  - Key decisions and findings
  - Attempted fixes (extracted and stored)
  - Error patterns and their resolutions
- Writes extracted info to `SessionMemory` and `ExperimentManager.learnings`
- Compaction checkpoints use `priority="critical"` (7-day TTL, auto-cleanup)

### Compaction Anchors
- Preserves plan state (current step title + status) across compaction
- Preserves skill context and phase information
- Uses `get('title')` field (bug fix from `get('text')`)

## Data Flow

```
Message added to history
    → Token count updated
    → Check: total > intra_turn_threshold?
        Yes → Score messages → Discard lowest-scored
             → Memory dump extracted info
             → Rebuild summary
    → Check: total > force_compact_threshold?
        Yes → Aggressive compaction (ratio 0.35)
             → Inject warning if failed attempts found
```

## Key Design Decisions

1. **Aggressive first compression**: 0.60 ratio releases 40% immediately (was 0.80)
2. **Never lose fix history**: Attempted fixes are synced to ExperimentManager before discard
3. **Scorer fault tolerance**: Partial scoring is better than no scoring
4. **Summary fusion**: Old summaries merge into new ones rather than being dropped
