# FlagScale Agent Documentation

## Architecture & Design
- [01 — Architecture Overview](01_architecture.md): Core loop, mixin composition, data flow, file layout.

## Core Mechanisms
- [02 — Gate System](02_gates.md): Pre-execution safety gates, hard blocks, soft warnings, auto-release.
- [03 — History & Compaction](03_history_compaction.md): Context window management, scorer, memory dump.
- [04 — Skill Lifecycle](04_skill_lifecycle.md): 6-phase ordering, error tracking, semantic dedup.
- [05 — Loop Detection](05_loop_detection.md): Duplicate tool-call detection, caching, FileCache.
- [06 — Memory](06_memory.md): SessionMemory, ExperimentManager, proactive recall.
- [07 — Plan Management](07_plan.md): Auto-sync, consistency checks, rebuild suggestions.
- [08 — Prompt Construction](08_prompt.md): System prompt layers, situational sections, turn context.
- [09 — Tool System](09_tools.md): Registry, parallel execution, specialized tools.
- [10 — Skill System](10_skills.md): Skill discovery, loading, domain knowledge packages.
- [11 — Monitor & Polling](11_monitor_polling.md): Long-running watch, process death detection, stderr scanning.
