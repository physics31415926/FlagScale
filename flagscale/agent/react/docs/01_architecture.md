# FlagScale Agent — Architecture Overview

## 1. What It Is

FlagScale Agent is a **ReAct (Reasoning + Acting) agent** designed for autonomous model porting and distributed training tasks. It wraps an LLM (Claude/GPT) in a structured loop that alternates between reasoning about the current state and executing tool calls, with multiple safety and efficiency mechanisms layered on top.

## 2. Core Loop

```
User Task
    │
    ▼
┌─────────────────────────────────────────────┐
│              ReactAgent.run()                │
│                                             │
│  ┌─────────┐    ┌──────────┐    ┌────────┐ │
│  │ Prompt  │───▶│ LLM Call │───▶│ Parse  │ │
│  │ Build   │    │(streaming)│    │Response│ │
│  └─────────┘    └──────────┘    └────────┘ │
│       ▲                              │      │
│       │         ┌──────────┐         │      │
│       │         │  Gates   │◀────────┘      │
│       │         │(pre-exec)│                │
│       │         └────┬─────┘                │
│       │              │                      │
│       │         ┌────▼─────┐                │
│       │         │Tool Exec │                │
│       │         │(parallel)│                │
│       │         └────┬─────┘                │
│       │              │                      │
│       └──────────────┘                      │
│         (append result to history)          │
└─────────────────────────────────────────────┘
```

## 3. Mixin Composition

The `ReactAgent` class is composed via multiple inheritance:

| Mixin | Responsibility |
|-------|---------------|
| `PromptMixin` | System prompt construction, situational context |
| `CompactionMixin` | Context window management, message compression |
| `GatesMixin` | Pre-execution safety gates (hard blocks + soft warnings) |
| `JudgesMixin` | LLM-as-judge evaluations (skill matching, quality) |
| `CommandsMixin` | User slash-commands (/plan, /memory, /status) |
| `LoopDetectMixin` | Duplicate tool-call detection and caching |
| `CheckpointMixin` | Proactive memory recall on errors |
| `PollMixin` | Non-LLM polling for long-running operations |
| `SkillLifecycleMixin` | Skill loading, phase transitions, error escalation |

## 4. Key Subsystems

### 4.1 History & Compaction
- `HistoryManager`: Token counting, message storage, compaction triggers
- `CompactionMixin`: Multi-ratio compression with scorer, memory dump before discard

### 4.2 Gate System
- 31 gate methods (hard blocks + soft warnings)
- Pre-execution: prevent dangerous/premature actions
- Post-execution: validate results before continuing

### 4.3 Skill Lifecycle
- 6-phase strict ordering: analysis → structure → verification → data → checkpoint → training
- Error pattern tracking with fix fingerprinting
- LLM-based semantic deduplication of repeated fixes

### 4.4 Memory & Experiments
- `SessionMemory`: Key-value store with TTL, priority, and relevance queries
- `ExperimentManager`: Structured experiment/attempt tracking with learnings

### 4.5 Plan Management
- `TaskPlan`: Step-based planning with auto-sync, consistency checks, rebuild suggestions

### 4.6 Tool System
- 18 registered tools with OpenAI/Anthropic schema generation
- Parallel execution via ThreadPoolExecutor
- File cache (TTL + mtime) for read deduplication

## 5. Data Flow

```
User message
    → HistoryManager (token budget check)
    → PromptMixin (build system prompt + situational context)
    → LLM provider (streaming response)
    → Parse tool calls from response
    → LoopDetectMixin (dedup check)
    → GatesMixin (pre-execution gates)
    → Tool execution (parallel when independent)
    → GatesMixin (post-execution validation)
    → SkillLifecycleMixin (phase tracking, error escalation)
    → Append results to history
    → CompactionMixin (if over budget, compress)
    → Next iteration or return to user
```

## 6. Configuration

`AgentConfig` (dataclass) loaded from YAML:
- `provider`: LLM backend (anthropic/openai)
- `model`: Model identifier
- `max_context_tokens`: Context window budget
- `max_output_tokens`: Per-response token limit
- `temperature`: Sampling temperature
- `tools_enabled`: List of active tools
- `skills_dir`: Path to skill definitions

## 7. File Layout

```
flagscale/agent/react/
├── agent.py              # ReactAgent class, main loop
├── gates.py              # GatesMixin — 31 gate methods
├── history.py            # HistoryManager — token tracking, compaction triggers
├── compact.py            # CompactionMixin — scorer, memory dump
├── prompt.py             # PromptMixin — system prompt, situational sections
├── memory.py             # SessionMemory — TTL key-value store
├── plan.py               # TaskPlan — step management
├── skill_lifecycle.py    # SkillLifecycleMixin — phases, error escalation
├── loop_detect.py        # LoopDetectMixin — dedup, caching
├── checkpoint.py         # CheckpointMixin — proactive memory recall
├── experiment_manager.py # ExperimentManager — attempt tracking
├── judges.py             # JudgesMixin — LLM evaluations
├── commands.py           # CommandsMixin — slash commands
├── poll.py               # PollMixin — non-LLM polling
├── config.py             # AgentConfig dataclass
├── providers/            # LLM provider adapters
├── tools/                # Tool implementations
├── skills/               # Skill definitions (SKILL.md files)
├── tests/                # Test suite (610 tests)
└── docs/                 # This documentation
```
