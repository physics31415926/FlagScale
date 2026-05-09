# FlagScale Agent User Guide

## Introduction

FlagScale Agent is an AI assistant for large model training infrastructure. It reads/writes files, executes commands, browses documentation, and helps you with environment setup, config tuning, model porting, and fault diagnosis.

The agent uses a ReAct (Reasoning + Acting) loop: think, call tools, observe results, repeat until the task is done.

### v6 Highlights

- **6-phase strict ordering** for model porting (analysis → structure → verification → data → checkpoint → training)
- **Gate system** with 31 safety gates (hard blocks + soft warnings) preventing premature or dangerous actions
- **Loop detection** preventing repeated identical tool calls
- **Deep checkpoint verification** (shape/dtype/NaN/Inf/all-zero anomaly detection)
- **Proactive memory recall** on errors — automatically retrieves relevant past findings
- **Config validation tool** catching YAML misplacements before training launch
- **Process death detection** in monitor tool — immediate feedback when training crashes
- **Semantic deduplication** of failed fix attempts via LLM judge

## Quick Start

### 1. Configure API Key

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI
export OPENAI_API_KEY="sk-..."
```

### 2. Launch Agent

```bash
# Interactive mode (default: Anthropic + claude-sonnet)
flagscale agent

# Specify provider and model
flagscale agent --provider openai --model gpt-4o

# Custom API endpoint (proxy/gateway)
flagscale agent --base-url https://my-proxy.example.com/v1

# Single-shot query (non-interactive)
flagscale agent "What dependencies does FlagScale training need?"
```

CLI arguments:

| Argument | Short | Description |
|----------|-------|-------------|
| `--provider` | `-p` | LLM backend: `anthropic` (default) or `openai` |
| `--model` | `-m` | Model name |
| `--base-url` | `-b` | API endpoint for proxy or self-hosted gateway |
| `--config` | `-c` | Config file path |

Python API:

```python
from flagscale.agent.react import AgentConfig, ReactAgent

config = AgentConfig.auto_load(provider="anthropic", model="claude-sonnet-4-20250514")
agent = ReactAgent(config)

# Interactive REPL
agent.run()

# Single-shot query
agent.run(single_shot_query="Check training environment dependencies")
```

### 3. Start a Conversation

```
╭─ FlagScale Agent ─────────────────────────────╮
│  Provider: anthropic | Model: claude-sonnet    │
│  Commands: /skill  /file  /save  /load  ...    │
╰────────────────────────────────────────────────╯

>
```

Type your question. The agent will reason, call tools, and respond.

## Configuration

### Config File

Agent searches for config in this order:

1. `FLAGSCALE_AGENT_CONFIG` environment variable
2. `.flagscale/agent.yaml` in current directory
3. `~/.flagscale/agent.yaml`

Example:

```yaml
# ~/.flagscale/agent.yaml

provider: anthropic
model: claude-sonnet-4-20250514

# Behavior
max_iterations: 200
max_context_tokens: 100000
shell_timeout: 120
dangerous_commands_check: true
confirm_commands: true

# Budget
max_cost: 5.0
pricing:
  my-custom-model:
    input: 3.0
    output: 15.0

# Memory
memory_ttl_days: 7
cache_ttl_days: 7

# Network proxy
shell_env:
  HTTP_PROXY: "http://proxy.example.com:8080"
  HTTPS_PROXY: "http://proxy.example.com:8080"

# Custom skill/tool directories
skill_dirs:
  - /path/to/custom/skills
plugin_tool_dirs:
  - /path/to/plugin/tools
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `ANTHROPIC_AUTH_TOKEN` | Anthropic Auth Token (higher priority) |
| `ANTHROPIC_BASE_URL` | Custom Anthropic API endpoint |
| `OPENAI_API_KEY` | OpenAI API Key |
| `OPENAI_BASE_URL` | Custom OpenAI API endpoint |
| `FLAGSCALE_AGENT_CONFIG` | Config file path |
| `HTTP_PROXY` / `HTTPS_PROXY` | Network proxy |

### Hot Reload

```
> /reload
Config and skills reloaded.
```

## Commands

Commands start with `/` and are not sent to the LLM:

| Command | Description |
|---------|-------------|
| `/quit` | Exit agent |
| `/reload` | Reload config and skills |
| `/skill list` | List available skills |
| `/skill load <name>` | Load a skill manually |
| `/file <path>` | Inject file content into context |
| `/save [name]` | Save current session |
| `/load [name]` | Load a saved session |
| `/export [path]` | Export conversation as Markdown |
| `/memory list` | List all memory entries |
| `/memory delete <key>` | Delete a memory entry |
| `/memory clear` | Clear all memory |

## Tools (18)

The agent automatically selects and calls tools during conversation:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (with 30s TTL cache) |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical string replacement in files |
| `shell` | Execute shell commands |
| `web_fetch` | Fetch web page content |
| `find_latest_log` | Find and filter log files (errors/progress/all) |
| `monitor` | Long-running file/process watch without LLM calls |
| `parse_training_metrics` | Extract loss/throughput/MFU from logs |
| `inspect_checkpoint` | Deep checkpoint verification (shape/dtype/anomalies) |
| `validate_config` | YAML config structure validation |
| `memory_write` | Store findings, decisions, or todos |
| `memory_read` | Retrieve a memory entry |
| `memory_list` | List all memory keys |
| `workspace_experiment` | Manage experiment records |
| `plan_create` | Create a task plan with steps |
| `plan_update` | Update plan step status |
| `plan_status` | Show current plan state |
| `load_skill` | Load a skill definition |

### Shell Safety

Three layers of protection:

1. **Fatal command blocking**: `rm -rf /`, `mkfs`, `dd if=` are rejected outright
2. **Risky command confirmation**: `rm`, `kill`, `git push`, `pip install` require your approval
3. **Self-protection**: Commands that could kill the agent process are auto-rewritten

### Monitor Tool

For long-running operations (training, model loading):

```
> Monitor the training — stop when step 1000 is reached or if it crashes

⚡ monitor(output_dir="outputs/qwen3_0_6b/", target_step=1000, duration=600)
```

The monitor polls locally without LLM calls and returns when:
- An error/crash is detected (stderr scanning)
- Training reaches the target step
- The training process dies
- Timeout is reached

### Checkpoint Inspection

Deep verification beyond key-count checks:

```
> Verify the converted checkpoint before training

⚡ inspect_checkpoint(path="checkpoints/qwen3/mp_rank_00_model_states.pt",
                      reference_path="original/pytorch_model.bin")
```

Reports: shape mismatches, dtype differences, NaN/Inf/all-zero anomalies, missing keys.

### Config Validation

```
> Validate my training config

⚡ validate_config(path="examples/qwen3/conf/train/0_6b.yaml")
```

Detects: wrong nesting (bf16 under model instead of system.precision), missing required keys, unknown keys.

## Skills

Skills are domain knowledge packages loaded automatically or manually.

### Built-in Skills

| Skill | Purpose |
|-------|---------|
| `model-porter` | Model porting methodology (6-phase) |
| `data-prep` | Data pipeline with parallelism awareness |
| `train-config` | Training config generation |
| `train-run` | Training execution |
| `train-monitor` | Training monitoring |
| `env-setup` | Environment setup |
| `topo-detect` | Hardware topology detection |
| `reproduce` | Baseline reproduction |
| `precision-alignment` | Loss curve alignment |

### Auto-Loading

The agent matches your input to relevant skills via keywords and loads them automatically. You can also load manually:

```
> /skill load model-porter
```

### Custom Skills

Create a directory with `SKILL.md`:

```markdown
---
name: my-skill
description: Custom skill description
keywords: [keyword1, keyword2]
---

## Instructions

Your domain-specific guidance here...
```

Place it in `~/.flagscale/skills/my-skill/` or configure `skill_dirs` in your config.

## Memory System

### Session Memory

The agent records key findings during conversation and recalls them in future sessions:

```
> Aquila 70B OOM with TP=8

⚡ memory_write(key="aquila70b_tp_oom", type="finding",
               content="Aquila 70B: TP=8 OOM, switched to TP=4+PP=2")
```

Next session, the agent already knows this and won't repeat the failed approach.

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `finding` | Discovered facts | "TP=8 OOM, memory exceeds 80GB/GPU" |
| `decision` | Decisions made | "Final strategy: TP=4+PP=2+DP=4" |
| `todo` | Pending items | "Still need to test EP for MoE" |
| `context` | Background info | "User preparing CSA/HCA hybrid attention" |

### Priority & TTL

| Priority | TTL | Use |
|----------|-----|-----|
| `high` | Never expires | Critical findings |
| `critical` | 7 days | Compaction checkpoints |
| `normal` | 30 days | Standard findings |
| `low` | 7 days | Ephemeral context |

Entries accessed ≥3 times auto-promote from `normal` to `high`.

### Proactive Recall

On training failures, the agent automatically queries memory for relevant past findings and injects them into context — no manual recall needed.

## Experiment Tracking

The agent tracks experiments with structured records:

```
> Create experiment for Qwen3 porting

⚡ workspace_experiment(action="create", name="qwen3_port",
                        purpose="Port Qwen3-0.6B to FlagScale")
```

Each attempt records: config, hardware, output directory, and result. Failed attempts are preserved across context compaction and injected as "DO NOT REPEAT" guidance.

## Plan Management

For complex tasks, the agent creates step-based plans:

```
> Port Qwen3 model to FlagScale

⚡ plan_create(steps=["Enumerate model structure", "Implement attention",
                      "Implement MLP", "Data pipeline", "Checkpoint conversion",
                      "Training verification"])
```

Plans auto-sync with tool execution, detect stale steps, and suggest rebuilds after repeated failures.

## Session Management

### Auto-Save

The agent auto-saves after each turn. If interrupted, it offers to resume:

```
╭─ Unfinished session detected ──────────────────╮
│  Time: 2026-04-24 10:30:00                      │
│  5 turns, 3 user messages                        │
│  Last message: Check training config...          │
╰─────────────────────────────────────────────────╯
Resume previous session? [Y/n]:
```

### Manual Save/Load

```
> /save my-debug-session
✓ Session saved

> /load my-debug-session
✓ Session loaded (5 turns)
```

## Budget Control

```yaml
max_cost: 5.0  # USD
```

- Warning at 80% usage
- Auto-stop when exceeded
- Per-turn cost display: `── Turn 3 | 12.5s | ↑8,234 ↓1,456 tokens | $0.12 / $5.00 ──`

## Common Usage Examples

### Diagnose Training Failure

```
> Training OOM, help me figure out what's wrong
```

The agent loads the training debug skill, checks logs (with error filtering), GPU state, and config.

### Port a Model

```
> Port Qwen3-0.6B to FlagScale
```

The agent loads `model-porter` skill, enumerates model structure, implements components phase by phase, validates checkpoint, and verifies training.

### Validate and Fix Config

```
> Check if my training config is correct
```

The agent runs `validate_config`, identifies misplaced keys, and fixes them.

### Monitor Training

```
> Start training and monitor until step 500
```

The agent launches training, then uses the monitor tool to watch progress without consuming tokens.

## Directory Structure

```
~/.flagscale/
  agent.yaml              # User-level config
  agent_memory/           # Session memory (YAML files)
  sessions/               # Session storage
    autosave.json
    session_<id>.json
  skills/                 # User-level custom skills
  input_history           # Input history (arrow key recall)

<project>/
  .flagscale/
    agent.yaml            # Project-level config
    skills/               # Project-level skills
    tools/                # Project-level plugin tools
```
