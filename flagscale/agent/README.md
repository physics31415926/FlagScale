# FlagScale Agent

AI-powered infrastructure engineer for large model training with the FlagScale framework.

## Quick Start

```bash
# Install
pip install -e ".[agent]"

# Run
flagscale-agent

# Or with specific provider/model
flagscale-agent --provider anthropic --model claude-sonnet-4-20250514
```

## What It Does

FlagScale Agent is a ReAct-based AI agent that helps you set up, configure, launch, monitor, and debug distributed training jobs. It has deep knowledge of FlagScale's architecture, Megatron-LM, and GPU cluster operations.

## Skills

The agent has 9 specialized skills that are automatically loaded based on your request:

| Skill | Description |
|-------|-------------|
| `env-setup` | Install conda env, FlagScale, and all FL-customized dependencies |
| `topo-detect` | Detect hardware topology — NVLink, NUMA, RDMA, storage |
| `model-porter` | Analyze source models, match architecture, convert checkpoints |
| `data-prep` | Prepare training data in Megatron binary format |
| `train-config` | Generate FlagScale training YAML configuration |
| `train-run` | Launch, stop, and manage distributed training jobs |
| `train-monitor` | Monitor training health, detect anomalies, parse metrics |
| `reproduce` | Reproduce training results from reference implementations |
| `precision-alignment` | Verify numerical alignment between implementations |

## Commands

| Command | Description |
|---------|-------------|
| `/skill <name>` | Manually load a skill |
| `/file <path>` | Add a file to conversation context |
| `/plan` | Show current task plan |
| `/memory list` | Show all memory entries |
| `/cache` | Show cached project knowledge |
| `/mode auto` | Execute all commands without confirmation |
| `/mode confirm` | Require confirmation for risky commands (default) |
| `/save [path]` | Save conversation |
| `/load <path>` | Load a saved conversation |
| `/reload` | Reload skills and config |
| `/quit` | Exit |

## Configuration

Create `~/.flagscale/agent.yaml`:

```yaml
provider: anthropic
model: claude-sonnet-4-20250514
api_key: ${ANTHROPIC_API_KEY}
mode: confirm
max_iterations: 50
```

Or use environment variables:
```bash
export ANTHROPIC_API_KEY=sk-...
export FLAGSCALE_AGENT_PROVIDER=anthropic
export FLAGSCALE_AGENT_MODEL=claude-sonnet-4-20250514
```

## Example: Port a Model to FlagScale

```
You: port Qwen3-0.6B to FlagScale

Agent: [loads model-porter skill]
       Analyzing Qwen3-0.6B architecture...
       - 28 layers, hidden_size=896, 14 attention heads, 4 KV heads (GQA)
       - RoPE positional encoding, SwiGLU activation, RMSNorm
       All components supported in Megatron-LM-FL. Mode A: direct config generation.
       
       Generating checkpoint conversion code...
       [creates ckpt.py, runs forward alignment]
       
       Forward alignment passed: max diff = 1.2e-6 (threshold: 1e-3)
       Model ported successfully.

You: generate training config for 8 GPUs

Agent: [loads train-config skill]
       [generates train.yaml with TP=1, PP=1, DP=8]

You: start training

Agent: [loads train-run skill]
       [runs preflight checks, launches training]
       Training started. Use "check training status" to monitor.
```

## Architecture

```
flagscale/agent/
├── __init__.py              # Version
├── react/
│   ├── agent.py             # Core ReAct loop + system prompt
│   ├── config.py            # Configuration
│   ├── skills/              # Skill manager
│   │   └── __init__.py
│   ├── tools/               # Tool implementations
│   │   ├── shell.py
│   │   ├── read_file.py
│   │   ├── write_file.py
│   │   ├── edit_file.py
│   │   ├── web_fetch.py
│   │   ├── load_skill.py
│   │   ├── memory_read.py
│   │   ├── memory_write.py
│   │   ├── cache_read.py
│   │   ├── cache_write.py
│   │   ├── find_log.py
│   │   ├── plan_create.py
│   │   ├── plan_update.py
│   │   └── plan_status.py
│   ├── memory.py            # Session memory
│   ├── cache.py             # Knowledge cache
│   ├── plan.py              # Task planning
│   ├── display.py           # Terminal UI
│   ├── history.py           # Conversation history
│   ├── providers.py         # LLM provider abstraction
│   ├── retry.py             # Retry with backoff
│   ├── cost.py              # Cost tracking
│   └── tests/               # Test suite
├── skills/                  # Skill definitions (SKILL.md files)
│   ├── env-setup/
│   ├── topo-detect/
│   ├── model-porter/
│   ├── data-prep/
│   ├── train-config/
│   ├── train-run/
│   ├── train-monitor/
│   ├── reproduce/
│   └── precision-alignment/
```

## Development

```bash
# Run tests
pytest flagscale/agent/react/tests/ -v

# Run specific test file
pytest flagscale/agent/react/tests/test_skill_validation.py -v
pytest flagscale/agent/react/tests/test_workflow_integration.py -v
```
