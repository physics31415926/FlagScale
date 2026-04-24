# FlagScale Agent 用户指南

## 简介

FlagScale Agent 是一个面向大模型训练基础设施的 AI 助手。它能够读写文件、执行命令、查阅文档，帮助你完成训练环境搭建、配置调优、故障排查等任务。

Agent 采用 ReAct（Reasoning + Acting）模式工作：先思考，再调用工具，根据结果继续推理，直到给出最终回答。

## 快速开始

### 1. 配置 API Key

Agent 支持 Anthropic 和 OpenAI 两种后端。通过环境变量配置：

```bash
# Anthropic（默认）
export ANTHROPIC_API_KEY="sk-ant-..."

# 或 OpenAI
export OPENAI_API_KEY="sk-..."
```

### 2. 启动 Agent

通过 FlagScale CLI 启动：

```bash
# 交互模式（默认 Anthropic + claude-sonnet）
flagscale agent

# 指定 provider 和模型
flagscale agent --provider openai --model gpt-4o

# 指定配置文件
flagscale agent --config ~/.flagscale/agent.yaml

# 自定义 API 地址（适配代理/网关）
flagscale agent --base-url https://my-proxy.example.com/v1

# 单次查询（非交互模式，自动关闭命令确认）
flagscale agent "FlagScale 训练环境有哪些依赖？"
```

CLI 参数说明：

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--provider` | `-p` | LLM 后端，`anthropic`（默认）或 `openai` |
| `--model` | `-m` | 模型名称，默认使用 provider 的默认模型 |
| `--base-url` | `-b` | API 地址，用于代理或自部署网关 |
| `--config` | `-c` | 配置文件路径 |

也可以通过 Python 代码启动：

```python
from flagscale.agent.react import run_agent

# 使用默认配置
run_agent()

# 指定 provider 和模型
run_agent(provider="openai", model="gpt-4o")
```

或者更精细地控制：

```python
from flagscale.agent.react import AgentConfig, ReactAgent

config = AgentConfig.auto_load(provider="anthropic", model="claude-sonnet-4-20250514")
agent = ReactAgent(config)

# 交互式 REPL
agent.run()

# 单次查询
agent.run(single_shot_query="FlagScale 训练环境有哪些依赖？")
```

### 3. 开始对话

启动后你会看到：

```
╭─ FlagScale Agent ─────────────────────────────╮
│  Provider: anthropic | Model: claude-sonnet    │
│  Commands: /skill  /file  /save  /load  ...    │
╰────────────────────────────────────────────────╯

>
```

直接输入问题即可。Agent 会自动思考、调用工具、给出回答。

## 配置

### 配置文件

Agent 按以下优先级查找配置文件：

1. 环境变量 `FLAGSCALE_AGENT_CONFIG` 指定的路径
2. 当前目录 `.flagscale/agent.yaml`
3. 用户目录 `~/.flagscale/agent.yaml`

配置文件示例：

```yaml
# ~/.flagscale/agent.yaml

# LLM 后端
provider: anthropic
model: claude-sonnet-4-20250514
# api_key: sk-ant-...        # 也可以用环境变量
# base_url: https://...      # 自定义 API 地址

# 行为控制
max_iterations: 50            # 单轮最大工具调用次数
max_context_tokens: 100000    # 上下文窗口大小
shell_timeout: 120            # Shell 命令超时（秒）
dangerous_commands_check: true # 拦截危险命令
confirm_commands: true         # 高风险命令需确认

# 费用控制
max_cost: 5.0                 # 预算上限（美元），0 表示不限
pricing:                      # 自定义模型定价（可选）
  my-custom-model:
    input: 3.0                # 每百万 input token
    output: 15.0              # 每百万 output token

# 知识缓存
cache_ttl_days: 7             # 缓存过期天数

# 会话记忆
memory_ttl_days: 7            # 记忆过期天数

# 网络代理
shell_env:
  HTTP_PROXY: "http://proxy.example.com:8080"
  HTTPS_PROXY: "http://proxy.example.com:8080"

# 技能目录（追加到内置目录之后）
skill_dirs:
  - /path/to/custom/skills

# 插件工具目录
plugin_tool_dirs:
  - /path/to/plugin/tools

# 会话存储目录
session_dir: ~/.flagscale/sessions
```

所有配置项都可以省略，Agent 会使用合理的默认值。

### 环境变量

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `ANTHROPIC_AUTH_TOKEN` | Anthropic Auth Token（优先级高于 API_KEY） |
| `ANTHROPIC_BASE_URL` | Anthropic API 自定义地址 |
| `ANTHROPIC_MODEL` | 覆盖默认模型 |
| `OPENAI_API_KEY` | OpenAI API Key |
| `OPENAI_BASE_URL` | OpenAI API 自定义地址 |
| `FLAGSCALE_AGENT_CONFIG` | 配置文件路径 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 网络代理（自动继承到 Shell 环境） |

### 热重载

修改配置文件后，无需重启 Agent，在对话中输入：

```
> /reload
Config and skills reloaded.
```

## 交互命令

在对话中以 `/` 开头的输入是命令，不会发送给 LLM：

| 命令 | 说明 |
|------|------|
| `/quit` | 退出 Agent |
| `/reload` | 重新加载配置文件和技能 |
| `/skill list` | 列出所有可用技能 |
| `/skill load <name>` | 手动加载一个技能 |
| `/file <path>` | 将文件内容注入到对话上下文 |
| `/save [name]` | 保存当前会话 |
| `/load [name\|path]` | 加载已保存的会话（无参数列出所有会话） |
| `/export [path]` | 导出对话为 Markdown 文件 |
| `/cache list` | 列出所有知识缓存条目 |
| `/cache delete <key>` | 删除指定缓存条目 |
| `/cache clear` | 清空所有缓存 |
| `/memory list` | 列出所有会话记忆条目 |
| `/memory delete <key>` | 删除指定记忆条目 |
| `/memory clear` | 清空所有记忆 |

## 内置工具

Agent 有 10 个内置工具，在对话中自动使用：

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 创建或覆盖文件 |
| `edit_file` | 精确替换文件中的字符串（支持全部替换） |
| `shell` | 执行 Shell 命令 |
| `web_fetch` | 抓取网页内容（支持代理） |
| `load_skill` | 加载技能指令 |
| `cache_write` | 缓存项目知识 |
| `cache_read` | 读取缓存的项目知识 |
| `memory_write` | 保存关键发现、决策或待办事项 |
| `memory_read` | 读取指定的记忆条目 |

### Shell 安全机制

Agent 执行 Shell 命令时有三层保护：

1. **致命命令拦截**：`rm -rf /`、`mkfs`、`dd if=` 等命令直接拒绝执行
2. **高风险命令确认**：`rm`、`kill`、`git push`、`pip install` 等命令需要你确认
3. **自杀保护**：如果命令可能杀死 Agent 自身进程，会自动改写命令排除 Agent 的 PID

```
> 帮我清理训练产生的临时文件

⚡ shell(command="rm -rf outputs/exp_20240101/tmp/")
⚠  Risky command: rm -rf outputs/exp_20240101/tmp/
   Allow? [y/N]: y
  ✓ shell (0.3s)
```

你可以在配置中关闭这些检查：

```yaml
dangerous_commands_check: false  # 关闭致命命令拦截
confirm_commands: false          # 关闭确认提示
```

## 技能系统

技能是预置的领域知识模块，让 Agent 在特定场景下获得专业指导。

### 技能文件格式

技能以 Markdown 文件形式存放，每个技能一个目录，包含 `SKILL.md`：

```
skills/
  train-debug/
    SKILL.md
  env-setup/
    SKILL.md
```

`SKILL.md` 格式：

```markdown
---
name: train-debug
description: 大模型训练故障排查
keywords: [OOM, hang, crash, 训练, 报错]
parameters:
  - name: model
    description: 模型名称
    default: aquila
---

## 排查步骤

1. 检查 {model} 的训练日志...
2. 查看 GPU 显存使用...
```

### 技能目录

Agent 按以下顺序扫描技能目录（后面的覆盖前面的同名技能）：

1. 内置目录：`flagscale/agent/skills/`
2. 项目目录：`<当前目录>/.flagscale/skills/`
3. 用户目录：`~/.flagscale/skills/`
4. 配置文件中 `skill_dirs` 指定的目录

### 自动加载

默认开启（`auto_skill: true`）。Agent 会根据你的输入自动匹配相关技能并加载。匹配基于技能的 `keywords`、`name`、`description` 与用户输入的关键词重叠度。

你也可以手动加载：

```
> /skill load train-debug
```

### 参数化技能

技能支持参数。在 frontmatter 中定义参数和默认值，技能正文中用 `{param_name}` 引用。加载时可以传入参数覆盖默认值。

## 知识缓存

Agent 分析项目文件后，会将结论缓存起来，避免下次重复读取。

### 工作原理

1. 你问"训练环境有哪些依赖？"
2. Agent 读取 `requirements.txt`、`Dockerfile` 等文件，分析后回答
3. Agent 调用 `cache_write` 保存分析结论，记录源文件的 SHA-256 哈希
4. 下次你问类似问题时，系统自动匹配缓存并注入到上下文
5. 如果源文件被修改，缓存自动失效，Agent 重新分析

### 缓存管理

```
> /cache list
  env_dependencies: training environment dependencies [valid, 3 sources]
  project_structure: project directory layout [valid, 1 sources]

> /cache delete env_dependencies
Deleted cache entry 'env_dependencies'.

> /cache clear
Cleared 2 cache entries.
```

### 缓存存储

缓存文件存放在 `~/.flagscale/agent_cache/`，每个条目一个 YAML 文件。默认 7 天过期，可通过 `cache_ttl_days` 配置。

## 会话记忆

知识缓存存的是项目结构知识（依赖、架构），会话记忆存的是工作上下文（发现、决策、待办）。Agent 在对话过程中实时记录关键信息，跨会话持久化。

### 工作原理

Agent 在对话中发现重要信息时，会主动调用 `memory_write` 记录：

```
> Aquila 70B 用 TP=8 训练 OOM 了

⚡ shell(command="grep -i 'out of memory' outputs/aquila70b/logs/latest.log | tail -5")
  ✓ shell (0.5s)

TP=8 时每卡显存不够，建议改为 TP=4+PP=2。

⚡ memory_write(key="aquila70b_tp_oom", type="finding", content="Aquila 70B: TP=8 OOM，改为 TP=4+PP=2 后正常")
  ✓ memory_write (0.0s)
```

下次启动时，Agent 自动加载最近的记忆：

```
╭─ FlagScale Agent ─────────────────────────────╮
│  Provider: anthropic | Model: mco-4            │
╰────────────────────────────────────────────────╯

> 继续调 Aquila 70B 的并行策略

（Agent 已经知道上次 TP=8 OOM，改用了 TP=4+PP=2，直接接着工作）
```

### 记忆类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `finding` | 发现的事实 | "TP=8 时 OOM，显存占用超过 80GB/卡" |
| `decision` | 做出的决策 | "最终采用 TP=4+PP=2+DP=4 的并行策略" |
| `todo` | 待办事项 | "还需要测试 EP 策略对 MoE 模型的效果" |
| `context` | 背景信息 | "用户在准备 V4 架构的 CSA/HCA 混合注意力开发" |

### 记忆管理

```
> /memory list
  [finding] aquila70b_tp_oom: Aquila 70B: TP=8 OOM，改为 TP=4+PP=2 后正常
  [decision] parallel_strategy_final: 最终采用 TP=4+PP=2+DP=4
  [todo] todo_test_ep: 还需要测试 EP 策略

> /memory delete aquila70b_tp_oom
Deleted memory 'aquila70b_tp_oom'.

> /memory clear
Cleared 3 memory entries.
```

### 与知识缓存的区别

| | 知识缓存 | 会话记忆 |
|---|---------|---------|
| 内容 | 项目结构知识 | 工作上下文和决策 |
| 写入方式 | Agent 分析文件后缓存 | Agent 对话中实时记录 |
| 失效机制 | 源文件哈希变化 | TTL 过期（默认 7 天） |
| 粒度 | 大块分析总结 | 单条事实/决策 |
| 存储位置 | `~/.flagscale/agent_cache/` | `~/.flagscale/agent_memory/` |

### 注入预算

启动时最多注入 800 tokens 的记忆（约 10-15 条），按时间从新到旧排列。超出预算的旧记忆不会注入，但仍可通过 `memory_read` 工具主动读取。

## 会话管理

### 自动保存与归档

Agent 每轮对话后自动保存状态到 `~/.flagscale/sessions/autosave.json`。如果意外退出（断网、终端关闭等），下次启动时会提示恢复：

```
╭─ Unfinished session detected ──────────────────╮
│  Time: 2026-04-24 10:30:00                      │
│  5 turns, 3 user messages                        │
│  Last message: 帮我检查训练配置...                │
╰─────────────────────────────────────────────────╯
Resume previous session? [Y/n]:
```

正常退出时（`/quit` 或 Ctrl+C），Agent 会自动归档当前会话到 `~/.flagscale/sessions/session_<id>.json`，然后清除 autosave。会话记忆已在对话过程中实时持久化，退出无需额外 LLM 调用，零延迟。

### 手动保存和加载

```
> /save my-debug-session
✓ Session saved: ~/.flagscale/sessions/my-debug-session.json

> /load
  session_1745500200  2026-04-24 10:30  (3 turns)
    ~/.flagscale/sessions/session_1745500200.json
  my-debug-session    2026-04-24 11:00  (5 turns)
    ~/.flagscale/sessions/my-debug-session.json

> /load my-debug-session
✓ Session loaded: ~/.flagscale/sessions/my-debug-session.json (5 user turns)
```

### 导出

将对话导出为可读的 Markdown 文件：

```
> /export conversation.md
✓ Exported to conversation.md
```

## 文件注入

用 `/file` 命令将文件内容直接注入到对话上下文，Agent 可以在后续回答中引用：

```
> /file configs/train_aquila.yaml
📎 Injected configs/train_aquila.yaml (2,340 chars)

> 这个配置有什么问题？
```

适合在提问前给 Agent 提供背景信息。

## 费用控制

Agent 实时追踪 API 调用费用。

### 预算限制

在配置中设置 `max_cost`（单位：美元）：

```yaml
max_cost: 5.0
```

- 费用达到 80% 时显示警告
- 超出预算时自动停止

### 费用显示

每轮对话和退出时都会显示费用：

```
── Turn 3 | 12.5s | ↑8,234 ↓1,456 tokens | $0.12 / $5.00 ──

Session: 3 turns | 45.2s | ↑24,567 ↓4,321 tokens | $0.35 / $5.00
Bye!
```

### 自定义定价

如果使用自部署模型或非标准定价，可以在配置中覆盖：

```yaml
pricing:
  my-custom-model:
    input: 3.0     # 每百万 input token 的价格
    output: 15.0   # 每百万 output token 的价格
```

## 插件工具

除了内置工具，你可以通过 JSON 文件定义自定义工具。

### 定义插件工具

创建一个 JSON 文件：

```json
{
  "name": "check_gpu",
  "description": "检查 GPU 使用情况",
  "command": "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv",
  "timeout": 30,
  "parameters": {
    "type": "object",
    "properties": {}
  }
}
```

带参数的工具：

```json
{
  "name": "grep_logs",
  "description": "在训练日志中搜索关键词",
  "command": "grep -rn {keyword} outputs/{exp_name}/logs/ | tail -50",
  "timeout": 30,
  "parameters": {
    "type": "object",
    "properties": {
      "keyword": {
        "type": "string",
        "description": "搜索关键词"
      },
      "exp_name": {
        "type": "string",
        "description": "实验名称"
      }
    },
    "required": ["keyword", "exp_name"]
  }
}
```

参数通过 `{param_name}` 占位符替换到命令中，自动使用 `shlex.quote()` 防止命令注入。

### 加载插件工具

在配置中指定插件目录：

```yaml
plugin_tool_dirs:
  - /path/to/my/tools
  - .flagscale/tools
```

目录下所有 `.json` 文件会被自动加载为工具。

## 网络代理

如果你的环境需要代理才能访问外网，有两种配置方式：

### 方式一：环境变量

```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
```

Agent 会自动继承这些变量到 Shell 命令和 Web 请求中。

### 方式二：配置文件

```yaml
shell_env:
  HTTP_PROXY: "http://proxy.example.com:8080"
  HTTPS_PROXY: "http://proxy.example.com:8080"
```

如果 Shell 命令遇到网络错误且未配置代理，Agent 会自动提示你配置。

## 常见用法示例

### 查看训练环境依赖

```
> FlagScale 训练环境有哪些依赖？
```

Agent 会读取 `requirements.txt`、`Dockerfile` 等文件，给出完整的依赖清单。首次分析后会缓存结果，下次直接复用。

### 排查训练故障

```
> 训练 OOM 了，帮我看看怎么回事
```

Agent 会自动加载训练排查技能（如果有），然后检查日志、配置、GPU 状态。

### 修改配置

```
> 帮我把 aquila 的训练配置改成用 4 机 32 卡
```

Agent 会读取配置文件，理解当前配置，然后修改并解释改动。

### 检查项目结构

```
> FlagScale 的目录结构是怎样的？各个目录是做什么的？
```

### 查阅文档

```
> 帮我查一下 Megatron-LM 的 tensor parallel 文档
```

Agent 会用 `web_fetch` 工具抓取相关文档页面。

## 目录结构参考

```
~/.flagscale/
  agent.yaml              # 用户级配置
  agent_cache/            # 知识缓存
    env_dependencies.yaml
    project_structure.yaml
  agent_memory/           # 会话记忆
    aquila70b_tp_oom.yaml
    parallel_strategy_final.yaml
  sessions/               # 会话存储
    autosave.json
    session_a1b2c3d4.json # 自动归档的会话
    my-session.json       # 手动保存的会话
  skills/                 # 用户级技能
    my-skill/
      SKILL.md
  input_history           # 输入历史（用于方向键回溯）

<项目目录>/
  .flagscale/
    agent.yaml            # 项目级配置
    skills/               # 项目级技能
    tools/                # 项目级插件工具
```
