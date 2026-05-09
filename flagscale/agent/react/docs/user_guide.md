# FlagScale Agent 用户指南

## 简介

FlagScale Agent 是面向大模型训练基础设施的 AI 助手。它能读写文件、执行命令、查阅文档，帮助你完成环境搭建、配置调优、模型迁移和故障排查。

Agent 采用 ReAct（Reasoning + Acting）模式：思考 → 调用工具 → 观察结果 → 继续推理，直到任务完成。

### v6 新特性

- **6 阶段严格有序**的模型迁移流程（分析 → 结构实现 → 结构验证 → 数据管道 → Checkpoint 转换 → 训练验证）
- **Gate 系统**：31 个安全门控（硬阻断 + 软警告），防止过早或危险操作
- **循环检测**：防止重复执行相同的工具调用
- **Checkpoint 深度验证**：shape/dtype/NaN/Inf/全零异常检测
- **主动记忆召回**：错误发生时自动检索相关历史发现
- **配置验证工具**：训练启动前捕获 YAML 错误放置
- **进程死亡检测**：训练崩溃时立即反馈
- **语义去重**：通过 LLM 判断避免重复尝试已失败的修复方案

## 快速开始

### 1. 配置 API Key

```bash
# Anthropic（默认）
export ANTHROPIC_API_KEY="sk-ant-..."

# 或 OpenAI
export OPENAI_API_KEY="sk-..."
```

### 2. 启动 Agent

```bash
# 交互模式（默认 Anthropic + claude-sonnet）
flagscale agent

# 指定 provider 和模型
flagscale agent --provider openai --model gpt-4o

# 自定义 API 地址（适配代理/网关）
flagscale agent --base-url https://my-proxy.example.com/v1

# 单次查询（非交互模式）
flagscale agent "FlagScale 训练环境有哪些依赖？"
```

CLI 参数：

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--provider` | `-p` | LLM 后端：`anthropic`（默认）或 `openai` |
| `--model` | `-m` | 模型名称 |
| `--base-url` | `-b` | API 地址，用于代理或自部署网关 |
| `--config` | `-c` | 配置文件路径 |

Python API：

```python
from flagscale.agent.react import AgentConfig, ReactAgent

config = AgentConfig.auto_load(provider="anthropic", model="claude-sonnet-4-20250514")
agent = ReactAgent(config)

# 交互式 REPL
agent.run()

# 单次查询
agent.run(single_shot_query="检查训练环境依赖")
```

### 3. 开始对话

```
╭─ FlagScale Agent ─────────────────────────────╮
│  Provider: anthropic | Model: claude-sonnet    │
│  Commands: /skill  /file  /save  /load  ...    │
╰────────────────────────────────────────────────╯

>
```

直接输入问题。Agent 会自动思考、调用工具、给出回答。

## 配置

### 配置文件

Agent 按以下优先级查找配置：

1. 环境变量 `FLAGSCALE_AGENT_CONFIG` 指定的路径
2. 当前目录 `.flagscale/agent.yaml`
3. 用户目录 `~/.flagscale/agent.yaml`

示例：

```yaml
# ~/.flagscale/agent.yaml

provider: anthropic
model: claude-sonnet-4-20250514

# 行为控制
max_iterations: 200
max_context_tokens: 100000
shell_timeout: 120
dangerous_commands_check: true
confirm_commands: true

# 费用控制
max_cost: 5.0
pricing:
  my-custom-model:
    input: 3.0
    output: 15.0

# 记忆
memory_ttl_days: 7
cache_ttl_days: 7

# 网络代理
shell_env:
  HTTP_PROXY: "http://proxy.example.com:8080"
  HTTPS_PROXY: "http://proxy.example.com:8080"

# 自定义技能/工具目录
skill_dirs:
  - /path/to/custom/skills
plugin_tool_dirs:
  - /path/to/plugin/tools
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `ANTHROPIC_AUTH_TOKEN` | Anthropic Auth Token（优先级高于 API_KEY） |
| `ANTHROPIC_BASE_URL` | Anthropic API 自定义地址 |
| `OPENAI_API_KEY` | OpenAI API Key |
| `OPENAI_BASE_URL` | OpenAI API 自定义地址 |
| `FLAGSCALE_AGENT_CONFIG` | 配置文件路径 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 网络代理 |

### 热重载

```
> /reload
Config and skills reloaded.
```

## 交互命令

以 `/` 开头的输入是命令，不会发送给 LLM：

| 命令 | 说明 |
|------|------|
| `/quit` | 退出 Agent |
| `/reload` | 重新加载配置和技能 |
| `/skill list` | 列出所有可用技能 |
| `/skill load <name>` | 手动加载技能 |
| `/file <path>` | 将文件内容注入对话上下文 |
| `/save [name]` | 保存当前会话 |
| `/load [name]` | 加载已保存的会话 |
| `/export [path]` | 导出对话为 Markdown |
| `/memory list` | 列出所有记忆条目 |
| `/memory delete <key>` | 删除指定记忆 |
| `/memory clear` | 清空所有记忆 |

## 工具（18 个）

Agent 在对话中自动选择和调用工具：

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容（带 30s TTL 缓存） |
| `write_file` | 创建或覆盖文件 |
| `edit_file` | 精确替换文件中的字符串 |
| `shell` | 执行 Shell 命令 |
| `web_fetch` | 抓取网页内容 |
| `find_latest_log` | 查找和过滤日志文件（errors/progress/all） |
| `monitor` | 长时间文件/进程监控，无 LLM 调用 |
| `parse_training_metrics` | 从日志提取 loss/throughput/MFU |
| `inspect_checkpoint` | Checkpoint 深度验证（shape/dtype/异常检测） |
| `validate_config` | YAML 配置结构验证 |
| `memory_write` | 保存发现、决策或待办 |
| `memory_read` | 读取记忆条目 |
| `memory_list` | 列出所有记忆键 |
| `workspace_experiment` | 管理实验记录 |
| `plan_create` | 创建任务计划 |
| `plan_update` | 更新计划步骤状态 |
| `plan_status` | 显示计划状态 |
| `load_skill` | 加载技能定义 |

### Shell 安全机制

三层保护：

1. **致命命令拦截**：`rm -rf /`、`mkfs`、`dd if=` 直接拒绝
2. **高风险命令确认**：`rm`、`kill`、`git push`、`pip install` 需要确认
3. **自杀保护**：可能杀死 Agent 自身进程的命令会自动改写

### Monitor 工具

用于长时间操作（训练、模型加载）：

```
> 启动训练后帮我监控日志

⚡ monitor(file="outputs/qwen3/logs/train.log", duration=300, target_step=100)
  ⠹ poll #15 — step=42, loss=2.31 (150s)
  ...
  ✓ monitor: target_reached (180s)

训练已到达 step 100，loss 从 4.2 降到 1.8，正常收敛。
```

Monitor 会在以下情况返回：
- 检测到错误模式（ERROR/OOM/NCCL 等）
- 匹配成功/失败模式
- 达到目标步数
- 训练进程死亡
- 超时

### Inspect Checkpoint 工具

深度验证 checkpoint 质量：

```
> 帮我检查转换后的 checkpoint

⚡ inspect_checkpoint(path="checkpoints/qwen3_converted.pt", reference_path="checkpoints/original.pt")
  ✓ inspect_checkpoint

Keys: 142 | Params: 0.6B | dtype: torch.bfloat16
Anomalies: None
Reference comparison: all keys match shape and dtype [OK]
```

检测项：NaN、Inf、全零张量、shape/dtype 不匹配、缺失/多余的键。

### Validate Config 工具

训练前验证配置正确性：

```
> 帮我验证训练配置

⚡ validate_config(path="conf/train/qwen3_0_6b.yaml")
  ✓ validate_config

Config type: model-level
ERRORS:
  - bf16 found under 'model' — should be under 'system.precision'
  - save_interval found under 'model' — should be under 'system.checkpoint'
```

## 技能系统

技能是预置的领域知识模块，让 Agent 在特定场景下获得专业指导。

### 内置技能

| 技能 | 用途 |
|------|------|
| `model-porter` | 模型迁移方法论（含 6 阶段流程） |
| `data-prep` | 数据管道准备（含并行策略前置要求） |
| `train-config` | 训练配置生成 |
| `train-run` | 训练执行 |
| `train-monitor` | 训练监控 |
| `env-setup` | 环境搭建 |
| `topo-detect` | 硬件拓扑检测 |
| `reproduce` | 基线复现 |
| `precision-alignment` | Loss 曲线对齐 |

### 自动加载

Agent 根据你的输入自动匹配并加载相关技能。也可手动加载：

```
> /skill load model-porter
```

### 技能文件格式

```markdown
---
name: my-skill
description: 技能描述
keywords: [关键词1, 关键词2]
parameters:
  - name: model
    description: 模型名称
    default: qwen3
---

## 步骤

1. 第一步...
2. 第二步...
```

技能目录扫描顺序：
1. 内置：`flagscale/agent/skills/`
2. 项目级：`.flagscale/skills/`
3. 用户级：`~/.flagscale/skills/`
4. 配置文件 `skill_dirs` 指定的目录

## 记忆系统

### SessionMemory

Agent 在对话中实时记录关键信息，跨会话持久化：

```
> Qwen3 用 TP=8 训练 OOM 了

⚡ memory_write(key="qwen3_tp_oom", type="finding", priority="high",
    content="Qwen3-0.6B: TP=8 OOM，改为 TP=4+PP=2 后正常")
```

下次启动时自动加载相关记忆。

### 记忆优先级

| 优先级 | TTL | 用途 |
|--------|-----|------|
| `high` | 永不过期 | 关键发现、永久决策 |
| `critical` | 7 天 | 压缩检查点 |
| `normal` | 30 天 | 标准发现 |
| `low` | 7 天 | 临时上下文 |

访问 ≥3 次的条目自动从 `normal` 提升为 `high`。

### 主动召回

训练失败时，Agent 自动从记忆中检索相关条目并注入上下文，无需手动查询。

### ExperimentManager

结构化追踪每次实验尝试：

```
> 记录这次训练尝试

⚡ workspace_experiment(action="add_attempt", name="qwen3_porting",
    change="修改 attention mask 为 causal", result="loss 正常下降")
```

失败尝试会被自动注入后续对话，标记为 `FAILED attempts (DO NOT REPEAT)`。

## 计划管理

Agent 可以创建和追踪任务计划：

```
> 帮我制定 Qwen3 迁移计划

⚡ plan_create(steps=["枚举模型结构", "实现 Attention 模块", "实现 MLP 模块", ...])
```

计划特性：
- **自动同步**：工具执行后自动更新步骤状态
- **一致性检查**：每 5 轮检测停滞和重复失败
- **重建建议**：连续 3 次失败时建议重新制定计划

## Gate 系统（安全门控）

Agent 内置 31 个安全门控，自动防止危险操作：

- **阶段顺序 Gate**：模型结构未完成前阻止 checkpoint 转换
- **源码阅读 Gate**：2 次以上失败后，修代码前必须先读框架源码
- **并行策略 Gate**：写数据管道前必须文档化全部并行维度
- **诊断打印提示**：写 forward/init 代码时建议添加 shape/dtype 打印

Gate 不需要用户干预，在后台自动工作。

## 会话管理

### 自动保存

每轮对话后自动保存。意外退出后下次启动会提示恢复：

```
╭─ Unfinished session detected ──────────────────╮
│  Time: 2026-05-09 10:30:00                      │
│  5 turns, 3 user messages                        │
╰─────────────────────────────────────────────────╯
Resume previous session? [Y/n]:
```

### 手动保存和加载

```
> /save my-session
✓ Session saved

> /load my-session
✓ Session loaded (5 turns)
```

## 费用控制

```yaml
max_cost: 5.0  # 预算上限（美元）
```

- 80% 时显示警告
- 超出预算自动停止
- 每轮显示费用：`Turn 3 | 12.5s | ↑8,234 ↓1,456 tokens | $0.12 / $5.00`

## 插件工具

通过 JSON 文件定义自定义工具：

```json
{
  "name": "check_gpu",
  "description": "检查 GPU 使用情况",
  "command": "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv",
  "timeout": 30,
  "parameters": {"type": "object", "properties": {}}
}
```

参数通过 `{param_name}` 占位符替换，自动 `shlex.quote()` 防注入。

配置中指定插件目录：

```yaml
plugin_tool_dirs:
  - .flagscale/tools
```

## 网络代理

```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
```

或在配置文件中：

```yaml
shell_env:
  HTTP_PROXY: "http://proxy.example.com:8080"
  HTTPS_PROXY: "http://proxy.example.com:8080"
```

## 常见用法

### 模型迁移

```
> 帮我把 Qwen3-0.6B 迁移到 FlagScale
```

Agent 自动加载 `model-porter` 技能，按 6 阶段流程执行。

### 排查训练故障

```
> 训练 OOM 了，帮我看看怎么回事
```

Agent 检查日志（优先级：OOM/CUDA > NCCL > loss 异常 > 慢迭代），给出诊断。

### 验证 Checkpoint

```
> 帮我验证转换后的 checkpoint 是否正确
```

Agent 使用 `inspect_checkpoint` 做深度检查，对比 reference。

### 修改配置

```
> 帮我把训练配置改成 TP=4 PP=2
```

Agent 读取配置、修改、然后调用 `validate_config` 验证。

### 监控训练

```
> 启动训练后帮我监控到 step 1000
```

Agent 使用 `monitor` 工具本地轮询，不消耗 LLM Token。

## 目录结构

```
~/.flagscale/
  agent.yaml              # 用户级配置
  agent_memory/           # 会话记忆
  sessions/               # 会话存储
  input_history           # 输入历史

<项目目录>/
  .flagscale/
    agent.yaml            # 项目级配置
    skills/               # 项目级技能
    tools/                # 项目级插件工具
```
