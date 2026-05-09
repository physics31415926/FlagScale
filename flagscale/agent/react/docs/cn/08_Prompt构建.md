# Prompt 构建

## 概述

`PromptMixin` 根据当前上下文、活跃技能和情境需求动态构建系统提示词。

## Prompt 层次

```
SYSTEM_PROMPT_CORE          （始终存在——身份、能力、规则）
    + SYSTEM_PROMPT_OPTIONAL （工具描述、技能上下文）
    + 情境段落               （按条件添加）
    = 最终系统提示词
```

## 情境段落

根据 Agent 当前状态有条件地包含：

| 段落键 | 触发条件 | 内容 |
|--------|----------|------|
| `training` | 活跃训练任务 | 训练工作流指导、日志分析优先级 |
| `porting` | 模型迁移技能已加载 | 迁移方法论、诊断打印策略 |
| `decision` | 始终（技能活跃时） | 决策规则、分层错误恢复 |
| `config_schema` | 训练上下文活跃 | FlagScale YAML 配置结构参考 |

### 训练上下文
- 日志分析优先级：OOM/CUDA > NCCL 超时 > loss 异常 > 迭代慢 > 警告
- 各阶段工具推荐
- 监控最佳实践

### 迁移上下文
- 模型结构枚举要求
- 诊断打印策略（模块边界处打印 shape/dtype）
- 阶段顺序提醒

### 配置 Schema
- 两级 Hydra 配置结构（顶层 + 模型层）
- 有效键位置
- 常见错误放置

## 记忆上下文

`_build_memory_context()`：
- 将相关记忆条目注入系统提示词
- 当 `_consecutive_train_failures >= 1` 时：查询错误相关记忆
- 注入条目标记为 `[RELEVANT:key]`

## 轮上下文

`_build_turn_context()`：
- 注入 ExperimentManager 的失败尝试（最近 5 条）
- 格式：`FAILED attempts (DO NOT REPEAT):`
- 确保 LLM 在生成新方案前看到历史失败

## 工具阶段检测

`_detect_tool_phase(tool_name, arguments)`：
- 将当前操作分类到阶段，供 Gate 系统使用
- 被阶段顺序 Gate 用于强制执行序列
- 基于工具名 + 参数模式（非输出关键词匹配）
