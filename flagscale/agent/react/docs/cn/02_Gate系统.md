# Gate 系统

## 概述

Gate 系统是执行前/后的安全拦截层，在工具调用执行前进行检查。Gate 可以 **硬阻断**（完全阻止执行）或 **软警告**（注入建议信息但不阻止执行）。

## 设计

Gate 方法签名：`(self, tool_name: str, arguments: dict) -> str`

返回值：
- 空字符串 `""`：Gate 通过，不干预
- 非空字符串：Gate 消息（阻断或警告文本）

Gate 分为两类：
- `hard_gates`：返回值包含 `"TOOL NOT EXECUTED"`，工具调用被跳过
- `soft_gates`：返回值作为建议注入，但不阻止执行

## Gate 清单（31 个方法）

### 硬阻断 Gate

| Gate | 用途 |
|------|------|
| `_check_training_hang` | 训练疑似卡死时阻断（评分制，需 ≥2 个独立信号） |
| `_check_distributed_prerequisite_gate` | 多节点操作前确保分布式环境已就绪 |
| `_check_checkpoint_verified_gate` | 训练启动前必须用 `inspect_checkpoint` 深度验证 |
| `_check_pipeline_comprehension_gate` | 修改 pipeline 代码前确保已理解代码 |
| `_check_structure_completeness_gate` | Checkpoint 转换前必须已枚举模型结构 |
| `_check_phase_ordering_gate` | 强制 6 阶段严格有序（分析→训练） |
| `_check_data_parallelism_gate` | 写数据管道代码前必须已文档化并行策略 |

### 软警告 Gate

| Gate | 用途 |
|------|------|
| `_check_context_pressure` | 上下文使用率过高时警告，触发强制压缩 |
| `_check_reading_quality` | 文件读取过浅时警告 |
| `_check_error_escalation` | 重复失败时升级为结构化诊断 |
| `_check_source_reading_gate` | 2 次以上失败后提醒阅读框架源码 |
| `_check_analysis_persistence` | 提醒持久化分析结果 |
| `_check_plan_maintenance_gate` | 计划步骤停滞 8 轮以上时警告 |
| `_check_config_validation_hint` | 写 YAML 到 conf/ 后建议调用 validate_config |
| `_check_diagnostic_print_hint` | 写 forward/init 代码时建议添加诊断打印 |

## 关键行为

### 自动解除机制
- **Monitor Gate**：连续 5 次阻断后自动清除（防止永久卡死）
- **Pipeline 理解 Gate**：Phase 3 连续 3 次阻断后自动通过（防止标记死锁）
- **训练卡死检测**：评分制（需 ≥2 个独立信号），非纯关键词匹配

### 错误升级（分层诊断）
2 次以上失败后，强制按以下顺序排查：
1. 环境检查（Python 路径、CUDA、包版本）
2. 依赖验证（版本兼容性）
3. 源码阅读（框架内部实现）
4. 代码修复（理解根因后再修改）

### 上下文压力管理
- 80% 使用率：软警告
- 90% 使用率：触发强制压缩，目标 50%
- 每会话最多 3 条警告（防止 Token 浪费）

## 集成

Gate 在主循环中通过 `_run_pre_execution_gates(tool_name, arguments)` 调用，位于解析工具调用之后、实际执行之前。硬阻断短路（第一个阻断即生效）。所有软警告均会执行，消息拼接后注入。
