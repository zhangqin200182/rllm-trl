---
description: LLM-based analysis of rllm-train trajectories. Identifies failure patterns,
  parameter safety boundaries, and generates optimization suggestions for rllm-xx
  skills.
metadata:
  categories:
  - trajectory
  - analysis
  - rllm
  version: 1.0.0
name: traj-analyze-rllm
---


# traj-analyze-rllm — rllm 训练轨迹分析

你是 rllm-train 训练轨迹的分析专家。你的职责是分析最近的 rllm-xx skill 执行轨迹，发现问题模式，并生成结构化的优化建议。

## 分析框架

### 分析维度

1. **训练动态** — reward 趋势、loss 变化、grad_norm 稳定性
2. **失败模式** — OOM、参数冲突、catastrophic forgetting、格式退化、monitor 静默失效
3. **参数安全** — 跨轮次 lr/batch_size/epoch 与 reward 的关系，推断安全边界
4. **配置合理性** — 检查是否有明显不合理的参数组合
5. **流程效率** — 是否有不必要的重复操作、遗漏的检查步骤

### 失败模式检测

| 模式 | 特征 |
|------|------|
| lr 过高崩溃 | reward 突然降为 0 或接近 0 |
| OOM | Bash 工具返回 CUDA OOM / MPS OOM / MemoryError |
| catastrophic forgetting | reward 先升后降 |
| 格式退化 | 后期 tool call 格式错误增加 |
| monitor 静默失效 | Monitor 工具长时间无输出事件 |
| 配置生成错误 | rllm-config 输出后训练立即报错 |

## 执行步骤

### 1. 加载轨迹

```python
from trajectory.analyzer.base import AnalyzerBase
from trajectory.config import DEFAULT_CONFIG

analyzer = AnalyzerBase(DEFAULT_CONFIG)
trajectories = analyzer.get_rllm_trajectories()
```

如果轨迹数 < 1，提示用户先执行训练再分析。

### 2. 读取训练详情

对每条 rllm-train 轨迹:
- 从 tool_calls 中提取 rllm-config 生成的配置参数
- 从 Bash 工具调用中提取训练日志输出
- 从 rllm-analyze 调用中提取 analysis.json 内容
- 记录失败的工具调用及其错误信息

### 3. 跨轮次关联分析

将多轮训练数据放在一起分析:
- 参数变化与 reward 变化的因果关系
- 同一参数在不同场景下的表现差异
- 递进式的错误模式（如第一轮 OOM 后调参，第二轮仍有问题）

### 4. 生成分析报告

报告格式:

```markdown
# rllm-train 轨迹分析报告

生成时间: {timestamp}
分析范围: 最近 {days} 天
rllm-train 轨迹: {count} 条

## 训练执行概览

| Session | 配置摘要 | 结果 | 关键问题 |
|---------|---------|------|---------|
| ... | ... | ... | ... |

## 问题发现

### 1. {问题标题} [影响: {target_skill}]
**现象**: ...
**证据**: ...
**建议**: ...

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 |
|--------|-----------|---------|--------|------|
| ... | ... | ... | ... | ... |

## 建议的 Patch 内容

### Patch 1: {description}
(完整 patch markdown 内容)
```

### 5. 保存报告

```python
from trajectory.analyzer.report import ReportWriter

writer = ReportWriter(DEFAULT_CONFIG)
report_path = writer.write_report(report_content, prefix="rllm-analysis")
```

输出报告路径供后续 traj-optimize 使用。

## --optimize 模式

当带 `--optimize` 参数调用时，分析完成后自动调用 traj-optimize:

1. 执行上述所有分析步骤
2. 生成报告和结构化 SkillOptimizationSuggestion
3. 调用 `Skill("traj-optimize", args="<report_path>")`

这实现了半自动流程: 一键完成 分割 → 分析 → 生成 patch。

## 领域知识（用于分析判断）

### rllm_trl 参数安全范围（基于历史经验）

| 模型 | lr 安全范围 | 推荐 batch_size | 最大 epoch |
|------|------------|----------------|-----------|
| qwen-0.5b | 1e-6 ~ 1e-5 | 4-8 | 3 |
| qwen-1.5b | 5e-7 ~ 5e-6 | 2-4 | 2 |
| qwen-3b | 1e-7 ~ 2e-6 | 1-2 | 2 |

### 常见问题与对应 skill 优化方向

| 问题 | 影响 Skill | 可能的优化 |
|------|-----------|-----------|
| lr 过高 | rllm-config | 增加模型级安全约束 |
| OOM | rllm-config | 增加内存估算逻辑 |
| monitor 静默 | rllm-monitor | 通用化 grep 模式 |
| 格式退化 | rllm-run | 增加格式检查点 |
| 数据过简单 | rllm-config | 增加难度参数 |
