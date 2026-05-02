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

## 数据边界

### 允许读取的数据源

| 路径 | 内容 | 用途 |
|------|------|------|
| `trajectory/output/raw/{session_id}/events.jsonl` | 原始 hook 事件 | 提取训练数据（从 tool_response 字段） |
| `trajectory/output/trajectories/{session_id}/trajectories.jsonl` | 分割后轨迹 | 主要分析输入 |
| `trajectory/output/reports/` | 历史分析报告 | 跨轮次对比 |
| `trajectory/output/index.jsonl` | 全局索引 | 查找相关 session |

### 禁止直接读取的数据源

| 路径 | 原因 |
|------|------|
| `rllm_trl/output/runs/*/` | 属于 rllm-xx，只能通过轨迹间接获取 |
| `rllm_trl/config.py` | 属于 rllm-xx 内部实现 |
| `rllm_trl/*.py` | 属于 rllm-xx 内部实现 |
| `skill-bank/rllm/*/base.md` | 属于 rllm-xx skill 源码 |
| `.claude/skills/rllm-*/SKILL.md` | 属于 rllm-xx 编译产物 |

### 上下文隔离说明

本 skill 在独立 Agent 子 agent 中执行，拥有全新的对话上下文。物理上无法看到:
- rllm-train 子 agent 中读取的 config.json 内容
- rllm-monitor 子 agent 中 tail 的 training_log.txt 输出
- 任何 rllm-xx 执行过程中的中间状态

唯一的数据来源是 trajectory/output/ 目录下的文件。

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

### 2. 从轨迹数据提取训练详情

对每条 rllm-train 轨迹:

a) 读取 trajectory/output/trajectories/{session_id}/trajectories.jsonl
   → 定位 skill_name="rllm-*" 的轨迹

b) 从轨迹的 tool_calls 中提取训练数据:
   - tool_name="Read" + file_path 含 "config.json" → 训练配置
   - tool_name="Bash" + command 含 "tail" + "training_log" → reward 趋势
   - tool_name="Read" + file_path 含 "perf_stats.json" → 性能统计
   - tool_name="Bash" + response 含 "Error"/"Traceback" → 错误信息

c) 使用 Python 辅助方法简化提取:
   ```python
   from trajectory.analyzer.base import AnalyzerBase
   analyzer = AnalyzerBase(DEFAULT_CONFIG)
   training_data_list = analyzer.get_available_training_data()
   # 返回: [{trajectory_summary, training_data: {config, reward_trend, perf_stats, errors, log_snippets}}]
   ```

d) 如果轨迹数据为空或 tool_response 中缺少关键信息:
   → 报告: "数据不足。请确认 rllm-xx skill 的数据表面化准则已实施。
            缺失: [config/reward_trend/perf_stats/errors]"
   → 不做猜测性分析，明确标注数据缺失

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

## 领域知识: 训练动态模式识别

### 可识别的训练动态模式

| 模式 | 轨迹中的特征 | 含义 | 分析方法 |
|------|-------------|------|---------|
| 快速崩溃 | reward 在前 10% steps 内从峰值降为 0 | 学习率过高或训练量过大 | 对比相同模型不同 lr 的轨迹 |
| 延迟崩溃 | reward 在 10-30% steps 时开始下降 | 数据量超出模型容量 | 对比不同 num_problems 的轨迹 |
| 稳定学习 | reward 单调递增或小幅波动 | 配置合理 | 记录为安全配置参考 |
| 高 variance | reward 大幅震荡（相邻步差值 > 0.3） | 学习率偏高或 batch 偏小 | 对比相邻轮次的 lr 和 reward variance |
| 格式退化 | 后期 tool_call 格式错误增加 | 模型遗忘了格式模板 | 检查训练长度和格式 reward 权重 |
| 零学习 | reward 始终接近 0（< 0.1） | 题目太难或 lr 太低 | 对比难度配置和模型能力 |
| Catastrophic Forgetting | reward 先升后降，峰值后持续下降 | 过拟合或 epochs 过多 | 检查 epoch 边界，对比 epochs 配置 |

### 分析方法论

1. **从轨迹提取事实** — 读取轨迹数据，提取配置参数、reward 序列、错误信息
2. **跨轮次对比** — 比较不同轮次的参数变化与结果变化，建立因果关系
3. **模式匹配** — 将观察到的 reward 趋势与上述模式表对照
4. **安全边界推断** — 基于多轮数据推断参数安全范围（而非预设）
   - 示例: 如果 3 轮轨迹显示 lr=1e-5 时 reward 崩溃，lr=5e-6 时稳定，推断该模型的 lr 安全上限为 5e-6
5. **生成假说** — 对观察到的现象提出可能的解释，标注置信度
   - 高置信度: 基于 3+ 轮轨迹的一致模式
   - 中置信度: 基于 1-2 轮轨迹
   - 低置信度: 推测性解释，需要更多数据验证

### 禁止事项

- 不引用预设的参数安全范围表（如 "qwen-0.5b lr 应在 1e-6~1e-5"）
- 不假设固定的 "问题 → skill" 映射（如 "lr 过高一定修改 rllm-config"）
- 所有数值判断须有轨迹证据支撑，引用具体的 session_id 和数据
- 不使用 "经验表明" / "通常情况下" 等无证据表述
- 如果轨迹数据不足以得出结论，明确报告 "数据不足，需要更多轮次训练"
