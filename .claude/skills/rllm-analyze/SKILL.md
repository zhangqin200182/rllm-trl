---
description: Analyze rllm_trl training results including reward effectiveness, training
  speed, and performance bottlenecks. Generates specific hyperparameter tuning recommendations
  for the next training round.
metadata:
  categories:
  - machine-learning
  - analysis
  version: 1.0.0
name: rllm-analyze
---


# rllm-analyze — 训练结果分析与调参建议

你是 rllm_trl 训练分析专家。你的任务是分析训练结果，诊断问题，并生成具体的调参建议。

## 分析输入

训练输出目录: `rllm_trl/output/runs/<run_id>/`

需要读取的文件：
1. `config.json` — 本轮训练配置
2. `training_log.txt` — 训练日志（reward/loss 趋势）
3. `perf_stats.json` — 性能统计（时间分解、吞吐量）
4. `trajectories/` 目录下的 JSONL 文件 — agent 行为轨迹

## 分析维度

### 一、训练效果分析

从 training_log.txt 和 perf_stats.json 中提取：

**Reward 分析**:
- 初始 reward（第 1 步）
- 最终 reward（最后一步）
- 最高 reward
- reward 趋势（上升/下降/震荡/plateau）
- 是否达到目标阈值

**Loss 分析**:
- loss 趋势（应该下降）
- loss 是否收敛
- loss 与 reward 的相关性

**Agent 行为分析**（从 trajectory JSONL）:
- 平均对话轮次
- 工具调用成功率
- 常见错误模式（格式错误、计算错误、未调用 finish）
- 正确回答的问题类型分布

### Epoch 分段分析（新增分析维度）

将 per_step_rollouts 按 epoch 切分:
  `steps_per_epoch = total_steps / num_epochs`
  `epoch_rewards = [avg(rewards[i*spe : (i+1)*spe]) for i in range(num_epochs)]`

检测规则:
  if `epoch_rewards[i+1] < epoch_rewards[i] * 0.3`:
      → 标记为 "catastrophic forgetting at epoch {i+1}"
      → 建议: 减少 epochs，当前模型容量不足以支撑多 epoch 训练

输出示例:
```
Epoch 分析:
  Epoch 1: avg_reward=0.45, tool_call_rate=85%
  Epoch 2: avg_reward=0.02, tool_call_rate=12%  ← 断崖下降
  诊断: catastrophic forgetting
  建议: num_epochs 不超过 1-2 (0.5B 模型限制)
```

### 格式退化检测（新增分析维度）

从 trajectory JSONL 中提取:
  前 25% 步骤的 tool_call 使用率 (前期)
  后 25% 步骤的 tool_call 使用率 (后期)

检测规则:
  if 后期 tool_call 率 < 前期 * 0.5:
      → 标记为 "format degradation"
      → 检查后期 assistant 输出样本，识别退化模式:
        - 纯文本数字 ("1097 + 38 = 11015") → 模型放弃工具调用
        - Python 代码 ("def calculate(...)") → 模型混淆了输出格式
        - 空输出或重复 → 模型崩溃

  建议: 减少 epochs/lr，或增加格式正确性辅助 reward

### 二、性能分析

从 perf_stats.json 中提取：

**时间分解**:
- LLM 推理时间占比
- 环境执行时间占比
- Logprob 计算时间占比
- GRPO 训练时间占比
- 其他开销

**吞吐量**:
- tokens/sec
- steps/min
- 每步平均耗时

**瓶颈识别**:
- 哪个阶段占比最高？
- 是否有异常慢的步骤？

### 三、对比分析（多轮训练时）

如果存在历史训练记录，对比：
- 本轮 vs 上一轮的 reward 变化
- 配置变更是否带来预期效果
- 性能是否有退化

## 调参建议生成

基于分析结果，生成具体的配置修改建议。每条建议包含：

```
建议 1 [优先级: 高]:
  修改: learning_rate 1e-5 → 2e-5
  原因: reward 持续上升但速度放缓，loss 仍在下降，可以加大学习率加速收敛
  预期: reward 提升速度加快
  风险: 可能导致训练不稳定，如果 reward 开始震荡则回退
```

### 调参决策树

```
reward 已达标 (=1.0)?
├── loss=0, grad=0 → 题目太简单
│   └── 建议: difficulty 切换到 mixed 或 hard
│       不要调其他参数，问题不在超参而在数据
└── loss>0 → 正常，训练有效

reward 未达标?
├── reward 在上升
│   ├── loss 在降 → 增加 epochs 或 problems（需要更多训练）
│   └── loss 不降 → 增大 lr 或 num_generations（学习信号不足）
├── reward 停滞 (plateau)
│   ├── loss 也停滞 → 调整 temperature、尝试不同 loss_type
│   └── loss 在降 → 增加 num_generations（探索不足）
├── reward 震荡
│   └── 降低 lr、增大 grad_accum_steps（训练不稳定）
├── reward 下降
│   └── 大幅降低 lr、回退配置（过拟合或 lr 过大）
├── Epoch 间断崖式下降
│   └── Epoch N avg_reward > 0.3 且 Epoch N+1 avg_reward < 0.1
│       → catastrophic forgetting
│       → 建议: 减少 epochs (不超过模型安全上限)
│       → 如果 epochs 已经是 1-2，建议换更大模型
└── 格式退化
    └── 后半段 tool_call 使用率 < 前半段的 50%
        → 模型"忘记"了工具调用格式
        → 建议: 减少 epochs，或增加格式正确性辅助 reward

性能问题？
├── LLM 推理占比 > 80%
│   └── 减小 max_completion_length、max_agent_steps
├── 总耗时过长
│   └── 减少 num_problems 或 num_epochs
└── 内存不足
    └── 减小 batch_size、num_generations
```

## 输出格式

### 分析报告

```
训练分析报告 [Run: <run_id>]
================================

效果分析:
  Reward:    0.25 → 0.45 (↑80%)  目标: 0.80  未达标
  Loss:      2.5 → 1.8 (↓28%)    趋势: 持续下降
  Agent 行为: 平均 2.3 轮对话, 工具调用成功率 78%
  主要错误:  22% 格式错误 (未正确调用 finish)

性能分析:
  总耗时:    3m12s
  LLM 推理:  72% (2m18s)  ← 主要瓶颈
  环境执行:  8% (15s)
  Logprob:   12% (23s)
  GRPO 训练: 8% (15s)
  吞吐量:    11.8 tok/s

调参建议:
  1. [高] num_epochs: 2 → 4     (reward 仍在上升，增加训练量)
  2. [高] num_problems: 64 → 128 (更多训练数据)
  3. [中] temperature: 0.7 → 0.6 (减少随机性，稳定输出)
  4. [低] max_agent_steps: 3 → 2 (加速推理，减少无效轮次)
```

### 分析 JSON（写入 analysis.json）

将分析结果写入 `rllm_trl/output/runs/<run_id>/analysis.json`：

```python
import json
analysis = {
    "run_id": "<run_id>",
    "round": 1,
    "reward": {"start": 0.25, "end": 0.45, "max": 0.48, "trend": "rising", "target": 0.8, "reached": False},
    "loss": {"start": 2.5, "end": 1.8, "trend": "falling"},
    "performance": {"total_time_s": 192, "llm_pct": 72, "env_pct": 8, "logprob_pct": 12, "train_pct": 8, "tok_per_s": 11.8},
    "suggestions": [
        {"param": "num_epochs", "old": 2, "new": 4, "priority": "high", "reason": "reward still rising"},
        {"param": "num_problems", "old": 64, "new": 128, "priority": "high", "reason": "more training data"},
    ]
}
with open("rllm_trl/output/runs/<run_id>/analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
```

## 独立使用

此 skill 可独立调用来分析任意已完成的训练：

```
/rllm-analyze <run_id>
```

如果未指定 run_id，自动查找最近一次训练的输出目录。
