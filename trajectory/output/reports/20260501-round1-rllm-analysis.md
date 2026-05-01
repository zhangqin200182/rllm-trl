# rllm-train 轨迹分析报告

生成时间: 2026-05-01
分析范围: traj-loop Round 1
数据来源: 训练日志 (hooks 在训练启动后才配置，原始事件未捕获)

## 训练执行概览

| Run ID | 配置摘要 | 结果 | 关键问题 |
|--------|---------|------|---------|
| run_1777650398 | qwen-0.5b, lr=1e-5, 2ep, 64problems, mixed | reward 1.00→0.00 (Step 4→14, 中止) | catastrophic forgetting |

## Reward 趋势

```
Step  2: 0.250
Step  3: 0.500
Step  4: 1.000  ← 峰值
Step  5: 0.750
Step  6: 0.750
Step  7: 0.500
Step  8: 0.250
Step  9: 0.000  ← 开始崩溃
Step 10: 0.000
Step 11: 0.000
Step 12: 0.250  (短暂恢复)
Step 13: 0.000
Step 14: 0.000  ← early stop 触发
```

训练在 Step 15 被中止 (14/128 = 11% 进度)。

## 问题发现

### 1. Catastrophic Forgetting [影响: rllm-config]

**现象**: Reward 在 Step 4 达到峰值 1.00 后持续下降至 0，模型丧失了已学会的能力。

**证据**:
- Step 4 reward=1.00 (4/4 trajectories 成功)
- Step 8 reward=0.25 (仅 1/4 成功)
- Step 9-14 几乎全部 reward=0

**根因分析**:
- lr=1e-5 对 0.5B 模型偏高，初期快速学习但后期过拟合
- 2 epochs 对 64 problems 来说训练量过大，模型在 Epoch 1 中期就开始退化
- mixed 难度下 hard 题目 (20%) 可能干扰了 simple 题目的学习

**建议**:
- 降低 lr 到 5e-6
- 减少 epochs 到 1
- 或减少 num_problems 到 32 配合 2 epochs

### 2. Early Stopping 时机 [影响: rllm-monitor]

**现象**: 从 reward 峰值到触发 early stopping 经过了 10 步，浪费了约 6 分钟训练时间。

**建议**: 增加"reward 从峰值下降超过 50% 持续 3 步"的 early stopping 条件。

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 |
|--------|-----------|---------|--------|------|
| P0 | rllm-config | param-ranges | append | 0.5B 模型 lr 上限从 1e-5 降到 5e-6 |
| P0 | rllm-config | param-ranges | append | 0.5B 模型 epochs 上限从 2 降到 1 (64+ problems 时) |
| P1 | rllm-monitor | anomaly-detection | append | 增加 reward 峰值回落检测 |

## 建议的 Patch 内容

### Patch 1: rllm-config 参数安全范围收紧

针对 0.5B 模型，当 num_problems >= 64 时:
- lr 上限: 1e-5 → 5e-6
- epochs 上限: 2 → 1
- 推荐配置: lr=5e-6, epochs=1, 64 problems

### Patch 2: rllm-monitor reward 峰值回落检测

增加检测条件:
- 记录训练过程中的 reward 峰值
- 当当前 reward < 峰值 * 0.5 且持续 3 步时，发出 early stopping 建议
