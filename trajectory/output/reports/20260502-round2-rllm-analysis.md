# rllm-train 轨迹分析报告 — Round 2

生成时间: 2026-05-02
分析范围: traj-loop Round 2
数据来源: 训练日志

## 训练执行概览

| Run ID | 配置摘要 | 结果 | 关键问题 |
|--------|---------|------|---------|
| run_1777651915 | qwen-0.5b, lr=5e-6, 1ep, 64problems, mixed | reward 0.75→0.00 (Step 1→16, 中止) | forgetting (延迟但未消除) |

## 配置变更 (Round 1 → Round 2)

| 参数 | Round 1 | Round 2 | 效果 |
|------|---------|---------|------|
| learning_rate | 1e-5 | 5e-6 | 崩溃延迟 4 步 (step 5→9) |
| num_epochs | 2 | 1 | 减少总训练量 |

## Reward 趋势

```
Step  1: 0.750
Step  2: 1.000  ← 峰值开始
Step  3: 0.500
Step  4: 1.000
Step  5: 1.000
Step  6: 1.000
Step  7: 1.000
Step  8: 1.000  ← 峰值结束 (连续 5 步 reward=1.0)
Step  9: 0.000  ← 开始崩溃
Step 10: 0.000
Step 11: 0.500  (短暂恢复)
Step 12: 0.000
Step 13: 0.000
Step 14: 0.000
Step 15: 0.000
Step 16: 0.000  ← early stop 触发
```

训练在 Step 16 被中止 (16/64 = 25% 进度)。

## 问题发现

### 1. Forgetting 延迟但未消除 [影响: rllm-config]

**现象**: lr=5e-6 将崩溃点从 Round 1 的 step 5 延迟到 step 9，但 64 problems mixed 难度下仍然发生 forgetting。

**证据**:
- Steps 2-8: 连续高 reward (avg 0.93)
- Steps 9-16: 几乎全零 (avg 0.06)
- 崩溃模式与 Round 1 相同，只是延迟了

**根因分析**:
- 64 problems 对 0.5B 模型仍然过多，即使 lr=5e-6 + 1 epoch
- mixed 难度中 20% hard 问题可能在训练后期干扰模型
- 0.5B 模型容量有限，64 problems 的多样性超出其学习能力

**建议 (Round 3)**:
- 方案 A: 减少 num_problems 到 32，保持 lr=5e-6, epochs=1
- 方案 B: 切换 difficulty 到 simple，保持 64 problems
- 方案 C: 进一步降低 lr 到 2e-6，保持其他不变
- 推荐方案 A: 减少数据量是最直接的解决方式

### 2. Early Stopping 时机改善 [影响: rllm-monitor]

Round 2 的 early stopping 在 step 16 触发 (崩溃后 7 步)，比 Round 1 (崩溃后 10 步) 有改善，但仍有优化空间。step 11 的短暂恢复 (0.5) 干扰了连续零检测。

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 |
|--------|-----------|---------|--------|------|
| P0 | rllm-config | param-ranges | append | 0.5B + mixed 难度时 num_problems 上限从 64 降到 32 |
| P1 | rllm-config | param-ranges | append | 或者: 64 problems 时强制 difficulty=simple |

## Round 3 推荐配置

```
model: Qwen/Qwen2.5-0.5B-Instruct
num_problems: 32
difficulty: mixed
num_epochs: 1
learning_rate: 5e-6
batch_size: 2
num_generations: 4
```

预期: 32 problems 减少训练步数到 32 步 (vs 64)，降低 forgetting 风险。
