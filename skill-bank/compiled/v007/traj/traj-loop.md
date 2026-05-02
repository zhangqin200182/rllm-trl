---
description: Top-level orchestration skill that drives the full trajectory optimization
  loop. Runs rllm-train, captures trajectories, segments, analyzes, and generates
  patches in an automated cycle.
metadata:
  categories:
  - trajectory
  - orchestration
  version: 1.0.0
name: traj-loop
---


# traj-loop — 全自动轨迹优化编排

你是轨迹优化的顶层编排者。你驱动完整的闭环: 训练 → 捕获 → 分割 → 分析 → 优化，循环指定轮次。每轮使用上一轮优化后的 skill 执行训练，实现 skill 的持续自动优化。

## 执行规则

1. 每个步骤必须通过调用对应的 skill 执行，不得内联
2. 唯一需要人工介入的环节是确认 patch（设计准则 3.4）
3. 每轮结束后输出本轮摘要，最终输出跨轮对比报告
4. 如果某轮训练失败，记录失败原因，继续下一轮（不中断循环）

## 执行步骤

### 0. 解析参数

从用户输入中提取:
- 训练描述（传给 rllm-train）
- 优化轮次（默认 3 轮）
- 执行模式（auto/approve，默认 approve）

示例输入:
```
/traj-loop "用 qwen-0.5b 训练数学 agent，自动优化 3 轮"
/traj-loop "qwen-0.5b, reward >= 0.8, 5 rounds, auto"
```

### 1. 循环执行

```
for round in 1..N:
    Step 1: 调用 /rllm-train 执行训练
            → hooks 自动捕获轨迹
    
    Step 2: 调用 /traj-segment 分割轨迹
    
    Step 3: 调用 /traj-analyze-rllm 分析轨迹
    
    Step 4: 调用 /traj-optimize 生成 patch
            → 展示 patch → 等待用户确认
            → 确认后编译更新 skill
    
    Step 5: 输出本轮摘要
```

### 2. Skill 调用方式

每个步骤使用 Skill 工具调用:
```
Skill("rllm-train", args="<训练描述>")
Skill("traj-segment")
Skill("traj-analyze-rllm")
Skill("traj-optimize")
```

调用 Skill 后当轮响应立即结束，等待下一轮系统注入。

### 3. 最终报告

所有轮次完成后，输出跨轮对比:

```
traj-loop 优化报告
==================
总轮次: {N}
训练描述: {description}

轮次对比:
  Round 1: reward {r1}  问题: {issues_1}  生成 patch: {patch_count_1}
  Round 2: reward {r2}  问题: {issues_2}  生成 patch: {patch_count_2}
  Round 3: reward {r3}  问题: {issues_3}  生成 patch: {patch_count_3}

优化效果:
  reward 变化: {r1} → {rN} ({improvement}%)
  累计 patch: {total_patches}
  优化的 skills: {skill_list}
```

## 状态管理

在 `trajectory/output/loop_state.json` 中维护循环状态:
```json
{
  "total_rounds": 3,
  "current_round": 1,
  "description": "...",
  "mode": "approve",
  "rounds": [
    {
      "round": 1,
      "run_id": "...",
      "reward": 0.45,
      "patches_generated": 2,
      "patches_accepted": 2
    }
  ]
}
```

支持中断后恢复: 读取 loop_state.json，从上次中断的 round 继续。
