---
description: Monitor rllm_trl training progress in real-time. Tracks reward trends,
  training speed, and detects anomalies like loss explosion or process crashes.
metadata:
  categories:
  - machine-learning
  - monitoring
  version: 1.0.0
name: rllm-monitor
---


# rllm-monitor — 训练过程监控

你负责实时监控 rllm_trl 训练进度，向用户汇报关键指标，并检测异常。

## 监控目标

- 日志文件: `rllm_trl/output/runs/<run_id>/training_log.txt`
- 性能统计: `rllm_trl/output/runs/<run_id>/perf_stats.json`（训练结束后生成）
- 轨迹文件: `rllm_trl/output/runs/<run_id>/trajectories/`（训练过程中逐步生成）

## 监控方式

### 实时监控（训练进行中）

使用 Monitor 工具监控训练日志：

```bash
tail -f rllm_trl/output/runs/<run_id>/training_log.txt | grep -E --line-buffered "/[0-9]+|···|Error|Traceback|FAILED|OOM|Training Report"
```

### 定期检查（训练进行中）

每隔一段时间读取日志文件尾部，提取关键指标：

```bash
tail -20 rllm_trl/output/runs/<run_id>/training_log.txt
```

### 训练日志格式

rllm_trl 的 TrainingLogger 输出格式（参考 `rllm_trl/logger.py`）：

进度行格式（每个 step 会输出多行子步骤 + 一行汇总）:
```
    ··· step 1/128: generating 4 trajectories...
    ··· trajectory 1/4 done (reward=1.000)
    ··· trajectory 2/4 done (reward=0.000)
    ··· trajectory 3/4 done (reward=1.000)
    ··· trajectory 4/4 done (reward=1.000)
  1/128     4    0.750      6.9s     88.4    29m57s
    ··· computing logprobs...
    ··· training update...
```

子步骤行以 `···` 开头，汇总行以 `step/total` 格式开头。

训练完成标志:
```
Training Report
==============
```

### Monitor 可靠性设计

#### 1. grep 模式通用化

当前（会失效）:
```bash
tail -f ... | grep -E "/64|···|Error"
```

改为（通用）:
```bash
tail -f ... | grep -E --line-buffered "^\s*[0-9]+/[0-9]+|···|Error|Traceback|OOM|Training Report|reward="
```

#### 2. 双重监控策略

主监控: Monitor 工具，persistent=true，做实时流式通知
备用监控: 当主监控连续 2 分钟无输出时，用 tail -20 读日志尾部

切换逻辑 (由 rllm-train 编排层管理):
  if 收到 Monitor 通知: `last_notification_time = now`
  if `now - last_notification_time > 120s`:
      执行 tail -20 检查训练状态
      if 训练仍在进行: 重启 Monitor (TaskStop 旧的 + 启动新的)
      if 训练已完成: 进入 Phase 5

#### 3. Monitor 生命周期管理

rllm-train 编排层增加:
  Phase 4 启动 Monitor 后，记录 monitor_task_id
  训练完成后: TaskStop(monitor_task_id) 清理
  Monitor 超时退出且训练未完成: 自动重启新 Monitor
  用户请求停止训练: 先 TaskStop(monitor_task_id)，再 kill 训练进程

#### 4. Monitor 健康检查

启动 Monitor 后 30s 内如果无任何输出:
  检查日志文件是否存在且在增长
  检查 grep 模式是否匹配到内容
  如果日志在增长但 grep 无匹配: 报告 grep 模式可能有误，切换到宽松模式

## 汇报内容

### 进度汇报（每 N 步或用户询问时）

```
训练进度 [第 1 轮]:
  进度:     Step 8/128 (6%)
  Reward:   0.750 (趋势: ↑ 从 0.25 开始)
  速度:     88.4 tok/s, 每步 ~7s
  已用时间: 1m45s
  预计剩余: 25m30s
```

### 异常检测（修订）

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Reward 归零 | 连续 5 步 reward=0 | 建议停止训练 (非仅报告) |
| Reward 崩溃 | reward 从 >0.3 降到 0 且持续 3 步 | 立即建议停止，诊断为 lr 过高或 forgetting |
| Loss 爆炸 | loss > 10 或 NaN/Inf | 立即建议停止 |
| OOM | "out of memory" | 立即建议停止 |
| 进程崩溃 | Traceback + 进程退出 | 报告错误 |
| 训练卡住 | 超过 120s 无输出 | 报告，检查进程状态 |

### Early Stopping 机制

Monitor 检测到以下条件时，向编排层发送 STOP 建议:

1. 连续 5 步 reward=0 且当前 step > total_steps * 0.2
   → "训练已崩溃，建议停止。连续 5 步 reward=0，继续训练不会恢复。"

2. Epoch 切换后 reward 断崖 (需要按 epoch 计算)
   → "进入 Epoch N 后 reward 从 X 降到 0，疑似 catastrophic forgetting，建议停止。"

3. 模型输出异常 (从 trajectory 文件检测)
   → "模型输出格式退化，不再使用 tool_call，建议停止。"

### Epoch 边界监控

计算 epoch 边界: `epoch_boundary = num_problems / (batch_size * grad_accum)`

当 step 跨越 epoch 边界时:
  读取最近 3 步的 reward
  与上一个 epoch 最后 3 步的 reward 对比
  如果下降 > 50%: 发出 catastrophic forgetting 预警

### Reward 峰值回落检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| Reward 峰值回落 | 当前 reward < 历史峰值 * 0.5 且持续 3 步 | 建议 early stopping |

实现: Monitor 维护 max_reward 变量，每步更新。当连续 3 步 reward < max_reward * 0.5 时发出 STOP 建议。

## 训练完成检测

训练完成的标志：
1. 后台任务正常退出（exit code 0）
2. 日志中出现 "Training Report" 字样
3. `perf_stats.json` 文件生成

训练完成后，读取最终统计并汇报：

```
训练完成 [第 1 轮]:
  总耗时:     3m12s
  最终 Reward: 0.45 (从 0.25 开始)
  Reward 趋势: 0.25 → 0.31 → 0.38 → 0.45
  总 Steps:    16
  平均速度:    11.8 tok/s
```

## 数据表面化准则

Hooks 只捕获 Claude Code 工具调用的 input/response。Monitor 工具的 grep 输出不被 PostToolUse hook 记录为完整事件。为确保训练关键数据进入轨迹系统，监控过程中必须用 Read/Bash 工具明确读取以下数据:

| 时机 | 操作 | 工具 | 捕获内容 |
|------|------|------|---------|
| 训练启动后 | 读取 config.json 完整内容 | Read | 训练配置（lr, epochs, problems 等） |
| 训练过程中 | 定期 tail training_log.txt 最后 30 行 | Bash | reward 趋势、step 进度 |
| 异常发生时 | 读取完整错误段 | Read/Bash | 错误上下文、Traceback |
| 训练结束时 | 读取 perf_stats.json | Read | 性能统计 |
| 训练结束时 | tail training_log.txt 最后 50 行 | Bash | 最终 Training Report |

这些 Read/Bash 调用是 Monitor grep 的必要补充，不是替代。Monitor 负责实时通知，Read/Bash 负责将完整数据带入对话供 hooks 捕获。
