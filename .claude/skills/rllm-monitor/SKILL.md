---
name: rllm-monitor
description: Monitor rllm_trl training progress in real-time. Tracks reward trends, training speed, and detects anomalies like loss explosion or process crashes.
metadata:
  version: "1.0.0"
  categories:
    - machine-learning
    - monitoring
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

### 异常检测

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Loss 爆炸 | loss > 10 或 loss = NaN/Inf | 立即报告，建议降低 lr |
| Reward 归零 | 连续 3 步 reward = 0 | 报告，可能是 env 或 reward 函数问题 |
| 进程崩溃 | 后台任务退出 + 日志含 Traceback | 报告错误信息 |
| OOM | 日志含 "out of memory" | 建议减小 batch_size |
| 训练卡住 | 超过 60s 无 `···` 或 step 行 | 报告，可能是死锁 |

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
