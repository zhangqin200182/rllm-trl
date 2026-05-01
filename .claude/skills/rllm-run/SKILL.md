---
description: Launch rllm_trl training process. Starts training in background using
  JSON config, records process info for monitoring.
metadata:
  categories:
  - machine-learning
  - agent-training
  version: 1.0.0
name: rllm-run
---


# rllm-run — 启动训练

你负责启动 rllm_trl 训练进程并确保它正常运行。

## 前置条件

- 配置文件已生成: `rllm_trl/output/runs/<run_id>/config.json`
- 工作目录: `/Users/kevin/code/MyProject`

## 启动流程

### 1. 验证配置

读取配置文件，确认关键参数合理：

```bash
python -c "
from rllm_trl.config import TrainingConfig
config = TrainingConfig.from_json('rllm_trl/output/runs/<run_id>/config.json')
print(config.summary())
"
```

### 2. 启动训练

使用 Bash 工具的 `run_in_background` 模式启动训练，将输出重定向到日志文件：

```bash
cd /Users/kevin/code/MyProject && python -m rllm_trl.run_training rllm_trl/output/runs/<run_id>/config.json 2>&1 | tee rllm_trl/output/runs/<run_id>/training_log.txt
```

### 3. 确认启动成功

启动后等待几秒，检查：
- 进程是否存活
- 日志文件是否开始写入
- 是否有启动错误（import error、CUDA error 等）

如果启动失败，读取日志文件诊断错误原因并报告。

## 输出

启动成功后报告：
```
训练已启动：
  Run ID:    <run_id>
  配置文件:  rllm_trl/output/runs/<run_id>/config.json
  日志文件:  rllm_trl/output/runs/<run_id>/training_log.txt
  后台任务:  <task_id>
```

## 错误处理

| 错误类型 | 处理方式 |
|---|---|
| ModuleNotFoundError | 提示安装依赖: pip install transformers trl |
| CUDA/MPS 错误 | 建议设置 CUDA_VISIBLE_DEVICES="" 使用 CPU |
| OOM | 建议减小 batch_size 和 num_generations |
| 配置文件不存在 | 提示先运行 rllm-config 生成配置 |
