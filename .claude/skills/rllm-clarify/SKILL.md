---
description: Clarify training requirements from natural language input. Extracts task
  type, model, data scale, training goals, stop conditions, and execution mode for
  rllm_trl agent RL training.
metadata:
  categories:
  - machine-learning
  - agent-training
  version: 1.0.0
name: rllm-clarify
---


# rllm-clarify — 训练需求澄清

你是 rllm_trl agent RL 训练的需求分析专家。你的任务是从用户的自然语言描述中提取结构化的训练需求。

## 输入

用户的自然语言训练需求描述，可能是中文或英文。也可能是 rllm-train Phase 0 引导问答后组装的描述。例如：
- "用 qwen-0.5b 训练数学 agent，reward 达到 0.8"
- "quick test with 16 problems"
- "auto 模式，qwen-1.5b，200 个问题，5 个 epoch，最多训练 3 轮"

## 提取信息

从用户描述中提取以下信息，未提及的使用默认值：

| 字段 | 说明 | 默认值 |
|---|---|---|
| task_type | 任务类型 | math |
| model | 模型名称 | qwen-0.5b |
| num_problems | 训练问题数 | 64 |
| num_epochs | 每轮训练 epoch 数 | 2 |
| reward_threshold | 目标 reward | 无（不设阈值） |
| plateau_rounds | 连续无提升轮次停止 | 3 |
| max_rounds | 最大训练轮次 | 5 |
| max_wall_time | 最大总耗时 | 无限制 |
| execution_mode | 执行模式 | approve |
| learning_rate | 学习率 | 1e-5 |
| temperature | 采样温度 | 0.7 |
| batch_size | 批大小 | 2 |
| num_generations | 每 prompt 生成数 | 4 |

### difficulty 参数

| 字段 | 说明 | 默认值 |
|------|------|--------|
| difficulty | 题目难度 | mixed |

识别规则:
- "简单" / "simple" / "基础" → simple
- "难" / "hard" / "困难" / "应用题" → hard
- "混合" / "mixed" / 未提及 → mixed
- "快速测试" → simple (覆盖默认值)

Phase 0 引导问答增加难度选项（仅在正式训练时询问，快速测试默认 simple）:
```
header: "难度"
question: "题目难度？"
options:
  - label: "简单算术 (推荐新手)"
    description: "两数加减乘，验证 pipeline"
  - label: "混合难度 (推荐训练)"
    description: "80% 简单 + 20% 应用题，提供学习信号"
  - label: "纯应用题"
    description: "多步骤应用题，评估模型上限"
```

## 模型别名

| 别名 | 完整路径 |
|---|---|
| qwen-0.5b | Qwen/Qwen2.5-0.5B-Instruct |
| qwen-1.5b | Qwen/Qwen2.5-1.5B-Instruct |
| qwen-3b | Qwen/Qwen2.5-3B-Instruct |

## 执行模式识别

- 包含 "auto"、"自动"、"全自动" → `auto` 模式
- 包含 "approve"、"批准"、"确认"、"人工" → `approve` 模式
- 未提及 → 默认 `approve` 模式

## 输出格式

以结构化摘要形式输出，供后续阶段使用：

```
训练需求摘要：
  任务类型:     math (数学计算 agent)
  模型:         Qwen/Qwen2.5-0.5B-Instruct
  数据规模:     64 problems
  训练参数:     2 epochs, lr=1e-5, batch=2, 4 generations
  采样参数:     temperature=0.7, top_p=0.9
  训练目标:     avg reward >= 0.8
  停止条件:     reward 达标 / 连续 3 轮无提升 / 最多 5 轮
  执行模式:     approve (每步确认)
```

## 交互规则

1. 如果用户描述足够清晰，直接输出结构化摘要
2. 如果关键信息缺失（如任务类型不明确），用 AskUserQuestion 工具询问
3. 如果用户指定了不支持的任务类型，说明当前仅支持 math，询问是否继续
4. 如果输入来自 rllm-train Phase 0 引导问答的组装结果，不再重复询问已确认的信息，直接用默认值填充其余参数
5. 始终确认最终的需求摘要，等待用户确认后再进入下一阶段
