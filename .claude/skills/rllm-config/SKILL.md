---
description: Generate or adjust TrainingConfig for rllm_trl agent RL training. Supports
  initial config generation from requirements and iterative hyperparameter tuning
  based on analysis results.
metadata:
  categories:
  - machine-learning
  - hyperparameter-tuning
  version: 1.0.0
name: rllm-config
---


# rllm-config — 训练配置生成与调参

你是 rllm_trl 训练配置专家。你有两个职责：根据需求生成初始配置，以及根据训练分析结果调整配置。

## 配置文件位置

- 配置定义: `rllm_trl/config.py` 中的 `TrainingConfig` dataclass
- 配置输出: `rllm_trl/output/runs/<run_id>/config.json`

## 模式一：初始配置生成

根据 rllm-clarify 阶段的需求摘要，生成 TrainingConfig 的 JSON 配置文件。

### 步骤

1. 读取 `rllm_trl/config.py` 确认当前 TrainingConfig 的所有字段和默认值
2. 根据需求摘要设置各参数
3. 根据运行环境（Mac CPU/MPS）调整参数：
   - batch_size 不超过 4（内存限制）
   - num_generations 不超过 8
   - 优先使用小模型（0.5B）做快速迭代
4. 生成 run_id（格式: `run_<timestamp>`）
5. 将配置写入 JSON 文件

### 初始配置推荐值

| 场景 | problems | epochs | lr | batch | generations |
|---|---|---|---|---|---|
| 快速测试 | 16 | 1 | 1e-5 | 2 | 2 |
| 标准训练 | 64 | 2 | 1e-5 | 2 | 4 |
| 深度训练 | 128-256 | 3-5 | 5e-6 | 2 | 4 |

### 难度配置

difficulty 参数控制训练数据的难度分布:
- `"simple"`: 100% 简单两数运算 (适合流程验证)
- `"hard"`: 100% 多步骤应用题 (0.5B 模型几乎无法学会)
- `"mixed"`: 80% simple + 20% hard (推荐，提供学习信号的同时引入挑战)

初始配置推荐:

| 场景 | difficulty | 原因 |
|------|-----------|------|
| 流程验证 | simple | 确认 pipeline 正常 |
| 正式训练 | mixed | 80/20 比例经验证有效 |
| 能力评估 | hard | 评估模型上限，不用于训练 |

调参时的难度调整:
- reward=1.0 + loss=0 → 题目太简单，切换到 mixed 或 hard
- reward<0.1 + difficulty=hard → 太难，切换到 mixed
- mixed 下 reward 在 0.3-0.7 → 比例合适，保持不变

## 模式二：调参优化

根据 rllm-analyze 阶段的分析结果，调整配置参数。

### 调参策略

读取上一轮的分析结果（`analysis.json`），根据以下规则调整：

**效果问题**:
| 症状 | 调整 | 原因 |
|---|---|---|
| reward 低 + loss 不降 | learning_rate ×2 或 num_generations +2 | 学习信号不足 |
| reward 低 + loss 降 | num_epochs +2 或 num_problems ×2 | 需要更多训练 |
| reward 震荡 | learning_rate ÷2, grad_accum_steps ×2 | 训练不稳定 |
| reward plateau | temperature ±0.1, 尝试不同 loss_type | 探索不足或过度 |
| reward 下降 | learning_rate ÷5, 回退到上一轮配置 | 过拟合或学习率过大 |

**性能问题**:
| 症状 | 调整 | 原因 |
|---|---|---|
| 训练太慢 | max_completion_length ÷2, max_agent_steps -1 | 减少生成长度 |
| 内存不足 | batch_size ÷2, num_generations ÷2 | 减少内存占用 |

### 参数联动约束（生成配置前必须验证）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   违反时: 自动调整 num_generations 为最近的合法值

2. **generation_batch_size 副作用**:
   `generation_batch_size = batch_size * gradient_accumulation_steps`
   当 grad_accum 增大时，每步生成的 trajectory 数量也增大
   影响: GRPO baseline 估计变化，训练动态改变
   建议: 调整 grad_accum 时同步说明对 generation_batch_size 的影响

3. **显存估算 (MPS)**:
   `estimated_mem = batch_size * num_generations * max_completion_length * model_params * 4`
   - 0.5B + batch=2 + gen=4 + len=256 ≈ 安全
   - 0.5B + batch=2 + gen=4 + len=512 ≈ 可能 OOM
   超出估算时: 自动降低 max_completion_length 或 num_problems

### 参数安全范围

#### 模型级别安全配置（硬约束）

生成配置时，必须根据模型大小查表，参数不得超出对应上限。

| 参数 | 0.5B 上限 | 1.5B 上限 | 3B 上限 | 依据 |
|------|----------|----------|--------|------|
| learning_rate | 1e-5 | 2e-5 | 5e-5 | 0.5B 在 2e-5 时策略崩溃 |
| num_epochs | 2 | 4 | 6 | 0.5B 在 4ep 时 catastrophic forgetting |
| max_completion_length | 256 | 512 | 512 | MPS 显存限制 |
| num_problems (MPS) | 32 | 32 | 16 | 配合 num_generations=4 的显存上限 |

调参建议超出上限时，必须警告并拒绝。例如：
  建议 num_epochs: 2 → 4
  → 检查: 0.5B 模型 epochs 上限为 2
  → 拒绝，改为建议: 增加 num_problems 或换 1.5B 模型

#### 通用参数范围

| 参数 | 最小值 | 最大值 | 说明 |
|---|---|---|---|
| learning_rate | 1e-7 | 1e-3 | 超出范围大概率不收敛 |
| temperature | 0.3 | 1.5 | 太低无探索，太高太随机 |
| num_generations | 2 | 8 | GRPO 至少需要 2 |
| batch_size | 1 | 4 | Mac 内存限制 |
| num_problems | 8 | 512 | 太少不够学，太多太慢 |
| num_epochs | 1 | 20 | 过多可能过拟合 |
| max_agent_steps | 1 | 8 | 影响生成长度和速度 |
| gradient_accumulation_steps | 1 | 16 | 等效增大 batch |

## 配置预检（生成配置后、启动前执行）

### 必检项（不通过则拒绝启动）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   失败时: 自动调整 num_generations

2. **模型安全上限**:
   查模型级别安全配置表，检查 lr/epochs/completion_length 是否超限
   失败时: 自动降到安全值并警告

3. **difficulty 合法性**:
   `difficulty in ("simple", "hard", "mixed")`
   失败时: 默认 "mixed"

### 建议检项（不通过则警告但允许启动）

4. **显存估算 (MPS)**:
   if `batch_size * num_generations * max_completion_length > 阈值`:
   警告: "可能 OOM，建议降低 max_completion_length 或 num_problems"

5. **训练时间估算**:
   `estimated_time = num_problems * num_epochs / (batch_size * grad_accum) * avg_step_time`
   if estimated_time > 30min:
   警告: "预计训练时间 Xm，确认继续？"

## 输出

生成配置后，用 Python 代码将配置写入 JSON 文件：

```python
python -c "
from rllm_trl.config import TrainingConfig
config = TrainingConfig(
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    num_problems=64,
    num_epochs=2,
    # ... 其他参数
)
config.to_json('rllm_trl/output/runs/<run_id>/config.json')
print(config.summary())
"
```

## 调参时的输出格式

```
配置调整（第 N 轮 → 第 N+1 轮）：
  learning_rate:  1e-5 → 2e-5  (reward 低但 loss 在降，加大学习率)
  num_epochs:     2 → 4        (reward 趋势向上，增加训练量)
  temperature:    0.7 → 0.6    (减少随机性，稳定输出)
  其他参数保持不变
```
