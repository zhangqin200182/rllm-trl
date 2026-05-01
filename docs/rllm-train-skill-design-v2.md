# rllm_trl Skill 系统补充设计：基于实战训练的优化

> 本文档是对 `rllm-train-skill-design.md` 的补充。基于 2026-04-30 的 8 轮训练实验（Qwen2.5-0.5B-Instruct, Mac MPS），记录实际暴露的问题和对应的 Skill 优化方案。

## 实验背景

| 轮次 | Run ID | 关键配置 | 结果 | 暴露的问题 |
|------|--------|----------|------|-----------|
| 1 | run_1777465401 | simple, 64p, 2ep | reward=1.0 | 题目太简单，无学习信号 |
| 2 | run_1777511359 | simple, 16p, 1ep | reward=1.0 | 同上 |
| 3 | run_1777512419 | hard, 16p, 1ep | reward=0.14 | 纯难题太难，无正向信号 |
| 4 | run_1777513160 | mixed, gen=6 | 启动失败 | num_generations 整除约束 |
| 4b | run_1777513256 | mixed, 512len | OOM | MPS 显存不足 |
| 4c | run_1777514924 | mixed, 64p | OOM | 同上 |
| 5 | run_1777516127 | mixed(80/20), 32p, 2ep | reward=0.50 | 最佳结果 |
| 6 | run_1777516933 | lr=2e-5, 4ep | reward→0 | lr 过高导致策略崩溃 |
| 7 | run_1777521505 | ga=4, 4ep | reward→0 | grad_accum 副作用 |
| 8 | run_1777530664 | 4ep, ga=2 | reward→0 | catastrophic forgetting |

---

## 一、rllm-config 优化：参数安全与联动约束

### 1.1 问题：参数安全范围过宽

当前设计的参数范围：
```
learning_rate: 1e-7 ~ 1e-3
num_epochs:    1 ~ 20
```

实际验证结果：0.5B 模型在 lr=2e-5 时策略崩溃（run_1777516933），在 epochs=4 时 catastrophic forgetting（run_1777530664）。

**优化：增加模型级别安全配置表**

```
## 模型级别安全配置（硬约束）

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
```

### 1.2 问题：参数间联动约束缺失

run_1777513160 因 `num_generations=6` 启动失败：
```
ValueError: generation_batch_size (4) must be divisible by num_generations (6)
```

run_1777521505 因 `grad_accum=4` 改变了 `generation_batch_size`，导致训练动态异常。

**优化：增加参数联动约束检查**

```
## 参数联动约束（生成配置前必须验证）

1. TRL 整除约束:
   assert (batch_size * gradient_accumulation_steps) % num_generations == 0
   违反时: 自动调整 num_generations 为最近的合法值

2. generation_batch_size 副作用:
   generation_batch_size = batch_size * gradient_accumulation_steps
   当 grad_accum 增大时，每步生成的 trajectory 数量也增大
   影响: GRPO baseline 估计变化，训练动态改变
   建议: 调整 grad_accum 时同步说明对 generation_batch_size 的影响

3. 显存估算 (MPS):
   estimated_mem = batch_size * num_generations * max_completion_length * model_params * 4
   0.5B + batch=2 + gen=4 + len=256 ≈ 安全
   0.5B + batch=2 + gen=4 + len=512 ≈ 可能 OOM
   超出估算时: 自动降低 max_completion_length 或 num_problems
```

### 1.3 问题：缺少 difficulty 参数

今天最关键的发现是难度配比决定训练成败，但原设计完全没有 difficulty 参数。

**优化：增加 difficulty 配置指导**

```
## 难度配置

difficulty 参数控制训练数据的难度分布:
- "simple": 100% 简单两数运算 (适合流程验证)
- "hard": 100% 多步骤应用题 (0.5B 模型几乎无法学会)
- "mixed": 80% simple + 20% hard (推荐，提供学习信号的同时引入挑战)

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
```

---

## 二、rllm-analyze 优化：决策树补充与 Epoch 分段分析

### 2.1 问题：决策树缺少关键分支

当前决策树只处理 "reward 未达标" 的情况，缺少以下场景：

**场景 A: reward=1.0，无学习信号**（run_1777465401, run_1777511359）
- reward=1.0, loss=0, grad_norm=0
- 模型已经会了，训练没有意义

**场景 B: Catastrophic forgetting**（run_1777530664）
- Epoch 1 reward > 0.3, Epoch 2 reward < 0.1
- 模型在多 epoch 训练中"忘记"了已学会的能力

**场景 C: 格式退化**（run_1777530664 step 40）
- 模型从 `<tool_call>` 格式退化为纯文本输出
- 如 "1097 + 38 = 11015"（既没用工具，数字也错）

**优化：扩展决策树**

```
## 扩展调参决策树

reward 已达标 (=1.0)?
├── loss=0, grad=0 → 题目太简单
│   └── 建议: difficulty 切换到 mixed 或 hard
│       不要调其他参数，问题不在超参而在数据
└── loss>0 → 正常，训练有效

reward 未达标?
├── (原有分支保持不变)
├── 新增: Epoch 间断崖式下降
│   └── Epoch N avg_reward > 0.3 且 Epoch N+1 avg_reward < 0.1
│       → catastrophic forgetting
│       → 建议: 减少 epochs (不超过模型安全上限)
│       → 如果 epochs 已经是 1-2，建议换更大模型
└── 新增: 格式退化
    └── 后半段 tool_call 使用率 < 前半段的 50%
        → 模型"忘记"了工具调用格式
        → 建议: 减少 epochs，或增加格式正确性辅助 reward
```

### 2.2 问题：缺少 Epoch 分段分析

当前分析只看整体 reward 趋势（start → end），无法发现 catastrophic forgetting。

**优化：增加 Epoch 分段分析**

```
## Epoch 分段分析（新增分析维度）

将 per_step_rollouts 按 epoch 切分:
  steps_per_epoch = total_steps / num_epochs
  epoch_rewards = [avg(rewards[i*spe : (i+1)*spe]) for i in range(num_epochs)]

检测规则:
  if epoch_rewards[i+1] < epoch_rewards[i] * 0.3:
      → 标记为 "catastrophic forgetting at epoch {i+1}"
      → 建议: 减少 epochs，当前模型容量不足以支撑多 epoch 训练

输出示例:
  Epoch 分析:
    Epoch 1: avg_reward=0.45, tool_call_rate=85%
    Epoch 2: avg_reward=0.02, tool_call_rate=12%  ← 断崖下降
    诊断: catastrophic forgetting
    建议: num_epochs 不超过 1-2 (0.5B 模型限制)
```

### 2.3 问题：Agent 行为分析太粗

**优化：增加格式退化检测**

```
## 格式退化检测（新增分析维度）

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
```

---

## 三、rllm-monitor 优化：异常检测与 Early Stopping

### 3.1 问题：异常阈值太保守

当前 "连续 3 步 reward=0" 才报告。实际 run_1777516933 从 step 5 开始 reward=0，一直跑到 step 51 才被人工停止，浪费了 46 步的计算。

**优化：调整异常阈值 + 增加 Early Stopping**

```
## 异常检测规则（修订）

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| Reward 归零 | 连续 5 步 reward=0 | 建议停止训练 (非仅报告) |
| Reward 崩溃 | reward 从 >0.3 降到 0 且持续 3 步 | 立即建议停止，诊断为 lr 过高或 forgetting |
| Loss 爆炸 | loss > 10 或 NaN/Inf | 立即建议停止 |
| OOM | "out of memory" | 立即建议停止 |
| 进程崩溃 | Traceback + 进程退出 | 报告错误 |
| 训练卡住 | 超过 120s 无输出 | 报告，检查进程状态 |

## Early Stopping 机制（新增）

Monitor 检测到以下条件时，向编排层发送 STOP 建议:

1. 连续 5 步 reward=0 且当前 step > total_steps * 0.2
   → "训练已崩溃，建议停止。连续 5 步 reward=0，继续训练不会恢复。"

2. Epoch 切换后 reward 断崖 (需要按 epoch 计算)
   → "进入 Epoch N 后 reward 从 X 降到 0，疑似 catastrophic forgetting，建议停止。"

3. 模型输出异常 (从 trajectory 文件检测)
   → "模型输出格式退化，不再使用 tool_call，建议停止。"
```

### 3.2 问题：缺少 Epoch 边界监控

**优化：增加 Epoch 边界检查**

```
## Epoch 边界监控（新增）

计算 epoch 边界: epoch_boundary = num_problems / (batch_size * grad_accum)

当 step 跨越 epoch 边界时:
  读取最近 3 步的 reward
  与上一个 epoch 最后 3 步的 reward 对比
  如果下降 > 50%: 发出 catastrophic forgetting 预警
```

---

## 四、rllm-monitor 优化：监控自身可靠性

### 4.1 问题：Monitor 静默失效

今天用户连续问了 "为什么不持续监控进度了？" 和 "看不到 monitor 进展通知了？"。Monitor 工具超时退出后，skill 没有恢复机制，训练继续跑但完全失去可见性。

### 4.2 问题：grep 模式硬编码

Monitor 的 grep 用了 `/64` 这样的硬编码总步数，当实际训练步数不同时匹配失败，Monitor 看起来"没有输出"。

### 4.3 问题：长训练 vs Monitor 生命周期不匹配

部分训练跑了 20+ 分钟，但 Monitor 默认 timeout 5 分钟，一次 Monitor 覆盖不了整个训练。

**优化方案**

```
## Monitor 可靠性设计（新增章节）

### 1. grep 模式通用化

当前（会失效）:
  tail -f ... | grep -E "/64|···|Error"

改为（通用）:
  tail -f ... | grep -E --line-buffered "^\s*[0-9]+/[0-9]+|···|Error|Traceback|OOM|Training Report|reward="

### 2. 双重监控策略

主监控: Monitor 工具，persistent=true，做实时流式通知
备用监控: 当主监控连续 2 分钟无输出时，用 tail -20 读日志尾部

切换逻辑 (由 rllm-train 编排层管理):
  if 收到 Monitor 通知:
      last_notification_time = now
  if now - last_notification_time > 120s:
      执行 tail -20 检查训练状态
      if 训练仍在进行:
          重启 Monitor (TaskStop 旧的 + 启动新的)
      if 训练已完成:
          进入 Phase 5

### 3. Monitor 生命周期管理

rllm-train 编排层增加:
  Phase 4 启动 Monitor 后，记录 monitor_task_id
  训练完成后: TaskStop(monitor_task_id) 清理
  Monitor 超时退出且训练未完成: 自动重启新 Monitor
  用户请求停止训练: 先 TaskStop(monitor_task_id)，再 kill 训练进程

### 4. Monitor 健康检查

rllm-monitor SKILL.md 增加自检逻辑:
  启动 Monitor 后 30s 内如果无任何输出:
    检查日志文件是否存在且在增长
    检查 grep 模式是否匹配到内容
    如果日志在增长但 grep 无匹配: 报告 grep 模式可能有误，切换到宽松模式
```

---

## 五、rllm-train 优化：训练中止与错误恢复

### 5.1 问题：缺少训练中止机制

当前流程是 Phase 3→4→5 线性执行，没有从 Phase 4 直接跳到"中止+分析"的路径。今天多次需要手动"停掉吧"。

**优化：增加 Phase 4.5 训练中止**

```
## Phase 4.5: 训练中止（新增）

触发条件 (任一):
  - Monitor 发出 STOP 建议 (early stopping)
  - 用户主动要求停止
  - 训练进程崩溃 (OOM, Traceback)

执行步骤:
  1. TaskStop 训练后台任务
  2. TaskStop Monitor 任务
  3. 等待 5s 确认进程退出
  4. 读取已生成的 trajectory 文件和日志
  5. 进入 Phase 5 分析（即使训练未完成，也分析已有数据）

中止后的分析要点:
  - 标记 analysis.json 中 "completed": false, "abort_reason": "..."
  - 分析崩溃前的 reward 趋势
  - 如果是 early stopping: 诊断崩溃原因并给出针对性调参建议
  - 如果是用户中止: 保存状态，支持后续恢复
```

### 5.2 问题：错误恢复表太简单

当前只有 4 行（崩溃/OOM/连续失败/用户中断），但今天遇到了更多场景。

**优化：扩展错误恢复表**

```
## 错误恢复策略（修订）

| 场景 | 检测方式 | 恢复策略 |
|------|---------|---------|
| OOM | "out of memory" | 自动: max_completion_length ÷2, 如仍 OOM 则 num_problems ÷2 |
| num_generations 不整除 | ValueError 启动失败 | 自动: 调整 num_generations 为最近合法值 |
| lr 过高致策略崩溃 | reward 从 >0 骤降到 0 且不恢复 | 自动: lr ÷2, 重新训练 |
| catastrophic forgetting | Epoch N+1 reward < Epoch N * 0.3 | 自动: epochs 设为当前 epoch 数 -1, 重新训练 |
| grad_accum 副作用 | 训练从第 1 步就 reward=0 | 建议: 回退 grad_accum 到上一轮值 |
| 格式退化 | tool_call 使用率后期 < 前期 50% | 建议: 减少 epochs, 或增加格式辅助 reward |
| 进程崩溃 (Traceback) | 日志含 Traceback | 读取错误信息，诊断后调整配置重试 |
| 连续 2 轮失败 | history 中连续 2 轮 reward 未提升 | 暂停，向用户报告，建议换模型或调整任务 |
```

### 5.3 问题：未明确 base model 重载机制

今天用户问"权重是不是已经训坏了？"，说明对训练机制有误解。

**优化：在编排层明确说明**

```
## 训练机制说明（新增，Phase 3 和 Phase 6 中展示）

每轮训练都从 base model (如 Qwen2.5-0.5B-Instruct) 重新加载权重。
上一轮的训练结果不会影响下一轮的初始权重。
调参循环改变的是训练配置（lr, epochs, difficulty 等），不是模型起点。

在 Phase 3 启动训练时提示:
  "第 N 轮训练: 从 base model 重新开始 (不继承上一轮权重)"

在 Phase 6 最终报告中说明:
  "每轮训练独立从 base model 开始，最终模型来自第 N 轮的训练结果"
```

---

## 六、rllm-config + rllm-run 优化：配置预检

### 6.1 问题：可预防的启动失败

run_1777513160 因整除约束启动失败，run_1777513256/run_1777514924 因 OOM 在训练中途崩溃。这些都可以在启动前检测到。

**优化：增加配置预检步骤**

```
## 配置预检（rllm-config 生成配置后、rllm-run 启动前执行）

### 必检项（不通过则拒绝启动）

1. TRL 整除约束:
   (batch_size * gradient_accumulation_steps) % num_generations == 0
   失败时: 自动调整 num_generations

2. 模型安全上限:
   查模型级别安全配置表，检查 lr/epochs/completion_length 是否超限
   失败时: 自动降到安全值并警告

3. difficulty 合法性:
   difficulty in ("simple", "hard", "mixed")
   失败时: 默认 "mixed"

### 建议检项（不通过则警告但允许启动）

4. 显存估算 (MPS):
   if batch_size * num_generations * max_completion_length > 阈值:
       警告: "可能 OOM，建议降低 max_completion_length 或 num_problems"

5. 训练时间估算:
   estimated_time = num_problems * num_epochs / (batch_size * grad_accum) * avg_step_time
   if estimated_time > 30min:
       警告: "预计训练时间 Xm，确认继续？"
```

---

## 七、rllm-clarify 优化：增加 difficulty 参数采集

### 7.1 问题：需求澄清阶段没有采集难度信息

**优化：提取信息表增加 difficulty**

```
## 提取信息（修订）

增加:
| 字段 | 说明 | 默认值 |
|------|------|--------|
| difficulty | 题目难度 | mixed |

识别规则:
- "简单" / "simple" / "基础" → simple
- "难" / "hard" / "困难" / "应用题" → hard
- "混合" / "mixed" / 未提及 → mixed
- "快速测试" → simple (覆盖默认值)

Phase 0 引导问答增加难度选项（仅在正式训练时询问，快速测试默认 simple）:
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

---

## 优先级排序

| 优先级 | 优化项 | 影响的 Skill | 依据 |
|--------|--------|-------------|------|
| **P0** | 模型级别安全配置表 | rllm-config | 避免 lr/epochs 超出安全范围 (run_1777516933, run_1777530664) |
| **P0** | 参数联动约束检查 | rllm-config | 避免启动失败 (run_1777513160) |
| **P0** | Monitor 可靠性（通用 grep + 自保活 + 双重监控） | rllm-monitor, rllm-train | 多次 Monitor 静默失效，用户投诉 |
| **P0** | Early Stopping 机制 | rllm-monitor, rllm-train | 避免在已崩溃训练上浪费时间 (run_1777516933 浪费 46 步) |
| **P1** | Epoch 分段分析 + catastrophic forgetting 检测 | rllm-analyze | 今天最核心的技术发现 (run_1777530664) |
| **P1** | difficulty 参数全链路支持 | rllm-clarify, rllm-config | 难度配比是训练成败的关键 |
| **P1** | Phase 4.5 训练中止 + 扩展错误恢复 | rllm-train | 当前缺少从失败中恢复的标准流程 |
| **P1** | 配置预检（整除 + 显存估算） | rllm-config, rllm-run | 可预防的启动失败 (run_1777513160, run_1777513256) |
| **P2** | 决策树补充（reward 满分 / 格式退化） | rllm-analyze | 覆盖更多实际场景 |
| **P2** | 明确 base model 重载机制 | rllm-train | 减少用户困惑 |

---

## 实施建议

1. **P0 项可以立即修改 SKILL.md**，不需要改 Python 代码。主要是在 skill 指令中增加检查规则和安全表。

2. **P1 中的 epoch 分段分析和 difficulty 支持**需要同时修改 SKILL.md 和 Python 代码（`math_env.py` 已支持 difficulty，`perf_stats.json` 需要增加 epoch 标记）。

3. **P2 项属于锦上添花**，可以在下一轮训练实验后根据新发现再调整。

4. 所有优化都应该先在 SKILL.md 中以指令形式实现，验证有效后再考虑是否需要固化到 Python 代码中。
