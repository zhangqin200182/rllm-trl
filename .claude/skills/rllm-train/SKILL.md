---
name: rllm-train
description: End-to-end automated agent RL training with rllm_trl. Orchestrates requirement clarification, config generation, training execution, monitoring, result analysis, and iterative hyperparameter tuning until training goals are met. Supports auto and approve execution modes.
metadata:
  version: "2.0.0"
  categories:
    - machine-learning
    - agent-training
    - automation
---

# rllm-train — 自动训练主编排

你是 rllm_trl agent RL 训练的全流程编排者。你负责串联需求澄清、配置生成、训练执行、过程监控、结果分析、调参优化的完整闭环，循环直到训练目标达成。

## 执行规则（必须遵守）

1. **每个 Phase 必须通过调用对应的子 skill 来执行**，不得跳过子 skill 直接执行其内部逻辑
2. 编排者（你）只负责: Phase 间的流转控制、数据传递、停止条件判断、状态追踪、Phase 0 引导问答、Phase 6 最终报告
3. 编排者不负责: 具体的需求解析、配置生成、训练启动、日志监控、结果分析 — 这些全部委托给子 skill
4. 禁止"内联执行" — 即使你知道子 skill 的逻辑，也必须通过下面的调用方式执行，不得自己手动操作（如直接写 config.json、直接读 config.py 解析参数）
5. **Skill 调用后立即停止** — 调用 `Skill("rllm-xxx")` 后，当轮响应必须立即结束，不得在同一轮响应中跟随任何 Bash、Read、Write、Edit 等工具调用。原因: Skill 工具是异步的，系统会在下一轮消息中注入 SKILL.md 内容，只有等到注入完成后才能按 SKILL.md 的步骤执行。如果在同一轮就开始执行操作，等于绕过了 skill 的注入流程，违反了规则 1 和 4

## 子 skill 调用方式

每个 Phase 调用子 skill 时，按以下优先级执行：

1. **首选: Skill 工具** — 使用 `Skill("rllm-xxx", args="...")` 调用。如果该 skill 出现在可用 skill 列表中，必须用此方式。**调用后当轮响应立即结束，等待下一轮系统注入 SKILL.md 后再执行步骤**
2. **备选: Read + 执行** — 如果 Skill 工具调用失败或 skill 不在可用列表中，则:
   - 用 Read 工具读取 `.claude/skills/rllm-xxx/SKILL.md`
   - 严格按照 SKILL.md 中描述的步骤逐步执行
   - 不得省略、合并或跳过 SKILL.md 中的任何步骤

## 数据传递契约

Phase 之间通过以下方式传递数据：

```
Phase 0 → Phase 1: 组装的自然语言描述（如"用 qwen-0.5b 训练数学 agent，reward 达到 0.5"）
Phase 1 → Phase 2: 需求摘要文本（包含模型、目标、所有参数、停止条件）
Phase 2 → Phase 3: config.json 文件路径 (rllm_trl/output/runs/<run_id>/config.json)
Phase 3 → Phase 4: 后台任务 ID + 日志文件路径 (rllm_trl/output/runs/<run_id>/training_log.txt)
Phase 4 → Phase 5: 训练完成确认 + run_id
Phase 5 → Phase 2（循环）: analysis.json 路径 (rllm_trl/output/runs/<run_id>/analysis.json)
```

## 工作目录

`/Users/kevin/code/MyProject`

## 整体流程

```
Phase 0: 输入分级与引导 (编排者自己执行)
    ↓
Phase 1: 需求澄清 → 调用 rllm-clarify
    ↓
Phase 2: 配置生成 → 调用 rllm-config
    ↓
Phase 3-5: 训练循环
    ├→ 启动训练 → 调用 rllm-run
    ├→ 过程监控 → 调用 rllm-monitor
    ├→ 结果分析 → 调用 rllm-analyze
    └→ 判断是否达成 → 未达成则调用 rllm-config 调参并重新训练
    ↓
Phase 6: 最终报告 (编排者自己执行)
```

## 执行模式

### approve 模式（默认）

在每个关键决策点暂停等待用户确认：
1. 需求摘要确认
2. 初始配置确认
3. 每轮训练结束后的调参方案确认
4. 停止训练确认

### auto 模式

全自动执行，仅在以下情况暂停：
- 训练出错需要人工干预
- 达到停止条件
- 连续 2 轮调参后 reward 无改善（可能需要人工介入）

用户可通过以下方式指定模式：
- "auto 模式" / "全自动" / "自动执行" → auto
- "approve 模式" / "人工确认" / "每步确认" → approve
- 未指定 → 默认 approve

## 详细执行步骤

### Phase 0: 输入分级与引导（编排者自己执行）

收到用户输入后，先判断信息完整度，决定走哪条路径。

#### 分级规则

检查用户输入中是否包含以下关键信息：

| 关键信息 | 识别标志 | 权重 |
|---|---|---|
| 模型 | qwen/Qwen/模型名/0.5b/1.5b/3b | 必要 |
| 训练目标 | reward/目标/达到/>=/准确率 | 必要 |
| 数据规模 | N 个问题/problems/题 | 可选 |
| 执行模式 | auto/approve/自动/确认 | 可选 |

- **充分**（含模型 + 训练目标）→ 输出"正在解析训练需求..."，直接进入 Phase 1
- **部分**（含其中之一）→ 输出"收到，还需要补充一些信息："，用 AskUserQuestion 补缺失项，然后进入 Phase 1
- **模糊**（两者都不含，如"启动训练"、"开始"、"跑一下"、空输入）→ 输出"好的，先确认几个关键参数："，用引导问答收集信息，然后进入 Phase 1

#### 模糊输入引导问答

用一次 AskUserQuestion 同时问 2 个问题：

问题 1 — 模型选择：
```
header: "模型"
question: "用哪个模型训练？"
options:
  - label: "qwen-0.5b (推荐)"
    description: "最小最快，适合快速实验和验证流程"
  - label: "qwen-1.5b"
    description: "中等大小，效果和速度的平衡点"
  - label: "qwen-3b"
    description: "最大，效果最好但训练最慢"
```

问题 2 — 训练目标：
```
header: "目标"
question: "训练到什么程度？"
options:
  - label: "快速测试 (reward >= 0.5)"
    description: "验证流程是否跑通，几分钟完成"
  - label: "标准训练 (reward >= 0.8)"
    description: "正式训练，追求较好效果"
  - label: "充分训练 (reward >= 0.95)"
    description: "追求高准确率，耗时较长"
```

收到回答后，组装成完整描述（如"用 qwen-0.5b 训练数学 agent，reward 达到 0.5"），交给 Phase 1 处理。

#### 部分输入补充

只问缺失的那一个问题，不重复问已知信息。

### Phase 1: 需求澄清

**调用子 skill: rllm-clarify**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-clarify", args="<Phase 0 组装的完整描述>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-clarify/SKILL.md`，然后严格按其步骤执行

输入: Phase 0 组装的自然语言描述
输出: 结构化的需求摘要（包含模型、目标、所有参数、停止条件）
完成标志: 输出了格式化的需求摘要

⚠️ 禁止跳过 rllm-clarify，直接从用户输入中提取参数。

### Phase 2: 配置生成

**调用子 skill: rllm-config**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-config", args="初始配置 | <Phase 1 的需求摘要>")`
   - 如果是调参循环（非首轮），args 改为: `"调参 | run_id=<run_id> | <Phase 5 的调参建议>"`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-config/SKILL.md`，然后严格按其步骤执行

输入: 需求摘要（首轮）或 analysis.json 中的调参建议（后续轮）
输出: config.json 文件路径
完成标志: `rllm_trl/output/runs/<run_id>/config.json` 已生成

在 approve 模式下，展示配置摘要并等待用户确认后再进入 Phase 3。

⚠️ 禁止跳过 rllm-config，直接写 config.json 或直接调用 TrainingConfig。

### Phase 3: 启动训练

**调用子 skill: rllm-run**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-run", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-run/SKILL.md`，然后严格按其步骤执行

输入: run_id（从 Phase 2 的 config.json 路径中提取）
输出: 后台任务 ID + 日志文件路径
完成标志: 训练进程已启动，日志文件开始写入

⚠️ 禁止跳过 rllm-run，直接用 Bash 启动 python 训练命令。

### Phase 4: 过程监控

**调用子 skill: rllm-monitor**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-monitor", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-monitor/SKILL.md`，然后严格按其步骤执行

输入: run_id + 后台任务 ID
输出: 训练完成确认（正常完成 / 异常退出）
完成标志: 训练进程退出 或 日志中出现 "Training Report"

⚠️ 禁止跳过 rllm-monitor，直接 tail 日志或轮询进程状态。

### Phase 5: 结果分析与调参

**调用子 skill: rllm-analyze**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-analyze", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-analyze/SKILL.md`，然后严格按其步骤执行

输入: run_id
输出: 分析报告 + 调参建议（写入 analysis.json）
完成标志: `rllm_trl/output/runs/<run_id>/analysis.json` 已生成

⚠️ 禁止跳过 rllm-analyze，直接读取日志文件分析 reward 趋势。

**Phase 5 后的编排决策（编排者自己执行）：**

1. 读取 analysis.json 中的 `reward.reached` 字段
2. 判断停止条件（见下方"停止条件判断"）
3. 如果达成目标 → 进入 Phase 6
4. 如果未达成:
   - 更新 training_state.json
   - 在 approve 模式下展示调参建议并等待确认
   - 回到 Phase 2，传入 analysis.json 的调参建议，调用 rllm-config 生成新配置
   - 然后继续 Phase 3 → 4 → 5 循环

### Phase 6: 最终报告（编排者自己执行）

训练目标达成（或达到停止条件）后，输出最终报告：

```
训练完成报告
============
目标:       avg reward >= <target>
结果:       avg reward = <final> ✓/✗

训练历程:
  第 1 轮: reward <start> → <end>  配置: <key params>
  第 2 轮: reward <start> → <end>  调参: <changes>
  ...

总耗时:     <time> (<N> 轮训练)
最终模型:   rllm_trl/output/runs/<run_id>/final_model/
所有记录:   rllm_trl/output/runs/<run_id>/
```

## 停止条件判断

每轮训练结束后（Phase 5 完成后），编排者检查以下条件（按优先级）：

1. **reward_threshold**: 最终 avg reward >= 目标值 → 成功停止
2. **max_rounds**: 已达最大轮次 → 停止（可能未达标）
3. **max_wall_time**: 总耗时超限 → 停止（可能未达标）
4. **plateau_rounds**: 连续 N 轮 reward 提升 < 5% → 停止（plateau）
5. **reward 下降**: 连续 2 轮 reward 下降 → 警告，建议停止

## 状态追踪

在训练循环中维护以下状态（编排者自己管理）：

```json
{
  "round": 1,
  "history": [
    {"round": 1, "run_id": "run_xxx", "reward_start": 0.25, "reward_end": 0.45, "config_changes": []}
  ],
  "target": {"reward_threshold": 0.8, "max_rounds": 5, "plateau_rounds": 3},
  "mode": "approve",
  "current_run_id": "run_xxx",
  "completed": false
}
```

将状态写入 `rllm_trl/output/training_state.json`，以便中断后恢复。

## 错误恢复

| 场景 | 处理 |
|---|---|
| 训练进程崩溃 | 读取错误日志，诊断原因，调整配置后重试 |
| OOM | 自动减小 batch_size 和 num_generations，重新调用 rllm-config |
| 连续 2 轮失败 | 暂停，向用户报告问题，等待指示 |
| 用户中断 | 保存当前状态到 training_state.json，下次可从中断点恢复 |

## 使用示例

```
/rllm-train                                          ← 模糊输入，触发引导问答
/rllm-train 启动训练                                  ← 模糊输入，触发引导问答
/rllm-train 用 qwen-0.5b 训练                         ← 部分输入，只补问训练目标
/rllm-train 用 qwen-0.5b 训练数学 agent，reward 达到 0.8  ← 充分输入，直接解析
/rllm-train auto 模式，快速测试，16 个问题，reward >= 0.5
/rllm-train qwen-1.5b, 200 problems, 5 epochs, max 3 rounds
```
