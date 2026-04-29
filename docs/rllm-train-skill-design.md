# rllm_trl 自动训练 Skill 系统设计

## Context

用户希望开发一套 Claude Code skill，实现 rllm_trl 训练的全自动闭环：需求澄清 → 配置生成 → 启动训练 → 过程监控 → 结果分析 → 调参优化 → 重新训练，循环直到达成目标。支持全自动和人工批准两种模式。

运行环境：本地 Mac (CPU/MPS)，小模型快速迭代。
任务范围：可扩展设计，数学 agent 作为首个实现，预留其他 agent/env 扩展点。
停止条件：用户自定义（reward 阈值、plateau 检测、最大轮次等）。

## 整体架构

```
用户: /rllm-train "用 qwen-0.5b 训练数学 agent，reward 达到 0.8"
        │
        ▼
┌─────────────────────────────────────────────────┐
│  rllm-train (主编排 skill)                       │
│  ┌───────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ 需求澄清   │→│ 配置生成   │→│ 训练循环      │ │
│  │ (Phase 1)  │  │ (Phase 2) │  │ (Phase 3-5)  │ │
│  └───────────┘  └───────────┘  └──────────────┘ │
│                                  │  ┌─────────┐  │
│                                  ├→│ 启动训练  │  │
│                                  │  └────┬────┘  │
│                                  │  ┌────▼────┐  │
│                                  ├→│ 过程监控  │  │
│                                  │  └────┬────┘  │
│                                  │  ┌────▼────┐  │
│                                  └←│ 结果分析  │  │
│                                     │ + 调参   │  │
│                                     └─────────┘  │
└─────────────────────────────────────────────────┘
```

## Skill 划分

采用 **1 个主 skill + 5 个子 skill** 的结构，每个子 skill 可独立调用：

| Skill 名称 | 目录 | 职责 |
|---|---|---|
| `rllm-train` | `.claude/skills/rllm-train/` | 主编排，串联全流程 |
| `rllm-clarify` | `.claude/skills/rllm-clarify/` | 需求澄清，提取训练意图 |
| `rllm-config` | `.claude/skills/rllm-config/` | 生成/调整 TrainingConfig |
| `rllm-run` | `.claude/skills/rllm-run/` | 启动训练进程 |
| `rllm-monitor` | `.claude/skills/rllm-monitor/` | 监控训练进度 |
| `rllm-analyze` | `.claude/skills/rllm-analyze/` | 分析结果 + 生成调参建议 |

### 为什么不合成一个大 skill

- 每个阶段可独立调用和调优（例如只跑 `/rllm-analyze` 分析上次训练结果）
- 单个 SKILL.md 过大会降低 Claude 的指令遵循质量
- 用户可以只安装需要的子 skill

## 各 Skill 详细设计

### 1. rllm-train（主编排）

**文件**: `.claude/skills/rllm-train/SKILL.md`

**职责**: 接收用户自然语言需求，编排整个训练循环。

**核心逻辑**:
1. 调用 rllm-clarify 阶段澄清需求
2. 调用 rllm-config 生成初始配置
3. 进入训练循环：
   - 调用 rllm-run 启动训练
   - 调用 rllm-monitor 监控进度
   - 调用 rllm-analyze 分析结果
   - 判断是否达成停止条件
   - 未达成则根据分析建议调整配置，继续循环
4. 达成后输出最终报告

**执行模式**:
- `auto` 模式：全自动执行，每轮训练结束后自动分析、调参、重训
- `approve` 模式（默认）：每个关键决策点暂停等待用户确认
  - 初始配置确认
  - 每轮调参方案确认
  - 停止训练确认

**停止条件**（用户可组合）:
- `reward_threshold`: 平均 reward 达到指定值
- `plateau_rounds`: 连续 N 轮 reward 无显著提升（提升 < 5%）
- `max_rounds`: 最大训练轮次
- `max_wall_time`: 最大总耗时

### 2. rllm-clarify（需求澄清）

**文件**: `.claude/skills/rllm-clarify/SKILL.md`

**职责**: 从用户自然语言中提取结构化训练需求。

**提取信息**:
- 任务类型（math / code / search / custom）
- 模型选择（qwen-0.5b / 1.5b / 3b / 自定义）
- 数据规模（问题数量）
- 训练目标（reward 阈值、训练轮次等）
- 停止条件
- 执行模式（auto / approve）
- 特殊约束（时间限制、内存限制等）

**输出**: 结构化的需求摘要，供 rllm-config 使用。

### 3. rllm-config（配置生成）

**文件**: `.claude/skills/rllm-config/SKILL.md`

**职责**: 根据需求生成 TrainingConfig，或根据分析结果调整配置。

**两种模式**:

**初始配置生成**:
- 读取 `rllm_trl/config.py` 中的 TrainingConfig 定义
- 根据需求选择合理的初始超参
- 根据硬件环境（Mac CPU/MPS）调整 batch_size、dtype 等
- 生成配置文件 `rllm_trl/output/runs/<run_id>/config.json`

**调参优化**（根据 rllm-analyze 的建议）:
- 读取上一轮的 perf_stats.json 和 training_log.txt
- 根据分析建议修改配置
- 调参策略：
  - reward 低 + loss 不降 → 增大 learning_rate 或 num_generations
  - reward 低 + loss 降 → 增加 epochs 或 num_problems
  - 训练慢 → 减小 batch_size、max_completion_length
  - reward 震荡 → 降低 learning_rate、增大 gradient_accumulation_steps
  - reward plateau → 调整 temperature、尝试不同 loss_type

**可调参数范围**:
- `learning_rate`: 1e-6 ~ 1e-4
- `temperature`: 0.5 ~ 1.2
- `num_generations`: 2 ~ 8
- `batch_size`: 1 ~ 4（Mac 内存限制）
- `num_problems`: 16 ~ 256
- `num_epochs`: 1 ~ 10
- `max_agent_steps`: 2 ~ 5
- `gradient_accumulation_steps`: 1 ~ 8

### 4. rllm-run（启动训练）

**文件**: `.claude/skills/rllm-run/SKILL.md`

**职责**: 启动 rllm_trl 训练进程。

**执行流程**:
1. 读取配置文件
2. 生成 Python 启动命令（调用 `rllm_trl.train.main()`）
3. 用 Bash 工具在后台启动训练（`run_in_background`）
4. 记录进程信息（PID、输出文件路径、启动时间）
5. 返回运行句柄供 rllm-monitor 使用

**关键实现**: 需要一个轻量的 Python 入口脚本 `rllm_trl/run_training.py`，接受 JSON 配置文件路径，加载配置并调用 `main()`。

### 5. rllm-monitor（训练监控）

**文件**: `.claude/skills/rllm-monitor/SKILL.md`

**职责**: 实时监控训练进度，汇报关键指标。

**监控方式**:
- 使用 Monitor 工具 tail 训练日志文件 `training_log.txt`
- 匹配关键行：step 进度行（包含 Reward、tok/s、ETA）
- 检测训练完成标志（"Training Report" 或进程退出）

**汇报内容**:
- 当前 step / 总 step
- 当前 reward 趋势
- 预计剩余时间
- 异常检测（loss 爆炸、reward 归零、进程崩溃）

### 6. rllm-analyze（结果分析）

**文件**: `.claude/skills/rllm-analyze/SKILL.md`

**职责**: 分析训练结果，生成调参建议。

**分析维度**:

**训练效果分析**:
- 读取 `perf_stats.json` 中的 reward_stats
- 读取 `training_log.txt` 中的 reward 趋势
- 读取 trajectory JSONL 文件，分析 agent 行为质量
- 判断：reward 是否达标、是否在提升、是否 plateau

**性能分析**:
- 读取 `perf_stats.json` 中的 time_breakdown
- 分析瓶颈：LLM 推理占比、env 执行占比、logprob 计算占比
- tokens/sec 吞吐量
- 每步耗时趋势

**调参建议生成**:
- 基于效果和性能分析，生成具体的配置修改建议
- 每条建议包含：修改项、修改值、修改原因、预期效果
- 建议按优先级排序

**输出**: 分析报告 + 调参建议列表，存入 `rllm_trl/output/runs/<run_id>/analysis.json`

## 文件结构

```
.claude/skills/
├── rllm-train/
│   └── SKILL.md          # 主编排 skill
├── rllm-clarify/
│   └── SKILL.md          # 需求澄清
├── rllm-config/
│   └── SKILL.md          # 配置生成/调参
├── rllm-run/
│   └── SKILL.md          # 启动训练
├── rllm-monitor/
│   └── SKILL.md          # 训练监控
└── rllm-analyze/
    └── SKILL.md          # 结果分析

rllm_trl/
├── run_training.py       # 新增：接受 JSON 配置的训练入口
├── output/
│   └── runs/             # 新增：按 run_id 组织的训练输出
│       └── <run_id>/
│           ├── config.json
│           ├── training_log.txt
│           ├── perf_stats.json
│           ├── analysis.json
│           ├── trajectories/
│           └── final_model/
```

## 需要修改的现有文件

1. **`rllm_trl/config.py`** — TrainingConfig 增加字段：
   - `run_id: str` — 训练运行标识
   - `output_dir` 默认值改为基于 run_id 的路径
   - 增加 `to_json()` / `from_json()` 序列化方法

2. **`rllm_trl/train.py`** — main() 函数调整：
   - 支持从 JSON 文件加载配置
   - 训练结束后将 perf_stats 写入 run 目录

## 需要新增的文件

1. **`rllm_trl/run_training.py`** — JSON 配置驱动的训练入口
2. **6 个 SKILL.md 文件** — 各阶段的 skill 定义

## 执行模式详细流程

### approve 模式（默认）

```
用户: /rllm-train "用 qwen-0.5b 训练数学 agent，reward 达到 0.8"

Claude: [需求澄清] 确认训练需求：
  - 任务: 数学计算 agent
  - 模型: Qwen2.5-0.5B-Instruct
  - 目标: avg reward >= 0.8
  - 停止: reward 达标 或 连续 3 轮无提升 或 最多 5 轮
  确认开始？

用户: 确认

Claude: [配置生成] 第 1 轮初始配置：
  - 64 problems, 2 epochs, lr=1e-5, batch=2, 4 generations
  确认启动训练？

用户: 开始

Claude: [训练中] Step 3/16 | Reward: 0.312 | 12.3 tok/s | ETA: 2m30s
        ...
        [训练完成] 第 1 轮结果：avg reward = 0.45, 耗时 3m12s

Claude: [分析] 效果分析：
  - Reward 从 0.25 提升到 0.45，趋势向上但未达标
  - Loss 持续下降，模型在学习
  - 瓶颈：LLM 推理占 72%
  建议调整：
  1. 增加 epochs: 2 → 4（reward 仍在上升）
  2. 增加 problems: 64 → 128（更多训练数据）
  3. 降低 temperature: 0.7 → 0.6（减少随机性）
  确认执行调参并开始第 2 轮？

用户: 确认

Claude: [第 2 轮训练中] ...
```

### auto 模式

```
用户: /rllm-train "auto 模式，qwen-0.5b 数学 agent，reward >= 0.8，最多 5 轮"

Claude: [自动执行] 需求已确认，开始全自动训练循环...
        [第 1 轮] 配置: 64 problems, 2 epochs, lr=1e-5
        [训练中] Step 8/16 | Reward: 0.375 | ETA: 1m45s
        [第 1 轮完成] avg reward = 0.45 (目标: 0.80) — 未达标
        [自动调参] epochs: 2→4, problems: 64→128, temp: 0.7→0.6
        [第 2 轮] 配置已更新，开始训练...
        ...
        [第 3 轮完成] avg reward = 0.82 (目标: 0.80) — 达标!

Claude: 训练目标达成！3 轮训练，总耗时 12m30s
        最终 reward: 0.82, 模型已保存到 output/runs/xxx/final_model/
```

## 实现计划

按以下顺序实现：

1. **修改 rllm_trl 基础设施**（config.py, train.py, run_training.py）
   - TrainingConfig 增加 run_id、序列化
   - 新增 run_training.py 入口
   - output 目录按 run_id 组织

2. **实现子 skill（按依赖顺序）**
   - rllm-clarify — 需求澄清
   - rllm-config — 配置生成
   - rllm-run — 启动训练
   - rllm-monitor — 训练监控
   - rllm-analyze — 结果分析

3. **实现主 skill**
   - rllm-train — 编排全流程

4. **端到端验证**
   - 用 `/rllm-train "quick test"` 跑一轮完整流程
   - 验证 approve 模式的交互流程
   - 验证 auto 模式的自动循环

## 验证方法

1. 单独调用每个子 skill 验证其功能：
   - `/rllm-config "qwen-0.5b 数学 agent"` → 检查生成的配置是否合理
   - `/rllm-run` → 检查训练是否正常启动和完成
   - `/rllm-analyze` → 检查分析报告是否准确
2. 端到端测试：`/rllm-train "quick test with 16 problems, reward >= 0.5"`
3. 验证 auto 模式下的自动循环是否正确执行
