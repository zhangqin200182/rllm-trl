# rllm-trl

> **⚠️ 项目已迁移到 [agent-evolution](https://github.com/zhangqin200182/agent-evolution)**
>
> 本仓库不再维护。项目已升级为三层自演进架构（agents train themselves, optimize their training process, and recursively improve the optimizer），所有后续开发都在新仓库进行。

---

将 [rLLM](https://github.com/agentification/rllm) 的 agent/environment 抽象与 HuggingFace [TRL](https://github.com/huggingface/trl) 的 GRPOTrainer 结合，在 Mac 上用强化学习训练语言 agent。

## 为什么做这个

rLLM 训练 RL agent 需要 vllm、flash-attn、deepspeed，这些都无法在 Mac 上运行。本项目内联了最小的 rLLM 抽象（agent、environment、trajectory），接入 TRL 的 GRPOTrainer，让你可以在 MPS 或 CPU 上本地快速迭代。

## 快速开始

```bash
pip install torch transformers trl datasets

# 默认配置（Qwen2.5-0.5B，64 道题，2 个 epoch）
python -m rllm_trl.train

# 自然语言配置（支持中英文）
python -m rllm_trl.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
python -m rllm_trl.train "quick test with 16 problems"

# 从配置文件启动（由 rllm-config skill 生成）
python -m rllm_trl.run_training rllm_trl/output/runs/<run_id>/config.json
```

训练输出在 `rllm_trl/output/runs/<run_id>/`：config.json、training_log.txt、trajectories/（JSONL）、perf_stats.json、analysis.json、final_model/。

## Claude Code Skill 自动训练系统

本项目的核心特色是一套 Claude Code skill 系统，实现训练全流程自动化闭环：

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

### Skill 一览

| Skill | 职责 |
|---|---|
| `rllm-train` | 主编排，串联全流程，支持 auto/approve 两种执行模式 |
| `rllm-clarify` | 从自然语言中提取结构化训练需求（中英文） |
| `rllm-config` | 生成初始配置 / 根据分析结果自动调参 |
| `rllm-run` | 后台启动训练进程 |
| `rllm-monitor` | 实时监控训练进度，检测异常（loss 爆炸、OOM、进程崩溃） |
| `rllm-analyze` | 分析训练结果，生成调参建议（含决策树） |

### 执行模式

**approve 模式**（默认）：每个关键决策点暂停等待确认 — 初始配置、每轮调参方案、停止训练。

**auto 模式**：全自动执行，自动分析、调参、重训，循环直到 reward 达标或触发停止条件。

```bash
# approve 模式
/rllm-train 用 qwen-0.5b 训练数学 agent，reward 达到 0.8

# auto 模式
/rllm-train auto 模式，快速测试，16 个问题，reward >= 0.5
```

### 自动调参策略

分析模块内置决策树，根据训练状态自动选择调参方向：

- reward 在上升 + loss 在降 → 增加 epochs 或 problems
- reward 停滞 → 调整 temperature，增加 num_generations
- reward 震荡 → 降低 learning_rate，增大 gradient_accumulation
- reward 下降 → 大幅降低 learning_rate，回退配置

## 训练管线工作原理

```
train.py → GRPOTrainer → rollout_func → HFAgentExecutionEngine → agent/env 循环
```

每个训练步骤：
1. GRPOTrainer 调用自定义 rollout 函数
2. `HFAgentExecutionEngine` 用 `model.generate()` 运行 agent-environment 循环
3. `ToolAgent` 解析工具调用，`MathCalcEnv` 执行并返回观测
4. 逐 token 响应掩码（1=模型，0=环境）确保只有模型 token 参与梯度更新
5. 轨迹、奖励、掩码回传给 GRPOTrainer 进行策略更新

## 模块结构

| 模块 | 职责 |
|---|---|
| `train.py` | 入口，构建数据集、模型、tokenizer，接入 GRPOTrainer |
| `config.py` | `TrainingConfig` + 自然语言解析器（中英文） |
| `rollout.py` | TRL 与 rLLM 风格 agent 执行的桥梁 |
| `hf_engine.py` | agent-env 循环执行引擎，管理 token 掩码 |
| `base.py` | 核心抽象：`BaseAgent`、`BaseEnv`、`Trajectory`、`ToolCall` |
| `tool_agent.py` | 对话管理、工具调用解析 |
| `math_env.py` | 计算器环境，算术题目生成 |
| `parsers.py` | 聊天模板 + 工具调用解析，token 级掩码生成 |
| `logger.py` | 训练实时进度表和总结报告 |
| `perf_stats.py` | 耗时分解：推理、环境、logprob、GRPO |
| `trajectory_writer.py` | 逐步 JSONL 输出 |

## 关键设计

**响应掩码**：掩码系统（1=模型 token，0=环境 token）是 GRPO 训练正确性的核心。环境注入的 token 不参与梯度计算，由 `parsers.py` 中的 `convert_messages_to_tokens_and_masks()` 实现。

**最小抽象**：不依赖完整 rLLM 框架，只内联 `BaseAgent`、`BaseEnv`、`Step`、`Trajectory`，保持依赖轻量。

**Skill 解耦**：每个 skill 可独立调用（如 `/rllm-analyze` 单独分析上次训练），也可由主编排串联成完整闭环。

## Skill Bank

Skill 通过 `skill-bank/` 目录管理，采用 base + patch + compile 架构。不要直接编辑 `.claude/skills/*/SKILL.md`，而是修改 base 或添加 patch，然后编译。

```bash
# 编译单个 skill
python skill-bank/compile.py rllm-config

# 编译整个 group
python skill-bank/compile.py --group rllm

# 查看 patch 状态
python skill-bank/compile.py --status

# 预览变更
python skill-bank/compile.py --diff rllm-config
```

每个 skill 的结构：`skill-bank/<group>/<skill>/base.md`（带 section 锚点）、`patches/*.md`、`manifest.yaml`。详见 `docs/skill-bank-design.md`。

## License

MIT
