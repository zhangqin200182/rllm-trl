# traj-xx / rllm-xx 观察者与被观察者隔离设计

> 基于 traj-loop 3 轮优化测试的实证发现，定义 traj-xx（观察者）与 rllm-xx（被观察者）之间的隔离架构。通过 Agent 子 agent 实现上下文物理隔离，通过 trajectory/output/ 实现数据流隔离，通过 skill 指令实现文件目录和领域知识隔离。

生成时间: 2026-05-02
状态: 设计方案（待实施）

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [设计原则](#2-设计原则)
3. [隔离架构总览](#3-隔离架构总览)
4. [上下文隔离: Agent 子 agent 方案](#4-上下文隔离-agent-子-agent-方案)
5. [数据流隔离: trajectory/output/ 单一通道](#5-数据流隔离-trajectoryoutput-单一通道)
6. [数据表面化准则](#6-数据表面化准则)
7. [traj-analyze-rllm 数据边界与领域知识重构](#7-traj-analyze-rllm-数据边界与领域知识重构)
8. [traj-loop Agent 隔离编排](#8-traj-loop-agent-隔离编排)
9. [Python 基础设施增强](#9-python-基础设施增强)
10. [traj-analyze-xx 通用协议](#10-traj-analyze-xx-通用协议)
11. [实施计划](#11-实施计划)
12. [验证方案](#12-验证方案)

---

## 1. 背景与问题

### 1.1 实证来源

traj-loop 3 轮优化测试（2026-05-01 ~ 05-02）完整执行了 训练 → 分割 → 分析 → 优化 的闭环。3 轮训练分别使用不同参数配置，最终 reward 从 0（Round 1 崩溃）提升到 0.742（Round 3 成功）。测试过程中暴露了 traj-xx 与 rllm-xx 之间缺乏隔离的架构问题。

详细问题记录见 `trajectory/output/reports/20260502-traj-loop-problems-analysis.md`。

### 1.2 发现的隔离违反

| # | 违反 | 现象 | 影响 |
|---|------|------|------|
| V1 | traj-analyze-rllm 内嵌 rllm 领域知识 | `domain-knowledge` section 硬编码了 qwen-0.5b lr 安全范围 (1e-6~1e-5)、问题→skill 固定映射表 | 分析器不是从轨迹观察推断，而是查表匹配，失去了自动发现新模式的能力 |
| V2 | traj-loop 直接读 rllm 训练日志 | 编排层 tail rllm_trl/output/runs/*/training_log.txt 做分析 | 绕过了 trajectory 数据管线，traj-segment 被跳过 |
| V3 | Hooks 无法捕获训练子进程数据 | trajectory/output/raw/ 始终为空 | 训练进程通过 `python -m rllm_trl.run_training` 运行，其内部操作不经过 Claude Code 工具系统 |
| V4 | 无上下文隔离 | traj-loop 在同一对话中先调 /rllm-train 再调 /traj-analyze-rllm | 分析器能看到 rllm 的全部执行上下文（config.json 内容、training_log.txt 输出等），隔离形同虚设 |

### 1.3 核心矛盾

trajectory-design.md 的设计原则 "rllm-xx 不知道 traj-xx 的存在" 和 "traj-xx 是通用框架" 在实际执行中被违反。根本原因是缺乏强制隔离机制 — 仅靠 skill 指令中的 "应该" 无法阻止 LLM 在同一对话上下文中直接读取 rllm 内部数据。

---

## 2. 设计原则

### 2.1 核心原则

**traj-xx 只能通过 trajectory/output/ 中的轨迹数据间接了解 rllm-xx 的行为。**

这包括四个维度的隔离:

1. **上下文隔离** — traj-analyze-rllm 的 LLM 对话上下文中不能包含 rllm-train 执行过程中读取的任何文件内容
2. **数据流隔离** — traj-xx 的唯一数据输入是 trajectory/output/，不直接访问 rllm_trl/output/
3. **文件目录隔离** — 明确的目录归属: trajectory/ 属于 traj-xx，rllm_trl/ 和 skill-bank/rllm/ 属于 rllm-xx
4. **领域知识隔离** — traj-analyze-rllm 的领域知识来自模式识别方法论，不来自硬编码的参数表

### 2.2 隔离强度分级

| 强度 | 机制 | 特点 | 适用场景 |
|------|------|------|---------|
| 物理隔离 | Agent 子 agent（独立对话上下文） | 子 agent 之间物理上无法互相访问上下文 | 上下文隔离 |
| 架构约束 | trajectory/output/ 作为唯一数据通道 | hooks 捕获 → 轨迹存储 → 分析读取 | 数据流隔离 |
| 指令约束 | skill 指令中的 data-boundary 规则 | 明确列出允许/禁止的路径 | 文件目录隔离 |
| 设计约束 | 模式识别替代硬编码表 | 从轨迹观察推断而非预设 | 领域知识隔离 |

### 2.3 与现有设计准则的关系

本文档是 `docs/trajectory-design.md` 的补充，不替代原文档。具体关系:

- **准则 3.1（数据完整性）** — 从 "应确保" 升级为 "必须确保"，增加实施方式（data-surfacing section）
- **准则 3.3（分析可插拔）** — 增加通用协议定义（Section 10）
- **新增准则 3.5** — 隔离边界准则（本文档的核心内容）

---

## 3. 隔离架构总览

### 3.1 架构图

```
traj-loop (主编排, 父对话)
  │
  │  ┌─────────────────────────────────────────────────┐
  │  │ Agent 子 agent 1: rllm-train                     │
  │  │                                                   │
  │  │  rllm-clarify → rllm-config → rllm-run           │
  │  │  → rllm-monitor → rllm-analyze                   │
  │  │                                                   │
  │  │  上下文: 只有训练相关内容                            │
  │  │  Hooks 捕获所有工具调用 → trajectory/output/raw/    │
  │  │                                                   │
  │  │  返回: run_id, reward, 是否成功                     │
  │  └──────────────────────┬──────────────────────────┘
  │                         │ hooks 写入
  │                         ▼
  │              ┌─────────────────────┐
  │              │ trajectory/output/   │ ← 唯一数据通道
  │              │  raw/               │
  │              │  trajectories/      │
  │              │  reports/           │
  │              │  index.jsonl        │
  │              └──────────┬──────────┘
  │                         │
  ├── /traj-segment ────────┤ 读 raw/ → 写 trajectories/
  │                         │
  │  ┌──────────────────────┴──────────────────────────┐
  │  │ Agent 子 agent 2: traj-analyze-rllm              │
  │  │                                                   │
  │  │  上下文: 全新的，看不到子 agent 1 的任何内容         │
  │  │  输入: 只从 trajectory/output/ 读取                │
  │  │  产出: trajectory/output/reports/                  │
  │  │                                                   │
  │  │  返回: 报告路径, 优化建议数量                        │
  │  └─────────────────────────────────────────────────┘
  │
  └── /traj-optimize ← 读 reports/ → 生成 skill-bank patch
        │
        ▼
  ┌──────────────┐
  │  skill-bank  │ ← 编译更新后的 SKILL.md
  └──────┬───────┘
         │ 更新
         ▼
  ┌──────────────────┐
  │ rllm-xx skills   │ ← 下一轮训练使用优化后的 skill
  └──────────────────┘
```

### 3.2 隔离边界表

| 目录/文件 | 归属 | traj-xx 可访问 | rllm-xx 可访问 |
|-----------|------|---------------|---------------|
| `rllm_trl/` | rllm-xx | 禁止直接读取 | 完全访问 |
| `rllm_trl/output/runs/*/` | rllm-xx | 禁止直接读取 | 完全访问 |
| `skill-bank/rllm/` | rllm-xx | 禁止直接读取 | 完全访问 |
| `trajectory/` | traj-xx | 完全访问 | 不感知 |
| `trajectory/output/` | traj-xx | 完全访问 | 不感知 |
| `skill-bank/traj/` | traj-xx | 完全访问 | 不感知 |
| `.claude/skills/*/SKILL.md` | 编译产物 | 只读自身 skill | 只读自身 skill |
| `docs/` | 共享 | 只读 | 只读 |

### 3.3 数据流路径

```
rllm-xx 执行训练
  │
  │ rllm-monitor 用 Read 读取 config.json        ← 数据表面化
  │ rllm-monitor 用 Bash tail training_log.txt    ← 数据表面化
  │ rllm-analyze 用 Read 读取 perf_stats.json     ← 数据表面化
  │
  ▼
Hooks 捕获工具调用的 input/response
  │
  ▼
trajectory/output/raw/{session_id}/events.jsonl   ← 原始事件
  │
  ▼ traj-segment
trajectory/output/trajectories/{session_id}/      ← 分割后轨迹
  │
  ▼ traj-analyze-rllm (在独立子 agent 中)
trajectory/output/reports/{timestamp}-report.md   ← 分析报告
  │
  ▼ traj-optimize
skill-bank/{group}/{skill}/patches/               ← 生成 patch
  │
  ▼ compile.py
.claude/skills/*/SKILL.md                         ← 更新 skill
```

---

## 4. 上下文隔离: Agent 子 agent 方案

### 4.1 问题: Skill 工具无法实现上下文隔离

Claude Code 的 Skill 工具在当前对话上下文中注入 SKILL.md 并执行。所有工具调用共享同一个对话历史。

在 traj-loop 中，如果用 Skill 调用:
1. `/rllm-train` 执行训练，过程中 Read config.json、tail training_log.txt、Read perf_stats.json
2. 这些文件内容留在对话上下文中
3. 随后 `/traj-analyze-rllm` 在同一对话中执行，能直接看到上述所有内容
4. 分析器实际上是在"开卷考试"，而非从轨迹数据中独立推断

这违反了隔离原则: 分析器应该只能通过 trajectory/output/ 间接了解训练行为。

### 4.2 方案: Agent 工具实现物理隔离

Claude Code 的 Agent 工具创建独立的子 agent，子 agent 拥有全新的对话上下文，看不到父对话的历史。

利用这个特性:

| 步骤 | 调用方式 | 上下文 | 理由 |
|------|---------|--------|------|
| rllm-train | Agent 子 agent 1 | 独立上下文，只有训练内容 | 训练上下文在子 agent 结束后被丢弃 |
| traj-segment | Skill（父对话） | 父对话上下文 | 只读 trajectory/output/raw/，无隔离需求 |
| traj-analyze-rllm | Agent 子 agent 2 | 独立上下文，全新 | 物理上无法看到子 agent 1 的内容 |
| traj-optimize | Skill（父对话） | 父对话上下文 | 只读 trajectory/output/reports/，无隔离需求 |

关键保证:
- 子 agent 1（rllm-train）和子 agent 2（traj-analyze-rllm）之间**唯一的数据通道是文件系统** trajectory/output/
- 父对话只接收子 agent 的**返回摘要**（一条消息），不接收完整执行过程
- 子 agent 2 物理上无法访问子 agent 1 中 rllm 读取的 config.json、training_log.txt 等内容

### 4.3 Agent 调用规范

#### rllm-train 子 agent

```
Agent(
    prompt="读取 .claude/skills/rllm-train/SKILL.md 并按其步骤执行训练。
            训练描述: {description}
            工作目录: /Users/kevin/code/MyProject
            训练完成后输出 run_id 和最终 reward。",
    description="rllm-train round {N}"
)
```

子 agent 执行完整的 rllm-train 流程（clarify → config → run → monitor → analyze）。Hooks 自动捕获子 agent 中所有工具调用到 trajectory/output/raw/。

返回值: run_id, 最终 reward, 是否成功。

#### traj-analyze-rllm 子 agent

```
Agent(
    prompt="读取 .claude/skills/traj-analyze-rllm/SKILL.md 并按其步骤执行分析。
            工作目录: /Users/kevin/code/MyProject
            只从 trajectory/output/ 读取数据，不要读取 rllm_trl/ 下的文件。
            分析完成后输出报告路径。",
    description="traj-analyze-rllm round {N}"
)
```

子 agent 在全新上下文中执行，物理上看不到 rllm-train 的执行细节。

返回值: 报告路径, 优化建议数量。

### 4.4 为什么 traj-segment 和 traj-optimize 不需要 Agent 隔离

- **traj-segment**: 只读 trajectory/output/raw/，将原始事件分割为轨迹。不涉及 rllm 内部数据，不存在信息泄漏风险。
- **traj-optimize**: 只读 trajectory/output/reports/，根据分析报告生成 skill-bank patch。输入已经是分析后的结构化建议，不含 rllm 原始数据。

这两个步骤在父对话中用 Skill 调用即可，无需额外的 Agent 隔离开销。

### 4.5 Hooks 在 Agent 子 agent 中的行为

Claude Code Hooks 对子 agent 中的工具调用同样生效:
- PostToolUse hook 捕获子 agent 1 中 rllm-monitor 的 Read/Bash 调用
- SubagentStop hook 在子 agent 结束时触发
- 捕获的事件写入 trajectory/output/raw/，conversation_id 标记为子对话

这是隔离方案的关键前提: 子 agent 1 中 rllm-xx 读取的训练数据被 hooks 捕获到 trajectory/output/，子 agent 2 才有数据可分析。

---

## 5. 数据流隔离: trajectory/output/ 单一通道

### 5.1 数据通道定义

trajectory/output/ 是 traj-xx 观察 rllm-xx 行为的**唯一数据通道**。

```
trajectory/output/
├── raw/                    # 原始 hook 事件（输入）
│   └── {session_id}/
│       └── events.jsonl    # hooks 捕获的工具调用
├── trajectories/           # 分割后轨迹（中间产物）
│   └── {session_id}/
│       └── trajectories.jsonl
├── reports/                # 分析报告（输出）
│   └── {timestamp}-report.md
└── index.jsonl             # 全局索引
```

### 5.2 数据流向约束

| 方向 | 允许 | 禁止 |
|------|------|------|
| rllm-xx → trajectory/output/ | 通过 hooks 自动捕获 | rllm-xx 不直接写入 trajectory/ |
| trajectory/output/ → traj-xx | traj-xx 读取 raw/, trajectories/, reports/ | — |
| rllm_trl/output/ → traj-xx | 禁止 | traj-xx 不直接读取 rllm_trl/output/runs/* |
| traj-xx → skill-bank | 通过 traj-optimize 生成 patch | traj-xx 不直接修改 rllm-xx 的 base.md |

### 5.3 V3 问题的应对: Hooks 无法捕获训练子进程

traj-loop 3 轮测试暴露: Hooks 只捕获 Claude Code 工具调用，不捕获 `python -m rllm_trl.run_training` 子进程的内部 I/O。

解决方案不是扩展 hooks 的捕获范围，而是**数据表面化**: 要求 rllm-xx skill 在执行过程中用 Read/Bash 工具明确读取训练关键数据，使这些数据通过工具调用进入对话，从而被 hooks 捕获。

详见 Section 6。

---

## 6. 数据表面化准则

### 6.1 原理

Hooks 只捕获 Claude Code 工具调用的 input 和 response。如果 rllm-monitor 只用 Monitor 工具的 grep 模式匹配训练日志，grep 的输出虽然会作为通知出现在对话中，但不会被 PostToolUse hook 捕获为完整的工具调用事件。

数据表面化要求: rllm-xx skill 必须用 Read/Bash 工具**明确读取**训练关键数据，使其作为工具调用的 response 被 hooks 记录。

### 6.2 rllm-monitor 数据表面化要求

在 `skill-bank/rllm/rllm-monitor/base.md` 新增 `<!-- section:data-surfacing -->`:

| 时机 | 操作 | 工具 | 目的 |
|------|------|------|------|
| 训练启动时 | 读取 config.json 完整内容 | Read | 捕获训练配置 |
| 训练过程中 | 定期 tail training_log.txt 最后 30 行 | Bash | 捕获 reward 趋势（不仅靠 Monitor grep） |
| 异常发生时 | 读取完整错误段 | Read/Bash | 捕获错误上下文 |
| 训练结束时 | 读取 perf_stats.json | Read | 捕获性能统计 |
| 训练结束时 | 读取 training_log.txt 最后 50 行 | Bash | 捕获最终 Training Report |

原因说明: Monitor 工具的 grep 输出不包含完整上下文，且不被 PostToolUse hook 捕获。定期用 Bash tail 读取日志是确保轨迹数据完整性的必要补充。

### 6.3 rllm-analyze 数据表面化要求

在 `skill-bank/rllm/rllm-analyze/base.md` 新增 `<!-- section:data-surfacing -->`:

分析时必须用 Read 工具逐一完整读取以下文件:
- `config.json` — 训练配置
- `training_log.txt` — 完整训练日志
- `perf_stats.json` — 性能统计
- `trajectories/*.jsonl` — 训练轨迹

禁止仅依赖对话上下文中已有的信息。即使 rllm-monitor 阶段已经读取过部分内容，rllm-analyze 仍必须重新 Read，确保 hooks 捕获到完整的分析输入。

### 6.4 数据表面化与隔离的关系

```
rllm-monitor (子 agent 1 中)
  │ Read config.json          → hooks 捕获 → raw/events.jsonl
  │ Bash tail training_log    → hooks 捕获 → raw/events.jsonl
  │ Read perf_stats.json      → hooks 捕获 → raw/events.jsonl
  ▼
子 agent 1 结束，上下文被丢弃

traj-segment (父对话中)
  │ 读取 raw/events.jsonl
  │ 从 tool_response 字段中提取训练数据
  ▼ 写入 trajectories/

traj-analyze-rllm (子 agent 2 中)
  │ 读取 trajectories/
  │ 从轨迹数据中提取 config、reward 趋势、错误信息
  │ 基于模式识别方法论进行分析
  ▼ 写入 reports/
```

数据表面化是隔离方案的前提: 没有表面化，trajectory/output/ 中就没有足够的数据供分析器使用。

---

## 7. traj-analyze-rllm 数据边界与领域知识重构

### 7.1 数据边界定义

在 `skill-bank/traj/traj-analyze-rllm/base.md` 新增 `<!-- section:data-boundary -->`:

#### 允许读取的数据源

| 路径 | 内容 | 用途 |
|------|------|------|
| `trajectory/output/raw/{session_id}/events.jsonl` | 原始 hook 事件 | 提取训练数据（从 tool_response 字段） |
| `trajectory/output/trajectories/{session_id}/trajectories.jsonl` | 分割后轨迹 | 主要分析输入 |
| `trajectory/output/reports/` | 历史分析报告 | 跨轮次对比 |
| `trajectory/output/index.jsonl` | 全局索引 | 查找相关 session |

#### 禁止直接读取的数据源

| 路径 | 原因 |
|------|------|
| `rllm_trl/output/runs/*/` | 属于 rllm-xx，只能通过轨迹间接获取 |
| `rllm_trl/config.py` | 属于 rllm-xx 内部实现 |
| `rllm_trl/*.py` | 属于 rllm-xx 内部实现 |
| `skill-bank/rllm/*/base.md` | 属于 rllm-xx skill 源码 |
| `.claude/skills/rllm-*/SKILL.md` | 属于 rllm-xx 编译产物 |

#### 上下文隔离说明

本 skill 在独立 Agent 子 agent 中执行，拥有全新的对话上下文。物理上无法看到:
- rllm-train 子 agent 中读取的 config.json 内容
- rllm-monitor 子 agent 中 tail 的 training_log.txt 输出
- 任何 rllm-xx 执行过程中的中间状态

唯一的数据来源是 trajectory/output/ 目录下的文件。

### 7.2 领域知识重构

#### 当前问题

`<!-- section:domain-knowledge -->` 中硬编码了:
```
| qwen-0.5b | 1e-6 ~ 1e-5 | 4-8 | 3 |
| qwen-1.5b | 5e-7 ~ 5e-6 | 2-4 | 2 |
```

这违反隔离原则: 分析器不应预设 rllm 的参数安全范围，而应从轨迹数据中观察推断。

#### 新设计: 模式识别方法论

替换硬编码表为通用的训练动态模式识别指南:

| 模式 | 轨迹中的特征 | 含义 | 分析方法 |
|------|-------------|------|---------|
| 快速崩溃 | reward 在前 10% steps 内从峰值降为 0 | 学习率过高或训练量过大 | 对比相同模型不同 lr 的轨迹 |
| 延迟崩溃 | reward 在 10-30% steps 时开始下降 | 数据量超出模型容量 | 对比不同 num_problems 的轨迹 |
| 稳定学习 | reward 单调递增或小幅波动 | 配置合理 | 记录为安全配置参考 |
| 高 variance | reward 大幅震荡 | 学习率偏高或 batch 偏小 | 对比相邻轮次的 lr 和 reward variance |
| 格式退化 | 后期 tool_call 格式错误增加 | 模型遗忘了格式模板 | 检查训练长度和格式 reward 权重 |
| 零学习 | reward 始终接近 0 | 题目太难或 lr 太低 | 对比难度配置和模型能力 |

#### 分析方法论

1. **从轨迹提取事实** — 读取轨迹数据，提取配置参数、reward 序列、错误信息
2. **跨轮次对比** — 比较不同轮次的参数变化与结果变化，建立因果关系
3. **模式匹配** — 将观察到的 reward 趋势与上述模式表对照
4. **安全边界推断** — 基于多轮数据推断参数安全范围（而非预设）
5. **生成假说** — 对观察到的现象提出可能的解释，标注置信度

#### 禁止事项

- 不引用预设的参数安全范围表
- 不假设固定的 "问题 → skill" 映射
- 所有数值判断须有轨迹证据支撑
- 不使用 "经验表明" / "通常情况下" 等无证据表述
- 如果轨迹数据不足以得出结论，明确报告 "数据不足"

### 7.3 Step 2 重写: 从轨迹提取训练详情

原 Step 2:
> 从 tool_calls 中提取 rllm-config 生成的配置参数，从 Bash 工具调用中提取训练日志输出

新 Step 2:

```
### 2. 从轨迹数据提取训练详情

对每条 rllm-train 轨迹:

a) 读取 trajectory/output/trajectories/{session_id}/trajectories.jsonl
   → 定位 skill_name="rllm-*" 的轨迹

b) 从轨迹的 tool_calls 中提取训练数据:
   - tool_name="Read" + file_path 含 "config.json" → 训练配置
   - tool_name="Bash" + command 含 "tail" + "training_log" → reward 趋势
   - tool_name="Read" + file_path 含 "perf_stats.json" → 性能统计
   - tool_name="Bash" + response 含 "Error"/"Traceback" → 错误信息

c) 如果轨迹数据为空或 tool_response 中缺少关键信息:
   → 报告: "数据不足。请确认 rllm-xx skill 的数据表面化准则已实施。
            缺失: [config/reward_trend/perf_stats/errors]"
   → 不做猜测性分析
```

---

## 8. traj-loop Agent 隔离编排

### 8.1 修改概要

`skill-bank/traj/traj-loop/base.md` 是最大的改动。核心变化:
- 执行规则新增隔离约束
- 循环步骤从 Skill 调用改为 Agent 调用
- 新增 agent-isolation section 解释设计

### 8.2 新增执行规则

在 `<!-- section:rules -->` 中追加:

```markdown
5. **上下文隔离** — rllm-train 和 traj-analyze-rllm 必须在独立的 Agent 子 agent 中执行，
   确保分析器无法看到训练的执行上下文
6. **数据流隔离** — traj-xx 步骤只从 trajectory/output/ 读数据，不直接访问 rllm_trl/output/
7. **traj-segment 不可跳过** — 即使轨迹数据为空也必须调用（记录空结果供追溯）
8. **禁止直接分析训练日志** — 不在编排层 Read/tail rllm_trl/ 文件做分析
```

### 8.3 循环执行步骤（重写）

```
for round in 1..N:

    Step 1: 训练 (在独立子 agent 中)
    ─────────────────────────────────
    使用 Agent 工具启动子 agent:
    
    Agent(
        prompt="读取 .claude/skills/rllm-train/SKILL.md 并按其步骤执行训练。
                训练描述: {description}
                工作目录: /Users/kevin/code/MyProject
                Round {N}/{total_rounds}。
                训练完成后输出: run_id, 最终 reward, 是否成功。",
        description="rllm-train round {N}"
    )
    
    → 子 agent 执行完整的 rllm-train 流程
    → Hooks 自动捕获子 agent 中所有工具调用到 trajectory/output/raw/
    → 子 agent 返回摘要，其完整上下文被丢弃

    Step 1.5: 验证轨迹数据完整性
    ─────────────────────────────
    → ls trajectory/output/raw/ 检查是否有新事件
    → 如果为空: 输出警告 "Hooks 未捕获到训练数据" 但不中断循环
    → 如果有数据: 报告事件数量

    Step 2: 分割轨迹
    ─────────────────
    调用 Skill("traj-segment")
    → 读取 trajectory/output/raw/ → 输出到 trajectory/output/trajectories/

    Step 3: 分析 (在独立子 agent 中)
    ─────────────────────────────────
    使用 Agent 工具启动子 agent:
    
    Agent(
        prompt="读取 .claude/skills/traj-analyze-rllm/SKILL.md 并按其步骤执行分析。
                工作目录: /Users/kevin/code/MyProject
                只从 trajectory/output/ 读取数据，不要读取 rllm_trl/ 下的文件。
                分析完成后输出: 报告路径, 优化建议数量。",
        description="traj-analyze-rllm round {N}"
    )
    
    → 子 agent 在全新上下文中执行
    → 物理上看不到 Step 1 的训练细节
    → 只能从 trajectory/output/ 获取数据
    → 返回报告路径

    Step 4: 生成 patch
    ──────────────────
    调用 Skill("traj-optimize", args="{report_path}")
    → 读取分析报告 → 展示 patch → 等待用户确认 → 编译

    Step 5: 本轮摘要
    ─────────────────
    输出本轮结果: round, run_id, reward, patches_generated, patches_accepted
```

### 8.4 与现有 rllm-train 编排规则的兼容

rllm-train 的 SKILL.md 定义了 Phase 0-6 的完整编排流程，要求 "每个 Phase 必须通过调用对应的子 skill 来执行"。

在 Agent 隔离方案中，rllm-train 子 agent 独立执行其完整流程，不受外层 traj-loop 影响。子 agent 看到的只是 "读取 SKILL.md 并执行训练" 的指令，它会按自己的编排规则走完所有 Phase。

变化:
- 旧方式: traj-loop 用 Skill("rllm-train") → 在同一对话中执行
- 新方式: traj-loop 用 Agent → 在独立对话中执行 → 结果通过 hooks 流入 trajectory/output/

### 8.5 循环模式下的效率优化

traj-loop 3 轮测试发现: 严格遵循 "Skill 调用后当轮响应立即结束" 规则导致 15+ 轮对话消耗。

Agent 方案自然解决此问题:
- 子 agent 内部可以走完整的多 Phase 流程而不占用父对话的轮次
- 父对话只等待子 agent 返回的一条摘要消息
- 不需要因为 Skill 调用规则而在每个 Phase 切换轮次

---

## 9. Python 基础设施增强

### 9.1 新增方法: `trajectory/analyzer/base.py`

为 traj-analyze-rllm 子 agent 提供从轨迹中提取训练数据的便利方法:

```python
class AnalyzerBase:
    # ... 现有方法保持不变 ...

    def extract_training_data(self, traj: Trajectory) -> Dict[str, Any]:
        """从轨迹的 tool_calls 中提取被 rllm-monitor/rllm-analyze 表面化的训练数据。
        
        返回:
            {
                "config": {...} or None,        # 从 Read config.json 的 response 中提取
                "reward_trend": [...] or None,   # 从 Bash tail training_log 的 response 中提取
                "perf_stats": {...} or None,     # 从 Read perf_stats.json 的 response 中提取
                "errors": [...],                 # 从含 Error/Traceback 的 response 中提取
                "log_snippets": [...],           # 从 tail training_log 的 response 中提取
            }
        """
        result = {
            "config": None,
            "reward_trend": None,
            "perf_stats": None,
            "errors": [],
            "log_snippets": [],
        }
        
        for tc in traj.tool_calls:
            if tc.tool_name == "Read" and tc.tool_response:
                file_path = tc.tool_input.get("file_path", "")
                response_text = str(tc.tool_response)
                
                if "config.json" in file_path:
                    try:
                        result["config"] = json.loads(response_text)
                    except (json.JSONDecodeError, TypeError):
                        result["config"] = {"raw": response_text[:2000]}
                        
                elif "perf_stats.json" in file_path:
                    try:
                        result["perf_stats"] = json.loads(response_text)
                    except (json.JSONDecodeError, TypeError):
                        result["perf_stats"] = {"raw": response_text[:2000]}
                        
            elif tc.tool_name == "Bash" and tc.tool_response:
                command = tc.tool_input.get("command", "")
                response_text = str(tc.tool_response)
                
                if "training_log" in command or "tail" in command:
                    result["log_snippets"].append(response_text[:3000])
                    # 尝试提取 reward 趋势
                    rewards = self._extract_rewards_from_log(response_text)
                    if rewards:
                        result["reward_trend"] = rewards
                        
                if "Error" in response_text or "Traceback" in response_text:
                    result["errors"].append({
                        "command": command,
                        "error": response_text[:2000]
                    })
        
        return result

    def get_available_training_data(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取所有 rllm 轨迹及其包含的训练数据。
        
        返回每条轨迹的摘要 + 提取的训练数据。
        供 traj-analyze-rllm 直接调用。
        """
        trajs = self.get_rllm_trajectories(days)
        results = []
        for traj in trajs:
            summary = self.summarize_trajectory(traj)
            training_data = self.extract_training_data(traj)
            results.append({
                **summary,
                "training_data": training_data,
            })
        return results

    def _extract_rewards_from_log(self, log_text: str) -> Optional[List[Dict[str, Any]]]:
        """从训练日志文本中提取 step/reward 对。"""
        import re
        # 匹配格式: "  1/128     4    0.750      6.9s ..."
        pattern = r'^\s*(\d+)/(\d+)\s+\d+\s+([\d.]+)'
        rewards = []
        for line in log_text.split('\n'):
            m = re.match(pattern, line)
            if m:
                rewards.append({
                    "step": int(m.group(1)),
                    "total": int(m.group(2)),
                    "reward": float(m.group(3)),
                })
        return rewards if rewards else None
```

### 9.2 设计说明

- `extract_training_data()` 从轨迹的 tool_calls 中提取训练数据，不直接读取 rllm_trl/ 文件
- 数据来源是 hooks 捕获的 tool_response 字段
- 如果 rllm-xx 没有执行数据表面化（没有 Read config.json 等），则对应字段为 None
- traj-analyze-rllm 子 agent 调用此方法获取数据，无需自己解析 JSONL

---

## 10. traj-analyze-xx 通用协议

### 10.1 目的

定义分析 skill 的标准接口，使得:
- 新增分析器（如 traj-analyze-devops）有明确的实现模板
- traj-optimize 能以统一方式消费任何分析器的输出
- 各分析器遵循相同的数据边界约束

### 10.2 输入协议

所有 traj-analyze-xx 分析器的输入只来自:

| 路径 | 用途 |
|------|------|
| `trajectory/output/trajectories/{session_id}/trajectories.jsonl` | 分割后的轨迹（主要输入） |
| `trajectory/output/raw/{session_id}/events.jsonl` | 原始事件（当轨迹信息不足时补充） |
| `trajectory/output/reports/` | 历史报告（跨轮次对比用） |
| `trajectory/output/index.jsonl` | 索引（查找相关 session） |

禁止直接读取被观察 skill 的:
- 内部输出目录（如 `rllm_trl/output/`）
- 源代码文件（如 `rllm_trl/*.py`）
- skill-bank 源文件（如 `skill-bank/rllm/*/base.md`）

### 10.3 输出协议

所有分析器输出到 `trajectory/output/reports/`，格式:

```markdown
# {analyzed-skill} 轨迹分析报告

生成时间: {timestamp}
分析器: {analyzer-skill-name}
分析范围: {scope_description}
数据来源: 轨迹数据 (trajectory/output/)

## 训练执行概览

| Session | 配置摘要 | 结果 | 关键问题 |
|---------|---------|------|---------|
| ... | ... | ... | ... |

## 问题发现

### {N}. {问题标题} [影响: {target_skill}]
**现象**: ...
**证据** (来自轨迹):  ...
**根因假说**: ...
**置信度**: 高/中/低
**建议**: ...

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 | 证据 Session |
|--------|-----------|---------|--------|------|-------------|
| P0/P1/P2 | ... | ... | ... | ... | ... |

## 建议的 Patch 内容

### Patch {N}: {description}
(完整 patch markdown 内容，符合 skill-bank patch 格式)
```

关键要求:
- **数据来源** 字段必须标注为 "轨迹数据"，不能是 "训练日志" 或 "直接读取"
- **证据** 必须引用具体的 session_id 和轨迹中的数据
- **置信度** 标注: 基于多轮轨迹数据的结论为"高"，单轮为"中"，推测性为"低"

### 10.4 领域知识规范

| 允许 | 禁止 |
|------|------|
| 通用的训练动态模式表（模式 → 特征 → 含义） | 硬编码的参数安全范围表 |
| 模式识别方法论（提取事实 → 对比 → 推断） | 固定的 "问题 → skill" 映射表 |
| 从轨迹历史推断出的安全边界（附证据） | "经验表明" 无证据表述 |
| 分析框架（维度、失败模式检测清单） | 预设结论（"lr > X 一定会崩溃"） |

### 10.5 创建新分析器的步骤

1. 在 `skill-bank/traj/` 下创建 `traj-analyze-{domain}/` 目录
2. 创建 `base.md` 包含:
   - `<!-- section:intro -->` — 分析器定位和职责
   - `<!-- section:data-boundary -->` — 数据边界定义（套用通用协议）
   - `<!-- section:analysis-framework -->` — 领域特定的分析维度和模式表
   - `<!-- section:steps -->` — 执行步骤
   - `<!-- section:domain-knowledge -->` — 模式识别指南（非硬编码表）
3. 在 `manifest.yaml` 中注册
4. 在 `skill-bank/bank.yaml` 中添加到 traj group
5. 编译: `python skill-bank/compile.py traj-analyze-{domain}`
6. 在 traj-loop 中注册: 修改 traj-loop 使其可选择不同分析器

---

## 11. 实施计划

### Phase A: rllm-xx 侧 — 数据表面化

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `skill-bank/rllm/rllm-monitor/base.md` | 新增 `<!-- section:data-surfacing -->` | 低 |
| `skill-bank/rllm/rllm-analyze/base.md` | 新增 `<!-- section:data-surfacing -->` | 低 |

### Phase B: traj-analyze-rllm — 数据边界 + 领域知识重构

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `skill-bank/traj/traj-analyze-rllm/base.md` | 新增 `data-boundary` section | 中 |
| `skill-bank/traj/traj-analyze-rllm/base.md` | 重写 `steps` section 的 Step 2 | 中 |
| `skill-bank/traj/traj-analyze-rllm/base.md` | 重写 `domain-knowledge` section | 高 |

### Phase C: traj-loop — Agent 隔离编排

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `skill-bank/traj/traj-loop/base.md` | rules section 追加隔离规则 | 低 |
| `skill-bank/traj/traj-loop/base.md` | 重写 steps section 循环执行部分 | 高 |
| `skill-bank/traj/traj-loop/base.md` | 新增 `agent-isolation` section | 中 |

### Phase D: Python 基础设施

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `trajectory/analyzer/base.py` | 新增 `extract_training_data()` | 中 |
| `trajectory/analyzer/base.py` | 新增 `get_available_training_data()` | 低 |
| `trajectory/analyzer/base.py` | 新增 `_extract_rewards_from_log()` | 低 |

### Phase E: 设计文档更新

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `docs/trajectory-design.md` | 准则 3.1 "应确保" → "必须确保" | 低 |
| `docs/trajectory-design.md` | 新增 Section 3.5 隔离边界准则 | 中 |
| `docs/trajectory-design.md` | 新增 Section 16 traj-analyze-xx 通用协议 | 中 |
| `docs/trajectory-design.md` | 更新 Section 1.3 模块关系图（标注 Agent 边界） | 低 |

### Phase F: 编译 & 验证

```bash
python skill-bank/compile.py --group rllm
python skill-bank/compile.py --group traj
```

### 依赖关系

```
Phase A (数据表面化) ─┐
                      ├── Phase B (数据边界) ── Phase C (Agent 编排) ── Phase F (编译验证)
Phase D (Python) ─────┘
Phase E (文档) ─── 独立，任何时候可做
```

---

## 12. 验证方案

### 12.1 静态验证（编译后）

| 检查项 | 方法 | 预期结果 |
|--------|------|---------|
| traj-loop SKILL.md 包含 Agent 调用 | grep "Agent(" .claude/skills/traj-loop/SKILL.md | 存在 |
| traj-analyze-rllm SKILL.md 无硬编码范围 | grep "1e-6" .claude/skills/traj-analyze-rllm/SKILL.md | 不存在 |
| traj-analyze-rllm SKILL.md 有 data-boundary | grep "data-boundary" .claude/skills/traj-analyze-rllm/SKILL.md | 存在 |
| rllm-monitor SKILL.md 有 data-surfacing | grep "data-surfacing" .claude/skills/rllm-monitor/SKILL.md | 存在 |

### 12.2 运行时验证（执行一轮 traj-loop）

| 检查项 | 方法 | 预期结果 |
|--------|------|---------|
| rllm-train 在子 agent 中执行 | 观察 traj-loop 输出是否使用 Agent 工具 | 是 |
| traj-analyze-rllm 在独立子 agent 中执行 | 观察是否有第二个 Agent 调用 | 是 |
| 分析报告数据来源标注 | 读取报告的 "数据来源" 字段 | "轨迹数据" |
| trajectory/output/raw/ 非空 | ls trajectory/output/raw/ | 有事件文件 |
| 分析器未读取 rllm_trl/ | 检查子 agent 2 的工具调用 | 无 Read rllm_trl/ |

### 12.3 隔离性验证

最严格的验证: 在 traj-analyze-rllm 子 agent 的 prompt 中故意不提供 run_id，观察它是否能从 trajectory/output/ 中自行找到正确的训练数据。

- 如果能: 说明 trajectory/output/ 中有足够数据（数据表面化成功）
- 如果不能: 说明数据表面化不充分，需要加强 Phase A

---

## 附录 A: 与现有设计文档的关系

| 文档 | 关系 |
|------|------|
| `docs/trajectory-design.md` | 本文档是其 Section 3.5 和 Section 16 的详细展开 |
| `docs/skill-bank-design.md` | 本文档依赖其 patch 格式定义，不修改 |
| `trajectory/output/reports/20260502-traj-loop-problems-analysis.md` | 本文档的问题来源 |

## 附录 B: 完整文件修改清单

| 文件 | 改动摘要 |
|------|---------|
| `skill-bank/traj/traj-loop/base.md` | 重写为 Agent 隔离编排，新增 agent-isolation section |
| `skill-bank/traj/traj-analyze-rllm/base.md` | 新增 data-boundary，重写 steps/Step2，重写 domain-knowledge |
| `skill-bank/rllm/rllm-monitor/base.md` | 新增 section:data-surfacing |
| `skill-bank/rllm/rllm-analyze/base.md` | 新增 section:data-surfacing |
| `trajectory/analyzer/base.py` | 新增 extract_training_data(), get_available_training_data() |
| `docs/trajectory-design.md` | 强化 3.1，新增 3.5 + 16，更新模块关系图 |
| `docs/traj-rllm-isolation-design.md` | 本文档（新增） |
