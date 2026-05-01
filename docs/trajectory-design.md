# Trajectory 模块设计规范

> Claude Code 交互轨迹的自动捕获、存储、分割与分析系统。通过 LLM 驱动的轨迹分析自动生成 skill-bank patch，形成 skill 持续优化的闭环。

## 1. 背景与目标

### 1.1 问题

rllm-train 通过 skill 驱动 agent RL 训练，skill-bank 管理 skill 的优化。但当前 skill 优化依赖人工经验：

- 用户跑了 8 轮训练实验（见 `docs/rllm-train-skill-design-v2.md`）
- 人工观察每轮的问题（lr 过高崩溃、OOM、catastrophic forgetting...）
- 人工总结优化方案
- 手动写成 skill-bank patches

这个过程耗时、依赖专家经验、且容易遗漏。trajectory 模块的目标是**自动化这个过程**。

### 1.2 目标

1. **自动捕获** — 通过 Claude Code Hooks 拦截 rllm-xx skill 执行过程中的所有交互轨迹
2. **结构化存储** — 按 Session > Conversation > Turn > ToolCall 层次存储
3. **智能分割** — 将连续的工具调用聚合为有意义的轨迹（Trajectory）
4. **LLM 分析** — 调用 LLM 分析轨迹，自动发现问题和优化方向
5. **自动生成 patch** — 将分析结果转化为 skill-bank patch，人工确认后编译

### 1.3 模块关系

```
rllm-xx（被观察者）              traj-xx（观察者）
┌──────────────────┐             ┌──────────────────┐
│ rllm-clarify     │             │ traj-setup       │ ← 一次性初始化
│ rllm-config      │──── 轨迹 ──→│ traj-segment     │ ← 分割轨迹
│ rllm-run         │             │ traj-analyze-*   │ ← LLM 分析（可插拔）
│ rllm-monitor     │             │ traj-optimize    │ ← 生成 patch
│ rllm-analyze     │             │ traj-loop        │ ← 全自动编排
└──────────────────┘             └────────┬─────────┘
        ↑                                 │ patches
        │           ┌─────────────────────┘
        │           ▼
        │     ┌──────────────┐
        └─────│  skill-bank  │ ← 编译更新后的 SKILL.md
              └──────────────┘
```

设计原则:
- **rllm-xx 不知道 traj-xx 的存在** — 完全解耦，rllm-train 不需要做任何修改
- **traj-xx 是通用框架** — 当前主要分析 rllm-train，但同样的机制可以分析任何 skill
- **分析 skill 可插拔** — 不同场景用不同的分析 skill（traj-analyze-rllm、traj-analyze-devops...）
- **skill-bank 是桥梁** — traj-optimize 输出标准的 skill-bank patch，走标准 patch 流程

## 2. 使用场景

### 2.1 前置条件（一次性）

```
/traj-setup
→ 自动写入 .claude/settings.json hooks 配置
→ 创建 trajectory/output/ 目录
→ 将 output/ 加入 .gitignore
→ 之后所有 Claude Code 工具调用自动被捕获
```

### 2.2 场景 1: 手动流程

用户完全控制每一步，适合调试和理解系统行为。

```
第一步: 训练（可跑多轮，积累轨迹）
  /rllm-train "用 qwen-0.5b 训练数学 agent"
  /rllm-train "lr 调低一点再试"
  /rllm-train "换 mixed 难度"

第二步: 分割
  /traj-segment                    # 分割最近的轨迹

第三步: 分析
  /traj-analyze-rllm               # LLM 分析，输出报告
  # 用户阅读 trajectory/output/reports/ 下的报告

第四步: 优化
  /traj-optimize                   # 根据报告生成 skill-bank patch
  # 用户审阅 patch 内容，确认后编译

重复第一步到第四步
```

### 2.3 场景 2: 半自动流程

训练仍由用户手动触发，但分割+分析+优化一条命令完成。

```
第一步: 训练（可跑多轮）
  /rllm-train "..."
  /rllm-train "..."

第二步: 一键分析优化
  /traj-analyze-rllm --optimize
  → 自动: 分割 → 分析 → 生成 patch → 展示报告 → 等待用户确认 patch

重复
```

### 2.4 场景 3: 全自动流程

一个编排 skill 驱动整个循环：训练 → 捕获 → 分割 → 分析 → 优化。用户只需启动一次。

```
/traj-loop "用 qwen-0.5b 训练数学 agent，自动优化 3 轮"

→ Round 1:
    调用 /rllm-train 执行训练（hooks 自动捕获）
    调用 /traj-segment 分割轨迹
    调用 /traj-analyze-rllm 分析
    调用 /traj-optimize 生成 patch
    展示报告和 patch → 用户确认
    编译更新 skill

→ Round 2:
    使用优化后的 skill 再次训练
    分割 → 分析 → 优化 → 用户确认
    ...

→ Round 3:
    ...

→ 输出最终报告: 3 轮优化的效果对比
```

`traj-loop` 是顶层编排 skill，类似 rllm-train 编排 rllm-xx 的关系:

```
traj-loop（编排层）
  ├── /rllm-train      ← 执行训练
  ├── /traj-segment    ← 分割轨迹
  ├── /traj-analyze-rllm ← 分析
  └── /traj-optimize   ← 生成 patch + 用户确认 + 编译
```

**全自动流程中唯一需要人工介入的环节是确认 patch** — 这是设计准则 3.4 的要求。

## 3. 设计准则

### 3.1 轨迹数据完整性准则

**被观察的 skill（如 rllm-xx）应确保把必要信息带给大模型。**

trajectory 通过 hooks 捕获 Claude Code 与 LLM 之间的交互。如果 skill 在执行过程中没有把关键数据（如训练 reward 趋势、perf_stats、错误日志）读取并传递给大模型，那么这些数据就不会出现在轨迹中，分析层也就无法利用。

具体要求:
- rllm-monitor 应读取训练日志并将关键指标传递给大模型
- rllm-analyze 应读取 `perf_stats.json`、`analysis.json`、trajectory JSONL 等文件
- 任何影响决策的数据都应通过工具调用（Read/Bash）进入对话

**如果发现轨迹中缺少关键信息，应优先修改 rllm-xx skill 使其将信息带入对话，而非在 trajectory 模块中额外采集。**

### 3.2 Hook 轻量性准则

Hook 脚本必须在 1 秒内完成。只做格式转换和文件追加，不做分割、不做分析。失败时静默，不影响 Claude Code 正常工作。

### 3.3 分析 Skill 可插拔准则

分析层使用 LLM 而非规则引擎。不同场景构建不同的分析 skill:
- `traj-analyze-rllm` — 分析 rllm-train 训练轨迹
- `traj-analyze-devops` — 未来：分析部署/CI 轨迹
- 每个分析 skill 本身也通过 skill-bank 管理，可以被持续优化

### 3.4 自动生成、人工确认准则

traj-optimize 自动生成完整的 skill-bank patch 内容（包括 markdown 文本），但需要人工确认后才编译生效。

## 4. 需求讨论结论

### 4.1 捕获边界

- **只捕获工程相关轨迹** — 包含工具调用的交互段
- **纯文本讨论不存储** — 没有任何工具调用的 turn 直接跳过
- **两类轨迹**: Skill 轨迹（以 Skill tool 调用为锚点）和自由轨迹（非 skill 触发但包含工具调用）

### 4.2 数据层次

原始数据层（Hooks 捕获）:

```
Session > Conversation > Turn > ToolCall
```

- **Session** — 一次 Claude Code 会话
- **Conversation** — 主对话是一个 Conversation，每次 Agent tool 调用创建子 Conversation
- **Turn** — 一次 user→assistant 交互（可能包含多个 tool calls）
- **ToolCall** — 最小粒度

分析数据层（Segmenter 生成）:

```
Trajectory = 一组有意义的 ToolCalls 聚合
```

Trajectory 是叠加在原始层次上的分析概念，不改变原始存储结构。

### 4.3 其他决策

- 子 agent 轨迹作为父轨迹的**嵌套部分**存储，保留因果链
- 轨迹分割策略**可插拔**，做成 Skill（traj-segment），通过 skill-bank 管理
- 暂不做跨 turn 合并，元数据中记录文件关联供分析层后续关联
- 通过 **Adapter 层**与 Hooks JSON schema 解耦
- 存储格式: **JSONL**
- `trajectory/` 目录与 `rllm_trl/`、`skill-bank/` 并行

## 5. traj-xx Skill 体系

### 5.1 Skill 清单

| Skill | 类型 | 触发方式 | 职责 |
|-------|------|---------|------|
| **traj-setup** | 工具 | 手动（一次性） | 配置 hooks、创建目录、初始化 |
| **traj-segment** | 工具 | 手动或被编排调用 | 对原始事件进行轨迹分割 |
| **traj-analyze-rllm** | 分析 | 手动或被编排调用 | 调用 LLM 分析 rllm-train 轨迹 |
| **traj-optimize** | 工具 | 手动或被编排调用 | 根据分析报告生成 skill-bank patch |
| **traj-status** | 工具 | 手动 | 查看捕获状态、session 列表、轨迹统计 |
| **traj-loop** | 编排 | 手动 | 全自动编排: 训练 → 分割 → 分析 → 优化 → 循环 |

### 5.2 Skill 层次关系

```
traj-loop（顶层编排）
  ├── /rllm-train          ← 执行训练（hooks 自动捕获轨迹）
  ├── /traj-segment        ← 分割轨迹
  ├── /traj-analyze-rllm   ← LLM 分析
  └── /traj-optimize       ← 生成 patch + 用户确认 + 编译
```

traj-loop 和 rllm-train 的关系: traj-loop 在外层编排，rllm-train 在内层编排自身的 phase。两层编排互不感知。

### 5.3 traj-analyze-* 可插拔分析 Skill

分析层是 trajectory 系统的核心价值，也是最需要持续优化的部分。

**为什么做成独立 skill:**
- 不同场景需要不同的领域知识来分析轨迹
- 分析 skill 本身通过 skill-bank 管理和优化
- 轨迹分析结果可以反向优化分析 skill 自身（meta-optimization）

**当前实现:**

`traj-analyze-rllm` — 分析 rllm-train 训练轨迹:
- 理解训练动态（reward 趋势、loss、grad_norm）
- 识别失败模式（OOM、参数冲突、catastrophic forgetting、格式退化）
- 关联多轮训练结果，发现参数安全边界
- 输出与 `docs/rllm-train-skill-design-v2.md` 类似的结构化分析

**扩展示例:**
- `traj-analyze-devops` — 未来：分析 CI/CD 轨迹

**在 skill-bank 中的位置:**

```
skill-bank/
├── rllm/                    # 训练相关
│   ├── rllm-train/
│   ├── rllm-config/
│   └── ...
├── traj/                    # 轨迹相关（新增 group）
│   ├── traj-setup/
│   ├── traj-segment/
│   ├── traj-analyze-rllm/
│   ├── traj-optimize/
│   ├── traj-status/
│   └── traj-loop/
```

### 5.4 v2 文档自动化映射

展示 `docs/rllm-train-skill-design-v2.md` 中每个人工发现如何被 trajectory 系统自动化:

| v2 中的人工发现 | 自动化路径 |
|----------------|-----------|
| "0.5B 在 lr=2e-5 时策略崩溃" | traj-analyze-rllm 从轨迹中读取 lr 配置和 reward 趋势，自动关联 |
| "num_generations=6 启动失败" | traj-analyze-rllm 检测到 Bash 工具调用返回 ValueError |
| "monitor 静默失效" | traj-analyze-rllm 检测到 Monitor 工具长时间无输出 |
| "提出模型级安全配置表" | traj-optimize 根据多轮训练的 lr/reward 数据自动生成参数安全表 |
| "建议增加 difficulty 参数" | traj-analyze-rllm 发现 reward=1.0+loss=0 模式，建议增加数据难度 |
| 整份 v2 文档 | traj-analyze-rllm 输出的分析报告 |

**关键前提**: 这些信息能被自动发现，依赖于 rllm-xx skill 在执行过程中把必要数据通过工具调用带入了对话（见 3.1 数据完整性准则）。

## 6. 数据模型

### 6.1 原始数据层

#### Session

```python
@dataclass
class Session:
    session_id: str                    # Claude Code session ID
    start_time: datetime
    end_time: Optional[datetime]
    transcript_path: str               # 原始 transcript JSONL 路径
    conversations: List[Conversation]
    metadata: Dict[str, Any]           # 工作目录、git branch 等
```

#### Conversation

```python
@dataclass
class Conversation:
    conversation_id: str               # 主对话用 session_id，子对话用 agent 生成的 ID
    parent_conversation_id: Optional[str]
    turns: List[Turn]
    is_subagent: bool
```

#### Turn

```python
@dataclass
class Turn:
    turn_index: int                    # 在 conversation 内的序号
    user_message: Optional[str]
    assistant_response: Optional[str]
    tool_calls: List[ToolCall]
    has_tool_calls: bool
    timestamp: datetime
```

#### ToolCall

```python
@dataclass
class ToolCall:
    tool_name: str                     # Bash, Read, Edit, Write, Skill, Agent, ...
    tool_input: Dict[str, Any]
    tool_response: Optional[Dict[str, Any]]
    timestamp: datetime
    duration_ms: Optional[float]
    success: bool
    files_touched: List[str]           # 从 tool_input 提取
```

### 6.2 分析数据层

#### Trajectory

```python
@dataclass
class Trajectory:
    trajectory_id: str
    session_id: str
    conversation_id: str
    trajectory_type: str               # "skill" | "free"

    # Skill 轨迹专有
    skill_name: Optional[str]
    skill_args: Optional[str]

    # 内容
    tool_calls: List[ToolCall]
    nested_conversations: List[Conversation]  # 子 agent 对话（嵌套）

    # 元数据
    start_time: datetime
    end_time: datetime
    duration_ms: float
    files_touched: List[str]

    # 分类
    intent_tags: List[str]             # exploration, implementation, testing, debugging
    outcome: str                       # success, failure, partial, abandoned
```

### 6.3 存储格式

所有数据使用 JSONL 格式。

**原始事件文件**（`output/raw/{session_id}/events.jsonl`）:

```json
{"type": "tool_call", "session_id": "abc", "conversation_id": "abc", "turn_index": 0, "tool_name": "Read", "tool_input": {"file_path": "/path"}, "tool_response": {"content": "..."}, "timestamp": "2026-05-01T10:00:00Z", "success": true}
{"type": "turn_end", "session_id": "abc", "conversation_id": "abc", "turn_index": 0, "timestamp": "2026-05-01T10:00:10Z"}
{"type": "session_end", "session_id": "abc", "timestamp": "2026-05-01T10:30:00Z"}
```

**轨迹文件**（`output/trajectories/{session_id}/trajectories.jsonl`）:

```json
{"trajectory_id": "traj_001", "session_id": "abc", "type": "skill", "skill_name": "rllm-train", "tool_calls": [...], "nested_conversations": [...], "intent_tags": ["implementation"], "outcome": "success", "duration_ms": 15000}
{"trajectory_id": "traj_002", "session_id": "abc", "type": "free", "tool_calls": [...], "intent_tags": ["debugging"], "outcome": "failure", "duration_ms": 45000}
```

**索引文件**（`output/index.jsonl`）:

```json
{"session_id": "abc", "start_time": "2026-05-01T10:00:00Z", "trajectory_count": 5, "skill_trajectories": 2, "free_trajectories": 3, "skills_used": ["rllm-train", "rllm-config"]}
```

## 7. 模块结构

```
trajectory/
├── hooks/                      # Hook 入口脚本（Claude Code 直接调用）
│   ├── post_tool.py            # PostToolUse handler — 实时记录工具调用
│   └── on_stop.py              # Stop handler — turn/session 结束处理
│
├── adapter/                    # Hooks JSON → 内部格式（解耦层）
│   ├── schema.py               # 内部数据模型定义
│   └── hooks_adapter.py        # Hooks stdin JSON → TrajectoryEvent
│
├── segmenter/                  # 轨迹分割（可插拔策略）
│   ├── base.py                 # SegmenterStrategy 接口
│   ├── skill_segmenter.py      # Skill 轨迹识别
│   ├── free_segmenter.py       # 自由轨迹分割（启发式）
│   └── registry.py             # 策略注册与选择
│
├── store/                      # 存储层
│   ├── writer.py               # 事件写入 JSONL
│   ├── reader.py               # 轨迹查询与读取
│   └── index.py                # 索引管理
│
├── analyzer/                   # 分析层 Python 基础设施
│   ├── base.py                 # LLM 调用封装
│   └── report.py               # 报告格式化输出
│
├── optimizer/                  # 优化层 Python 基础设施
│   ├── patch_generator.py      # 生成 skill-bank patch 文件
│   └── compiler_bridge.py      # 调用 skill-bank/compile.py
│
├── output/                     # 运行时输出（gitignore）
│   ├── raw/                    # 原始 hook 事件，按 session 组织
│   │   └── {session_id}/
│   │       └── events.jsonl
│   ├── trajectories/           # 分割后的轨迹
│   │   └── {session_id}/
│   │       └── trajectories.jsonl
│   ├── reports/                # 分析报告
│   │   └── {timestamp}-report.md
│   └── index.jsonl             # 全局索引
│
└── config.py                   # 模块配置
```

**注意**: 分析的核心逻辑在 `traj-analyze-*` skill 的 SKILL.md 中，`analyzer/` Python 代码只提供基础设施（读取轨迹、调用 LLM、输出报告）。具体的分析策略和领域知识由 skill 指令控制。

## 8. 捕获层设计

### 8.1 Claude Code Hooks 配置

由 `traj-setup` skill 自动写入 `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python trajectory/hooks/post_tool.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python trajectory/hooks/on_stop.py"
          }
        ]
      }
    ],
    "SubagentStop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python trajectory/hooks/on_stop.py --subagent"
          }
        ]
      }
    ]
  }
}
```

### 8.2 PostToolUse Hook（post_tool.py）

**输入**（stdin JSON）:

```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../transcript.jsonl",
  "tool_name": "Edit",
  "tool_input": { "file_path": "...", "old_string": "...", "new_string": "..." },
  "tool_response": { ... }
}
```

**处理**: 读取 stdin → HooksAdapter 转换 → Writer 追加到 events.jsonl。失败时静默。

### 8.3 Stop Hook（on_stop.py）

写入 `turn_end` 事件。`--subagent` 标记子 conversation 结束。

### 8.4 过滤规则

| 条件 | 处理 |
|------|------|
| 包含工具调用的 turn | 存储 |
| 纯文本讨论 | 跳过 |

## 9. 适配器层设计

### 9.1 TrajectoryEvent

```python
@dataclass
class TrajectoryEvent:
    """适配器输出 — 下游所有模块只依赖此格式"""
    event_type: str                    # "tool_call" | "turn_end" | "session_end"
    session_id: str
    conversation_id: str
    timestamp: datetime
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    tool_response: Optional[Dict[str, Any]]
    success: Optional[bool]
    files_touched: List[str]
    raw_hook_data: Dict[str, Any]      # 保留原始数据
```

### 9.2 HooksAdapter

```python
class HooksAdapter:
    """Claude Code Hooks JSON → TrajectoryEvent — 唯一的 schema 耦合点"""

    def adapt(self, hook_type: str, stdin_json: dict) -> TrajectoryEvent: ...
    def extract_files(self, tool_name: str, tool_input: dict) -> List[str]: ...
    def infer_success(self, tool_name: str, tool_response: dict) -> bool: ...
```

Hooks schema 变化时只需修改 `HooksAdapter`，下游不受影响。

### 9.3 Conversation ID 推断

- 主对话: `conversation_id = session_id`
- 子 agent: 检测 `tool_name == "Agent"` 创建新 conversation，ID 格式 `{session_id}:sub_{index}`
- `SubagentStop` 事件标记子 conversation 结束

## 10. 轨迹分割层设计

### 10.1 策略接口

```python
class SegmenterStrategy(ABC):
    """轨迹分割策略接口 — 可插拔替换"""

    @abstractmethod
    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]: ...

    @abstractmethod
    def name(self) -> str: ...
```

### 10.2 Skill Segmenter

```
1. 扫描事件流，找到 tool_name == "Skill" 的事件作为起点
2. 收集后续所有工具调用，直到:
   a. 下一个 Skill 调用
   b. 下一个 user 消息（turn 边界）
   c. session 结束
3. 包含嵌套的子 conversation
4. 输出 Trajectory(type="skill")
```

### 10.3 Free Segmenter

```
1. 取 Skill Segmenter 未覆盖的事件
2. 按 Turn 边界切分
3. Turn 内按文件亲和性聚合:
   Read("a.py") → Edit("a.py") → Bash("python a.py") 归为一组
4. 打意图标签:
   - exploration: 连续 Read/Grep
   - implementation: 包含 Edit/Write
   - testing: Bash 运行测试
   - debugging: test → fix → test 循环
5. 输出 Trajectory(type="free")
```

### 10.4 策略注册

```python
class SegmenterRegistry:
    def __init__(self):
        self._strategies = {}
        self.register(SkillSegmenter())
        self.register(FreeSegmenter())

    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]:
        """先 Skill Segmenter，剩余事件用 Free Segmenter"""
        skill_trajs = self._strategies["skill"].segment(events)
        covered = {id(e) for t in skill_trajs for e in t.tool_calls}
        remaining = [e for e in events if id(e) not in covered]
        free_trajs = self._strategies["free"].segment(remaining)
        return skill_trajs + free_trajs
```

分割策略通过 `traj-segment` skill 管理，可以通过 skill-bank patch 优化。

## 11. 分析层设计

### 11.1 架构

```
traj-analyze-rllm (SKILL.md)        trajectory/analyzer/ (Python)
┌──────────────────────────┐        ┌────────────────────────┐
│ 领域知识:                 │        │ 基础设施:               │
│ - 训练动态理解            │  调用   │ - 读取轨迹文件          │
│ - 失败模式识别            │ ─────→ │ - 格式化 LLM prompt     │
│ - 参数安全范围判断         │        │ - 调用 LLM API          │
│ - 优化建议生成策略         │        │ - 输出报告              │
│                          │        │                        │
│ 可通过 skill-bank 优化    │        │ 稳定，不频繁修改        │
└──────────────────────────┘        └────────────────────────┘
```

核心智能在 SKILL.md 中，Python 代码只提供基础设施。这样分析策略可以通过 skill-bank patch 快速迭代，不需要改 Python 代码。

### 11.2 分析流程

```
1. traj-analyze-rllm skill 被调用
2. Skill 指令引导 LLM:
   a. 读取轨迹文件
   b. 识别 rllm-train 相关轨迹
   c. 分析每条轨迹的配置、结果、失败模式
   d. 跨轨迹关联: 多轮训练参数变化与结果关系、安全边界推断
   e. 生成分析报告 + 结构化优化建议
3. 报告写入 output/reports/
```

### 11.3 分析报告格式

```markdown
# 轨迹分析报告

生成时间: 2026-05-01 10:00:00
分析范围: 最近 7 天
rllm-train 轨迹: 8 条

## 训练执行概览

| Session | 配置摘要 | 结果 | 关键问题 |
|---------|---------|------|---------|
| abc123 | qwen-0.5b, lr=5e-6, 2ep | reward=0.50 | 正常 |
| def456 | qwen-0.5b, lr=2e-5, 4ep | reward→0 | lr 过高 |

## 问题发现

### 1. 参数安全范围不足 [影响: rllm-config]
**现象**: 2/8 轮训练因 lr 过高导致策略崩溃
**证据**: session def456 中 lr=2e-5 后 reward 从 0.45 降到 0
**建议**: 增加模型级安全配置表

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 |
|--------|-----------|---------|--------|------|
| P0 | rllm-config | param-ranges | append | 增加模型级安全配置表 |
| P0 | rllm-monitor | anomaly-detection | replace | 通用化 grep 模式 |

## 建议的 Patch 内容
（完整的 patch markdown 内容，可直接用于 skill-bank）
```

### 11.4 SkillOptimizationSuggestion

```python
@dataclass
class SkillOptimizationSuggestion:
    skill_name: str                    # 目标 skill
    target_section: str                # patch 的 target_section
    action: str                        # replace | append | prepend | insert_after
    description: str
    rationale: str                     # 依据
    priority: str                      # P0 | P1 | P2
    patch_content: str                 # 完整 patch 内容
    source_sessions: List[str]         # 证据来源
```

## 12. 优化层设计

### 12.1 traj-optimize 流程

```
1. 读取最新的分析报告
2. 解析 SkillOptimizationSuggestion 列表
3. 逐条展示给用户: skill_name、description、patch 内容预览
4. 用户选择: 接受 / 修改 / 跳过
5. 对接受的建议:
   a. 生成 skill-bank patch 文件
   b. 更新 manifest.yaml
6. 用户确认后编译: python skill-bank/compile.py --group <groups>
```

### 12.2 Patch 生成

生成的 patch 遵循 skill-bank 规范:

```markdown
---
id: "{NNN}-traj-{description}"
target_section: "{section}"
action: {action}
description: "{description}"
source: "trajectory analysis, sessions: {session_ids}"
created: "{date}"

depends_on: []
conflicts_with: []
status: active
superseded_by: ""
---

{patch_content}
```

文件放置: `skill-bank/<group>/<skill>/patches/{NNN}-traj-{description}.md`

前缀 `traj-` 标识来源为轨迹分析。

## 13. 配置

```python
@dataclass
class TrajectoryConfig:
    output_dir: str = "trajectory/output"

    # 捕获
    capture_all_tools: bool = True
    core_tools: List[str] = field(default_factory=lambda: [
        "Bash", "Read", "Edit", "Write", "Skill", "Agent",
        "AskUserQuestion", "EnterPlanMode"
    ])

    # 分割
    default_segmenter: str = "default"
    file_affinity_threshold: float = 0.8

    # 分析
    analysis_lookback_days: int = 7
    min_trajectories_for_analysis: int = 5
```

## 14. 注意事项

### 14.1 性能

Hook 脚本 < 1s（只做文件追加）。分割和分析是离线后处理。

### 14.2 容错

Hook 失败静默。事件写入追加模式。分割和分析可重复执行（幂等）。

### 14.3 隐私

`output/` 加入 `.gitignore`。不存储纯文本讨论。

### 14.4 TrajProxy 借鉴

- **元数据+详情分离** — index.jsonl vs events.jsonl
- **JSONL+归档** — 未来可扩展 GZIP 归档
- **层级化 ID** — Session > Conversation > Turn > ToolCall

## 15. 扩展性

### 15.1 新增分析场景

在 `skill-bank/traj/` 下创建新分析 skill（如 `traj-analyze-devops/`），复用 `trajectory/analyzer/` 基础设施。

### 15.2 新增分割策略

实现 `SegmenterStrategy` 接口，注册到 `SegmenterRegistry`。通过 `traj-segment` skill 管理。

### 15.3 Meta-optimization

分析结果可反向优化: 分析 skill 自身、分割策略、捕获过滤规则。
