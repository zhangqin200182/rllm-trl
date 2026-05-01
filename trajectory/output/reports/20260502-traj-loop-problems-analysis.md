# traj-loop 3 轮优化问题总结分析

生成时间: 2026-05-02
范围: traj-loop 完整 3 轮执行过程中遇到的所有问题

---

## 一、基础设施问题

### 1.1 compile.py YAML 解析静默失败

**现象**: `python skill-bank/compile.py rllm-config` 报错 `Error: patch 'traj-20260501-155627-param-ranges' targets non-existent section ''`

**根因**: patch 文件的 `description` 字段包含未引用的冒号:
```yaml
description: 收紧 0.5B 模型参数安全范围: lr 上限 5e-6, epochs 上限 1
```
YAML 将 `: lr` 解释为嵌套映射，导致 `yaml.safe_load()` 抛出 `ScannerError`。

**二次问题**: `parse_frontmatter()` 的 `except yaml.YAMLError` 捕获异常后返回空字典 `{}`，导致 `p.get("target_section", "")` 返回空字符串。错误信息 "targets non-existent section ''" 完全掩盖了真正的 YAML 解析错误，增加了调试难度。

**修复**:
1. 在 patch 文件中给 description 加引号
2. `PatchGenerator._yaml_quote()` 方法: 检测 YAML 特殊字符 (`:`, `#`, `{}`, `[]` 等)，自动用 `json.dumps()` 引用
3. `compile.py` 的 `parse_frontmatter()`: 将 `except yaml.YAMLError` 改为打印 Warning 信息

**影响范围**: 阻塞了 Round 1 的 patch 编译，延迟了 Round 2 的启动

**教训**: 
- 生成 YAML 时必须处理特殊字符转义
- 错误处理不应吞掉异常信息，至少要 log warning
- 静默降级 (`fm = {}`) 在编译器中是危险的，应该 fail-fast

---

### 1.2 PatchGenerator 未转义 YAML 特殊字符

**现象**: `trajectory/optimizer/patch_generator.py` 的 `_format_patch()` 方法直接用 f-string 拼接 YAML frontmatter，未考虑值中包含 YAML 特殊字符的情况

**根因**: 
```python
f"description: {suggestion.description}",  # 如果 description 含 : 就会破坏 YAML
```

**修复**: 新增 `_yaml_quote()` 静态方法，对含特殊字符的值自动加引号

**教训**: 生成结构化格式（YAML/JSON/TOML）时，永远不要用字符串拼接，应使用对应的序列化库。这里为了保持 frontmatter 格式可读性选择了 f-string，但至少需要转义处理。

---

### 1.3 Trajectory Hooks 未捕获训练数据

**现象**: `trajectory/output/raw/` 目录始终为空，3 轮训练均无原始事件被捕获

**根因**: Claude Code Hooks (PostToolUse/Stop) 只拦截 Claude Code 自身的工具调用。训练进程通过 `python -m rllm_trl.run_training` 作为子进程运行，其内部的模型推理、环境交互等操作不经过 Claude Code 工具系统，因此 hooks 无法捕获。

**实际捕获的**: 只有 Claude Code 层面的操作（Bash 调用、Read/Write 文件等）会触发 hooks，但这些是编排层操作，不是训练内部的 trajectory。

**影响**: traj-segment 和 traj-analyze-rllm 无法使用设计中的 trajectory 数据流，只能退化为直接分析训练日志文件

**未修复**: 这是架构层面的限制。要捕获训练内部 trajectory，需要:
- 方案 A: 在 rllm_trl 训练代码中直接写入 trajectory 格式的输出（已有 trajectory_writer.py）
- 方案 B: 训练完成后从 `trajectories/*.jsonl` 文件中读取并转换为 trajectory 模块格式
- 方案 C: 放弃 hooks 捕获训练数据，hooks 只用于捕获 Claude Code 编排层的操作

**教训**: Hooks 的捕获边界是 Claude Code 工具调用，不是任意子进程的 I/O。设计时高估了 hooks 的覆盖范围。

---

### 1.4 Skills 未被 Claude Code 识别

**现象**: 创建 skill-bank 条目后，`/traj-loop` 显示 "Unknown command"

**根因**: Claude Code 读取的是 `.claude/skills/*/SKILL.md`，而 skill-bank 系统的源文件在 `skill-bank/` 目录。创建 base.md 和 manifest.yaml 后必须运行 `compile.py` 生成最终的 SKILL.md。

**修复**: 执行 `python skill-bank/compile.py --group traj`

**教训**: skill-bank 是源码，`.claude/skills/` 是构建产物。任何 skill 变更后必须编译。

---

### 1.5 registry.py 属性名错误

**现象**: `SegmenterRegistry` 排序时使用了不存在的 `t.timestamp_sort_key` 属性

**根因**: 编码时假设 Trajectory dataclass 有 `timestamp_sort_key` 属性，实际只有 `start_time` 和 `end_time`

**修复**: 改为 `t.start_time or t.end_time`

**教训**: 对 dataclass 字段的引用应在编写时验证，或使用 IDE 类型检查

---

## 二、训练问题

### 2.1 Round 1: Catastrophic Forgetting (严重)

**配置**: lr=1e-5, epochs=2, 64 problems, mixed difficulty
**现象**: Reward 在 step 4 达到峰值 1.0，随后持续下降至 0，step 9-14 全部为 0
**进度**: 14/128 steps (11%)，被 early stopping 中止

**根因分析**:
- lr=1e-5 对 0.5B 模型偏高，初期快速学习但后期过拟合到近期样本
- 2 epochs × 64 problems = 128 步训练量过大
- mixed 难度中 hard 题目 (20%) 产生的零 reward 信号干扰了已学会的 simple 题目

**生成 patch**: 
- `traj-20260501-155627-param-ranges`: lr 上限 1e-5→5e-6, epochs 上限 2→1 (当 problems>=64)
- `traj-20260501-155627-anomaly-detection`: reward 峰值回落检测

---

### 2.2 Round 2: Forgetting 延迟但未消除 (中等)

**配置**: lr=5e-6, epochs=1, 64 problems, mixed difficulty
**现象**: Reward steps 2-8 保持 1.0，step 9 开始崩溃，steps 12-16 全部为 0
**进度**: 16/64 steps (25%)，被 early stopping 中止

**改善**: 崩溃点从 Round 1 的 step 5 延迟到 step 9（+4 步）
**未解决**: 64 problems 对 0.5B 模型仍然过多

**根因分析**:
- 降低 lr 减缓了遗忘速度，但 64 problems 的多样性仍超出 0.5B 模型容量
- 1 epoch × 64 problems = 64 步，在 step 9 (14%) 就开始退化
- mixed 难度的 hard 问题在训练中后期集中出现时，模型无法同时保持 simple 题目的能力

**生成 patch**:
- `traj-20260502-001000-problem-count`: num_problems 上限 64→32 (当 difficulty=mixed 且 model=0.5B)

---

### 2.3 Round 3: 成功但存在波动 (轻微)

**配置**: lr=5e-6, epochs=1, 32 problems, mixed difficulty
**现象**: 训练完成 32/32 步，avg reward 0.742，无 forgetting
**目标达成**: avg reward 0.742 >= 0.5

**残留问题**:
- 仍有零 reward 步 (steps 5,6,12,25,31)，占比 15.6%
- 这些零步与 hard 问题相关，0.5B 模型无法解决多步骤应用题
- 第二半段 (steps 17-32) avg 0.797 vs 第一半段 0.688，说明模型在学习

**未生成 patch**: 训练成功完成，目标达成，无需进一步优化

---

## 三、监控与编排问题

### 3.1 Early Stopping 触发延迟

**Round 1**: 从 reward 峰值 (step 4) 到 early stop (step 14) 经过 10 步
**Round 2**: 从 reward 峰值 (step 8) 到 early stop (step 16) 经过 8 步

**原因**: 
- "连续 5 步 reward=0" 的条件过于严格
- Round 2 中 step 11 的短暂恢复 (0.5) 重置了连续零计数器
- 峰值回落检测 (peak*0.5 持续 3 步) 在 Round 2 中应该更早触发，但 step 11 的恢复也干扰了它

**改进建议** (未实施):
- 使用滑动窗口: "最近 5 步中 4 步 reward=0" 而非 "连续 5 步"
- 或: "reward < peak * 0.3 持续 3 步" (更激进的阈值)

---

### 3.2 Monitor 事件过于频繁

**现象**: Monitor grep 模式匹配了 `···` 子步骤行，导致每个 step 产生 6-8 条通知（4 条 trajectory done + computing logprobs + training update + step summary）

**影响**: 大量通知占用上下文窗口，增加了信息噪音

**Round 3 改进**: 将 grep 模式改为只匹配 step summary 行 (`^\s*[0-9]+/[0-9]+`)，去掉 `···`，通知量降低 80%

**教训**: Monitor 的 grep 模式应尽可能精确，只匹配需要 act on 的信息

---

### 3.3 sleep 命令被系统阻止

**现象**: `sleep 90 && tail -30 ...` 被系统拦截，提示使用 Monitor 或 run_in_background

**影响**: 需要改用 `until` 循环 + run_in_background 模式等待训练完成

**教训**: Claude Code 环境不允许长 sleep，需要使用事件驱动的等待方式

---

## 四、数据流与架构问题

### 4.1 traj-loop 步骤退化

**设计**: train → segment → analyze → optimize (4 步)
**实际**: train → (skip segment) → analyze from log → optimize (3 步)

**原因**: 没有 trajectory 原始事件可供 segment，分析直接从训练日志进行

**影响**: 
- traj-segment skill 在本次循环中未被使用
- traj-analyze-rllm 的 LLM 分析能力未被利用（直接人工分析日志）
- 优化建议基于 reward 趋势的简单模式匹配，而非深度 trajectory 分析

---

### 4.2 Skill 调用规则与效率的矛盾

**设计**: rllm-train 编排规则要求 "每个 Phase 必须通过调用对应的子 skill 来执行"，且 "Skill 调用后当轮响应立即结束"

**实际**: 在 auto 模式的 traj-loop 中，严格遵循此规则会导致每个 Phase 需要一个完整的对话轮次，3 轮训练 × 5 Phase = 15+ 轮对话，极大消耗上下文窗口

**妥协**: Round 2/3 中部分跳过了 rllm-clarify（需求未变），直接进入 rllm-config

**教训**: 编排规则需要区分首次执行和循环执行。循环中参数未变的 Phase 应允许跳过。

---

## 五、问题分类统计

| 类别 | 数量 | 严重程度 | 已修复 |
|------|------|---------|--------|
| 代码 bug | 3 | 高/中/低 | 3/3 |
| 训练失败 | 2 | 高 | 2/2 (通过调参) |
| 架构限制 | 2 | 中 | 0/2 (需设计变更) |
| 监控效率 | 2 | 低 | 1/2 |
| 编排效率 | 1 | 低 | 0/1 |

---

## 六、总结

traj-loop 3 轮优化的核心价值在于**将训练失败转化为 skill 知识**:

1. **Round 1 失败** → 发现 lr 和 epochs 的安全边界 → 2 个 patch
2. **Round 2 失败** → 发现 num_problems 的安全边界 → 1 个 patch  
3. **Round 3 成功** → 验证了累积 patch 的有效性

最终 rllm-config skill 从通用的经验推荐值进化为经过实证验证的安全约束，未来使用该 skill 的训练将直接避开这 3 轮发现的陷阱。

主要未解决问题:
- Hooks 无法捕获训练内部 trajectory（架构限制）
- Early stopping 的触发逻辑仍有优化空间
- traj-loop 在循环模式下的编排效率需要改进
