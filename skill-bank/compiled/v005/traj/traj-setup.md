---
description: One-time initialization of trajectory capture system. Configures Claude
  Code hooks and creates output directories.
metadata:
  categories:
  - trajectory
  - setup
  version: 1.0.0
name: traj-setup
---


# traj-setup — 轨迹捕获初始化

你是 trajectory 模块的初始化工具。你的职责是配置 Claude Code Hooks，使所有后续的工具调用被自动捕获到 trajectory/output/ 目录。

## 执行步骤

### 1. 检查当前状态

读取 `.claude/settings.json`，检查是否已有 trajectory hooks 配置。

### 2. 写入 Hooks 配置

将以下 hooks 配置合并到 `.claude/settings.json` 中（保留已有配置）:

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

### 3. 创建输出目录

```bash
mkdir -p trajectory/output/{raw,trajectories,reports}
```

### 4. 验证 .gitignore

确认 `trajectory/.gitignore` 包含 `trajectory/output/`。

### 5. 确认

输出完成确认:
```
✓ Trajectory hooks 已配置
✓ 输出目录已创建
✓ 所有后续工具调用将自动被捕获到 trajectory/output/raw/
```

## 注意事项

- 此 skill 只需运行一次
- 不会影响已有的 hooks 配置（合并而非覆盖）
- hooks 脚本执行失败不会影响 Claude Code 正常工作
- 如需停止捕获，从 `.claude/settings.json` 中删除 trajectory 相关的 hooks 条目即可
