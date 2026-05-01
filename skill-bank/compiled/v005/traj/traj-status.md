---
description: Shows trajectory capture status, session list, and statistics.
metadata:
  categories:
  - trajectory
  - status
  version: 1.0.0
name: traj-status
---


# traj-status — 轨迹状态查看

你是轨迹状态查看工具。你的职责是展示当前 trajectory 系统的运行状态、已捕获的 session 列表和统计信息。

## 执行步骤

### 1. 检查 hooks 配置

读取 `.claude/settings.json`，确认 trajectory hooks 是否已配置。

### 2. 读取索引

```python
from trajectory.store.index import IndexManager
from trajectory.store.reader import EventReader, TrajectoryReader
from trajectory.config import DEFAULT_CONFIG

index = IndexManager(DEFAULT_CONFIG)
sessions = index.list_sessions()
```

### 3. 输出状态报告

```
Trajectory 状态
===============
Hooks: {已配置/未配置}
Sessions: {count}
轨迹总数: {total} (skill: {skill_count}, free: {free_count})

最近 Sessions:
  {session_id}  {start_time}  轨迹: {traj_count}  Skills: {skills_used}
  ...

最近报告:
  {report_path}  {timestamp}
  ...
```

### 4. 详细模式

带 `--detail <session_id>` 参数时，展示指定 session 的详细轨迹列表:
```
Session: {session_id}
时间: {start} ~ {end}
轨迹:
  1. [skill] rllm-train  {duration}  {outcome}  {files_count} files
  2. [free]  exploration  {duration}  {outcome}  {files_count} files
  ...
```
