---
description: Segments raw trajectory events into meaningful trajectory units. Applies
  skill segmenter first, then free segmenter on remaining events.
metadata:
  categories:
  - trajectory
  - analysis
  version: 1.0.0
name: traj-segment
---


# traj-segment — 轨迹分割

你是轨迹分割工具。你的职责是将 trajectory/output/raw/ 中的原始事件流分割为有意义的轨迹单元（Trajectory），存储到 trajectory/output/trajectories/。

## 执行步骤

### 1. 读取原始事件

```python
from trajectory.store.reader import EventReader
from trajectory.config import DEFAULT_CONFIG

reader = EventReader(DEFAULT_CONFIG)
sessions = reader.list_sessions()
```

列出可用 session，默认处理最近的未分割 session。

### 2. 执行分割

```python
from trajectory.segmenter.registry import SegmenterRegistry

registry = SegmenterRegistry()
events = reader.read_session_events(session_id)
trajectories = registry.segment(events)
```

分割策略:
1. **Skill Segmenter** — 以 `Skill` 工具调用为锚点，收集后续所有工具调用直到下一个 Skill 调用或 turn 边界
2. **Free Segmenter** — 对 Skill Segmenter 未覆盖的事件，按 turn 边界切分，按文件亲和性聚合

### 3. 写入轨迹

```python
from trajectory.store.writer import TrajectoryWriter
from trajectory.store.index import IndexManager

writer = TrajectoryWriter(DEFAULT_CONFIG)
writer.write_trajectories(trajectories)

index = IndexManager(DEFAULT_CONFIG)
index.update_session(session_id, trajectories)
```

### 4. 输出报告

输出分割结果摘要:
```
Session: {session_id}
轨迹数: {total} (skill: {skill_count}, free: {free_count})
Skills: {skill_names}
文件: {file_count} 个文件被操作
```

## 参数

- 无参数: 处理最近未分割的 session
- `--session <id>`: 指定 session
- `--all`: 处理所有未分割的 session
- `--resegment`: 重新分割已处理的 session
