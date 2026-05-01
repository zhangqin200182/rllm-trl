"""Event writer — appends TrajectoryEvents to JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from trajectory.adapter.schema import TrajectoryEvent, Trajectory, SessionSummary
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class EventWriter:
    """Appends events to per-session JSONL files."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def write_event(self, event: TrajectoryEvent) -> None:
        """Append a single event to the session's events.jsonl."""
        session_dir = self.config.raw_dir / event.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        events_file = session_dir / "events.jsonl"
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_events(self, events: List[TrajectoryEvent]) -> None:
        """Append multiple events (batched by session)."""
        for event in events:
            self.write_event(event)


class TrajectoryWriter:
    """Writes segmented trajectories to per-session JSONL files."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def write_trajectory(self, trajectory: Trajectory) -> None:
        """Append a trajectory to the session's trajectories.jsonl."""
        session_dir = self.config.trajectories_dir / trajectory.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        traj_file = session_dir / "trajectories.jsonl"
        line = json.dumps(trajectory.to_dict(), ensure_ascii=False)
        with open(traj_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_trajectories(self, trajectories: List[Trajectory]) -> None:
        """Write multiple trajectories."""
        for traj in trajectories:
            self.write_trajectory(traj)
