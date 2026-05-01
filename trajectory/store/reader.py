"""Trajectory and event reader — queries stored data."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from trajectory.adapter.schema import TrajectoryEvent, Trajectory, SessionSummary
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class EventReader:
    """Reads raw events from JSONL files."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def read_session_events(self, session_id: str) -> List[TrajectoryEvent]:
        """Read all events for a session."""
        events_file = self.config.raw_dir / session_id / "events.jsonl"
        if not events_file.exists():
            return []
        return self._read_events_file(events_file)

    def list_sessions(self) -> List[str]:
        """List all session IDs that have raw events."""
        raw_dir = self.config.raw_dir
        if not raw_dir.exists():
            return []
        return sorted(
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and (d / "events.jsonl").exists()
        )

    def read_recent_events(self, days: Optional[int] = None) -> List[TrajectoryEvent]:
        """Read events from recent sessions."""
        if days is None:
            days = self.config.analysis_lookback_days

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_events: List[TrajectoryEvent] = []

        for session_id in self.list_sessions():
            events = self.read_session_events(session_id)
            if events and events[0].timestamp >= cutoff:
                all_events.extend(events)

        return all_events

    def _read_events_file(self, path: Path) -> List[TrajectoryEvent]:
        events: List[TrajectoryEvent] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    events.append(TrajectoryEvent.from_dict(d))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return events


class TrajectoryReader:
    """Reads segmented trajectories."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def read_session_trajectories(self, session_id: str) -> List[Trajectory]:
        """Read all trajectories for a session."""
        traj_file = self.config.trajectories_dir / session_id / "trajectories.jsonl"
        if not traj_file.exists():
            return []
        return self._read_trajectories_file(traj_file)

    def read_recent_trajectories(self, days: Optional[int] = None) -> List[Trajectory]:
        """Read trajectories from recent sessions."""
        if days is None:
            days = self.config.analysis_lookback_days

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_trajs: List[Trajectory] = []

        for session_id in self.list_trajectory_sessions():
            trajs = self.read_session_trajectories(session_id)
            if trajs and trajs[0].start_time and trajs[0].start_time >= cutoff:
                all_trajs.extend(trajs)

        return all_trajs

    def list_trajectory_sessions(self) -> List[str]:
        """List sessions that have segmented trajectories."""
        traj_dir = self.config.trajectories_dir
        if not traj_dir.exists():
            return []
        return sorted(
            d.name for d in traj_dir.iterdir()
            if d.is_dir() and (d / "trajectories.jsonl").exists()
        )

    def read_skill_trajectories(self, skill_name: str, days: Optional[int] = None) -> List[Trajectory]:
        """Read trajectories for a specific skill."""
        all_trajs = self.read_recent_trajectories(days)
        return [t for t in all_trajs if t.skill_name == skill_name]

    def _read_trajectories_file(self, path: Path) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    trajectories.append(Trajectory.from_dict(d))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return trajectories
