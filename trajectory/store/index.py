"""Index management — maintains global index.jsonl for quick lookups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from trajectory.adapter.schema import SessionSummary, Trajectory, TrajectoryType
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class IndexManager:
    """Manages the global session index (output/index.jsonl)."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def update_session(self, session_id: str, trajectories: List[Trajectory]) -> None:
        """Update or create an index entry for a session based on its trajectories."""
        summary = self._build_summary(session_id, trajectories)
        entries = self._read_all()
        entries[session_id] = summary
        self._write_all(entries)

    def get_session(self, session_id: str) -> Optional[SessionSummary]:
        """Get index entry for a specific session."""
        entries = self._read_all()
        return entries.get(session_id)

    def list_sessions(self) -> List[SessionSummary]:
        """List all indexed sessions, ordered by start_time."""
        entries = self._read_all()
        summaries = list(entries.values())
        summaries.sort(key=lambda s: s.start_time or "", reverse=True)
        return summaries

    def _build_summary(self, session_id: str, trajectories: List[Trajectory]) -> SessionSummary:
        skills_used: List[str] = []
        files_touched: List[str] = []
        skill_count = 0
        free_count = 0
        start_time = None
        end_time = None
        layer = "rllm"

        for traj in trajectories:
            if traj.trajectory_type == TrajectoryType.SKILL:
                skill_count += 1
                if traj.skill_name and traj.skill_name not in skills_used:
                    skills_used.append(traj.skill_name)
            else:
                free_count += 1

            for f in traj.files_touched:
                if f not in files_touched:
                    files_touched.append(f)

            if traj.start_time:
                if start_time is None or traj.start_time < start_time:
                    start_time = traj.start_time
            if traj.end_time:
                if end_time is None or traj.end_time > end_time:
                    end_time = traj.end_time

            layer = traj.layer

        return SessionSummary(
            session_id=session_id,
            layer=layer,
            start_time=start_time,
            end_time=end_time,
            trajectory_count=len(trajectories),
            skill_trajectories=skill_count,
            free_trajectories=free_count,
            skills_used=skills_used,
            files_touched=files_touched,
        )

    def _read_all(self) -> Dict[str, SessionSummary]:
        index_path = self.config.index_path
        if not index_path.exists():
            return {}

        entries: Dict[str, SessionSummary] = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    summary = SessionSummary.from_dict(d)
                    entries[summary.session_id] = summary
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return entries

    def _write_all(self, entries: Dict[str, SessionSummary]) -> None:
        index_path = self.config.index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w", encoding="utf-8") as f:
            for summary in entries.values():
                line = json.dumps(summary.to_dict(), ensure_ascii=False)
                f.write(line + "\n")
