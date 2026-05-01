"""Analyzer base — infrastructure for LLM-based trajectory analysis.

The actual analysis logic lives in traj-analyze-* SKILL.md files.
This module provides utilities for reading trajectories and formatting reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trajectory.adapter.schema import Trajectory, TrajectoryType
from trajectory.store.reader import TrajectoryReader
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class AnalyzerBase:
    """Provides trajectory reading and summary utilities for analysis skills."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config
        self.reader = TrajectoryReader(config)

    def get_rllm_trajectories(self, days: Optional[int] = None) -> List[Trajectory]:
        """Get rllm-train related trajectories from recent sessions."""
        all_trajs = self.reader.read_recent_trajectories(days)
        return [
            t for t in all_trajs
            if t.trajectory_type == TrajectoryType.SKILL
            and t.skill_name
            and t.skill_name.startswith("rllm-")
        ]

    def get_skill_trajectories(self, skill_prefix: str, days: Optional[int] = None) -> List[Trajectory]:
        """Get trajectories for skills matching a prefix."""
        all_trajs = self.reader.read_recent_trajectories(days)
        return [
            t for t in all_trajs
            if t.trajectory_type == TrajectoryType.SKILL
            and t.skill_name
            and t.skill_name.startswith(skill_prefix)
        ]

    def summarize_trajectory(self, traj: Trajectory) -> Dict[str, Any]:
        """Create a concise summary of a trajectory for LLM consumption."""
        tool_summary: Dict[str, int] = {}
        for tc in traj.tool_calls:
            tool_summary[tc.tool_name] = tool_summary.get(tc.tool_name, 0) + 1

        failed_tools = [
            {"tool": tc.tool_name, "input": tc.tool_input}
            for tc in traj.tool_calls
            if not tc.success
        ]

        return {
            "trajectory_id": traj.trajectory_id,
            "type": traj.trajectory_type.value,
            "skill_name": traj.skill_name,
            "skill_args": traj.skill_args,
            "duration_ms": traj.duration_ms,
            "tool_count": len(traj.tool_calls),
            "tool_summary": tool_summary,
            "files_touched": traj.files_touched,
            "intent_tags": traj.intent_tags,
            "outcome": traj.outcome,
            "failed_tools": failed_tools,
        }

    def format_for_llm(self, trajectories: List[Trajectory]) -> str:
        """Format trajectories as JSON for LLM analysis prompt."""
        summaries = [self.summarize_trajectory(t) for t in trajectories]
        return json.dumps(summaries, indent=2, ensure_ascii=False)
