"""Segmenter registry — orchestrates skill + free segmentation pipeline."""

from __future__ import annotations

from typing import Dict, List, Set

from trajectory.adapter.schema import EventType, TrajectoryEvent, Trajectory
from trajectory.segmenter.base import SegmenterStrategy
from trajectory.segmenter.skill_segmenter import SkillSegmenter
from trajectory.segmenter.free_segmenter import FreeSegmenter


class SegmenterRegistry:
    """Runs segmenters in order: skill first, then free on remaining events."""

    def __init__(self):
        self._strategies: Dict[str, SegmenterStrategy] = {}
        self.register(SkillSegmenter())
        self.register(FreeSegmenter())

    def register(self, strategy: SegmenterStrategy) -> None:
        self._strategies[strategy.name()] = strategy

    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]:
        """Run the full segmentation pipeline.

        1. Skill segmenter claims events anchored by Skill tool calls.
        2. Free segmenter processes unclaimed events.
        """
        skill_segmenter = self._strategies.get("skill")
        free_segmenter = self._strategies.get("free")

        if not skill_segmenter or not free_segmenter:
            return []

        skill_trajs = skill_segmenter.segment(events)

        claimed_timestamps: Set[str] = set()
        for traj in skill_trajs:
            for tc in traj.tool_calls:
                if tc.timestamp:
                    claimed_timestamps.add(tc.timestamp.isoformat())

        remaining = [
            e for e in events
            if e.event_type != EventType.TOOL_CALL
            or e.timestamp.isoformat() not in claimed_timestamps
        ]

        free_trajs = free_segmenter.segment(remaining)

        all_trajs = skill_trajs + free_trajs
        all_trajs.sort(key=lambda t: t.start_time or t.end_time)

        return all_trajs

    def get_strategy(self, name: str) -> SegmenterStrategy:
        return self._strategies[name]
