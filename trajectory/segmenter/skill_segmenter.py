"""Skill Segmenter — groups events anchored by Skill tool calls."""

from __future__ import annotations

from typing import List, Optional

from trajectory.adapter.schema import (
    EventType, TrajectoryEvent, Trajectory, TrajectoryType, ToolCall,
)
from trajectory.segmenter.base import SegmenterStrategy


class SkillSegmenter(SegmenterStrategy):
    """Identifies skill-anchored trajectories.

    Scans for tool_name == "Skill" events and collects all subsequent tool calls
    until the next Skill call, turn boundary, or session end.
    """

    def name(self) -> str:
        return "skill"

    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        current: Optional[_SkillGroup] = None

        for event in events:
            if event.event_type == EventType.TOOL_CALL and event.tool_name == "Skill":
                if current:
                    trajectories.append(current.to_trajectory())
                current = _SkillGroup(anchor=event)

            elif event.event_type == EventType.TOOL_CALL and current:
                current.add_event(event)

            elif event.event_type in (EventType.TURN_END, EventType.SESSION_END):
                if current:
                    trajectories.append(current.to_trajectory())
                    current = None

        if current:
            trajectories.append(current.to_trajectory())

        return trajectories


class _SkillGroup:
    """Accumulates events belonging to a single skill trajectory."""

    def __init__(self, anchor: TrajectoryEvent):
        self.anchor = anchor
        self.events: List[TrajectoryEvent] = [anchor]

    def add_event(self, event: TrajectoryEvent) -> None:
        self.events.append(event)

    def to_trajectory(self) -> Trajectory:
        skill_input = self.anchor.tool_input or {}
        skill_name = skill_input.get("skill", skill_input.get("name", "unknown"))
        skill_args = skill_input.get("args")

        tool_calls = [self._event_to_tool_call(e) for e in self.events]

        files: List[str] = []
        for e in self.events:
            for f in e.files_touched:
                if f not in files:
                    files.append(f)

        start_time = self.events[0].timestamp
        end_time = self.events[-1].timestamp
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return Trajectory(
            trajectory_id=Trajectory.new_id(),
            session_id=self.anchor.session_id,
            conversation_id=self.anchor.conversation_id,
            trajectory_type=TrajectoryType.SKILL,
            layer=self.anchor.layer,
            skill_name=skill_name,
            skill_args=skill_args,
            tool_calls=tool_calls,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            files_touched=files,
        )

    @staticmethod
    def _event_to_tool_call(event: TrajectoryEvent) -> ToolCall:
        return ToolCall(
            tool_name=event.tool_name or "",
            tool_input=event.tool_input or {},
            tool_response=event.tool_response,
            timestamp=event.timestamp,
            success=event.success if event.success is not None else True,
            files_touched=event.files_touched,
        )
