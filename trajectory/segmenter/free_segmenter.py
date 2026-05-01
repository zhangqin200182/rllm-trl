"""Free Segmenter — groups non-skill events by turn boundary and file affinity."""

from __future__ import annotations

from typing import Dict, List, Set

from trajectory.adapter.schema import (
    EventType, IntentTag, TrajectoryEvent, Trajectory, TrajectoryType, ToolCall,
)
from trajectory.segmenter.base import SegmenterStrategy


class FreeSegmenter(SegmenterStrategy):
    """Segments remaining (non-skill) events into free trajectories.

    Strategy:
    1. Split by turn boundaries (turn_end events)
    2. Within each turn, group by file affinity
    3. Tag intent based on tool patterns
    """

    def name(self) -> str:
        return "free"

    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]:
        turns = self._split_by_turns(events)
        trajectories: List[Trajectory] = []

        for turn_events in turns:
            tool_events = [e for e in turn_events if e.event_type == EventType.TOOL_CALL]
            if not tool_events:
                continue

            groups = self._group_by_file_affinity(tool_events)
            for group in groups:
                traj = self._build_trajectory(group)
                trajectories.append(traj)

        return trajectories

    def _split_by_turns(self, events: List[TrajectoryEvent]) -> List[List[TrajectoryEvent]]:
        """Split events at turn_end boundaries."""
        turns: List[List[TrajectoryEvent]] = []
        current: List[TrajectoryEvent] = []

        for event in events:
            if event.event_type in (EventType.TURN_END, EventType.SESSION_END):
                if current:
                    turns.append(current)
                    current = []
            else:
                current.append(event)

        if current:
            turns.append(current)

        return turns

    def _group_by_file_affinity(self, events: List[TrajectoryEvent]) -> List[List[TrajectoryEvent]]:
        """Group events that touch overlapping files.

        Events that share file paths are grouped together. Events without files
        attach to the nearest group or form their own single-event group.
        """
        if not events:
            return []

        groups: List[List[TrajectoryEvent]] = []
        current_group: List[TrajectoryEvent] = [events[0]]
        current_files: Set[str] = set(events[0].files_touched)

        for event in events[1:]:
            event_files = set(event.files_touched)

            if event_files and current_files and event_files & current_files:
                current_group.append(event)
                current_files |= event_files
            elif not event_files and current_group:
                current_group.append(event)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [event]
                current_files = event_files

        if current_group:
            groups.append(current_group)

        return groups

    def _build_trajectory(self, events: List[TrajectoryEvent]) -> Trajectory:
        tool_calls = [
            ToolCall(
                tool_name=e.tool_name or "",
                tool_input=e.tool_input or {},
                tool_response=e.tool_response,
                timestamp=e.timestamp,
                success=e.success if e.success is not None else True,
                files_touched=e.files_touched,
            )
            for e in events
        ]

        files: List[str] = []
        for e in events:
            for f in e.files_touched:
                if f not in files:
                    files.append(f)

        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration_ms = (end_time - start_time).total_seconds() * 1000

        intent_tags = self._infer_intent(events)

        return Trajectory(
            trajectory_id=Trajectory.new_id(),
            session_id=events[0].session_id,
            conversation_id=events[0].conversation_id,
            trajectory_type=TrajectoryType.FREE,
            tool_calls=tool_calls,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            files_touched=files,
            intent_tags=intent_tags,
        )

    def _infer_intent(self, events: List[TrajectoryEvent]) -> List[str]:
        """Infer intent tags from tool usage patterns."""
        tool_names = [e.tool_name for e in events if e.tool_name]
        tags: List[str] = []

        read_heavy = tool_names.count("Read") + tool_names.count("Bash") > len(tool_names) * 0.7
        has_edit = any(t in ("Edit", "Write") for t in tool_names)
        has_test_bash = any(
            e.tool_name == "Bash" and self._is_test_command(e.tool_input or {})
            for e in events
        )

        if has_edit:
            tags.append(IntentTag.IMPLEMENTATION.value)

        if has_test_bash:
            tags.append(IntentTag.TESTING.value)

        if has_edit and has_test_bash:
            tags.append(IntentTag.DEBUGGING.value)

        if read_heavy and not has_edit:
            tags.append(IntentTag.EXPLORATION.value)

        if not tags:
            tags.append(IntentTag.EXPLORATION.value)

        return tags

    @staticmethod
    def _is_test_command(tool_input: Dict) -> bool:
        command = tool_input.get("command", "")
        test_indicators = ["pytest", "test", "unittest", "npm test", "yarn test", "jest", "mocha"]
        return any(ind in command.lower() for ind in test_indicators)
