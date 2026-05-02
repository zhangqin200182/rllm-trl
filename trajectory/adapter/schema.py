"""Trajectory data models.

Defines the core data structures for the trajectory capture and analysis pipeline.
Two layers: raw data (Session/Conversation/Turn/ToolCall) and analysis (Trajectory).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    TOOL_CALL = "tool_call"
    TURN_END = "turn_end"
    SESSION_END = "session_end"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"


class TrajectoryType(str, Enum):
    SKILL = "skill"
    FREE = "free"


class IntentTag(str, Enum):
    EXPLORATION = "exploration"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEBUGGING = "debugging"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Adapter output — the single format all downstream modules depend on
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEvent:
    """Standardised event produced by HooksAdapter."""

    event_type: EventType
    session_id: str
    conversation_id: str
    timestamp: datetime
    layer: str = "rllm"

    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_response: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    files_touched: List[str] = field(default_factory=list)

    raw_hook_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "layer": self.layer,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_response": self.tool_response,
            "success": self.success,
            "files_touched": self.files_touched,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryEvent":
        return cls(
            event_type=EventType(d["type"]),
            session_id=d["session_id"],
            conversation_id=d["conversation_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            layer=d.get("layer", "rllm"),
            tool_name=d.get("tool_name"),
            tool_input=d.get("tool_input"),
            tool_response=d.get("tool_response"),
            success=d.get("success"),
            files_touched=d.get("files_touched", []),
        )


# ---------------------------------------------------------------------------
# Raw data layer: Session > Conversation > Turn > ToolCall
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    tool_input: Dict[str, Any]
    tool_response: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    files_touched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_response": self.tool_response,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "files_touched": self.files_touched,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(
            tool_name=d["tool_name"],
            tool_input=d.get("tool_input", {}),
            tool_response=d.get("tool_response"),
            timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None,
            duration_ms=d.get("duration_ms"),
            success=d.get("success", True),
            files_touched=d.get("files_touched", []),
        )


@dataclass
class Turn:
    turn_index: int
    tool_calls: List[ToolCall] = field(default_factory=list)
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    timestamp: Optional[datetime] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Conversation:
    conversation_id: str
    parent_conversation_id: Optional[str] = None
    turns: List[Turn] = field(default_factory=list)
    is_subagent: bool = False


@dataclass
class Session:
    session_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    transcript_path: Optional[str] = None
    conversations: List[Conversation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis data layer: Trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    trajectory_id: str
    session_id: str
    conversation_id: str
    trajectory_type: TrajectoryType
    layer: str = "rllm"

    skill_name: Optional[str] = None
    skill_args: Optional[str] = None

    tool_calls: List[ToolCall] = field(default_factory=list)
    nested_conversations: List[Conversation] = field(default_factory=list)

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    files_touched: List[str] = field(default_factory=list)

    intent_tags: List[str] = field(default_factory=list)
    outcome: str = Outcome.SUCCESS.value

    @staticmethod
    def new_id() -> str:
        return f"traj_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "type": self.trajectory_type.value,
            "layer": self.layer,
            "skill_name": self.skill_name,
            "skill_args": self.skill_args,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "files_touched": self.files_touched,
            "intent_tags": self.intent_tags,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(
            trajectory_id=d["trajectory_id"],
            session_id=d["session_id"],
            conversation_id=d["conversation_id"],
            trajectory_type=TrajectoryType(d["type"]),
            layer=d.get("layer", "rllm"),
            skill_name=d.get("skill_name"),
            skill_args=d.get("skill_args"),
            tool_calls=[ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])],
            start_time=datetime.fromisoformat(d["start_time"]) if d.get("start_time") else None,
            end_time=datetime.fromisoformat(d["end_time"]) if d.get("end_time") else None,
            duration_ms=d.get("duration_ms", 0.0),
            files_touched=d.get("files_touched", []),
            intent_tags=d.get("intent_tags", []),
            outcome=d.get("outcome", Outcome.SUCCESS.value),
        )


# ---------------------------------------------------------------------------
# Index / summary
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    session_id: str
    layer: str = "rllm"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    trajectory_count: int = 0
    skill_trajectories: int = 0
    free_trajectories: int = 0
    skills_used: List[str] = field(default_factory=list)
    files_touched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "layer": self.layer,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "trajectory_count": self.trajectory_count,
            "skill_trajectories": self.skill_trajectories,
            "free_trajectories": self.free_trajectories,
            "skills_used": self.skills_used,
            "files_touched": self.files_touched,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionSummary":
        return cls(
            session_id=d["session_id"],
            layer=d.get("layer", "rllm"),
            start_time=datetime.fromisoformat(d["start_time"]) if d.get("start_time") else None,
            end_time=datetime.fromisoformat(d["end_time"]) if d.get("end_time") else None,
            trajectory_count=d.get("trajectory_count", 0),
            skill_trajectories=d.get("skill_trajectories", 0),
            free_trajectories=d.get("free_trajectories", 0),
            skills_used=d.get("skills_used", []),
            files_touched=d.get("files_touched", []),
        )


# ---------------------------------------------------------------------------
# Optimization suggestion
# ---------------------------------------------------------------------------

@dataclass
class SkillOptimizationSuggestion:
    skill_name: str
    target_section: str
    action: str
    description: str
    rationale: str
    priority: str
    patch_content: str
    source_sessions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "target_section": self.target_section,
            "action": self.action,
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
            "patch_content": self.patch_content,
            "source_sessions": self.source_sessions,
        }
