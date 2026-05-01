"""HooksAdapter — converts Claude Code Hooks stdin JSON to TrajectoryEvent.

This is the single coupling point with the Hooks JSON schema. When Hooks schema
changes, only this file needs updating; downstream modules depend solely on
TrajectoryEvent.
"""

from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trajectory.adapter.schema import EventType, TrajectoryEvent


class HooksAdapter:
    """Claude Code Hooks JSON → TrajectoryEvent."""

    def adapt(self, hook_type: str, stdin_json: Dict[str, Any]) -> TrajectoryEvent:
        """Convert raw hook input to a standardised TrajectoryEvent.

        Args:
            hook_type: One of "PostToolUse", "Stop", "SubagentStop".
            stdin_json: Parsed JSON from hook stdin.
        """
        session_id = stdin_json.get("session_id", "unknown")
        conversation_id = self._infer_conversation_id(hook_type, stdin_json)
        timestamp = datetime.now(timezone.utc)

        if hook_type == "PostToolUse":
            tool_name = stdin_json.get("tool_name", "")
            tool_input = stdin_json.get("tool_input", {})
            tool_response = stdin_json.get("tool_response")

            return TrajectoryEvent(
                event_type=EventType.TOOL_CALL,
                session_id=session_id,
                conversation_id=conversation_id,
                timestamp=timestamp,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_response=tool_response,
                success=self.infer_success(tool_name, tool_response),
                files_touched=self.extract_files(tool_name, tool_input),
                raw_hook_data=stdin_json,
            )

        elif hook_type == "Stop":
            return TrajectoryEvent(
                event_type=EventType.TURN_END,
                session_id=session_id,
                conversation_id=conversation_id,
                timestamp=timestamp,
                raw_hook_data=stdin_json,
            )

        elif hook_type == "SubagentStop":
            return TrajectoryEvent(
                event_type=EventType.CONVERSATION_END,
                session_id=session_id,
                conversation_id=conversation_id,
                timestamp=timestamp,
                raw_hook_data=stdin_json,
            )

        else:
            return TrajectoryEvent(
                event_type=EventType.TURN_END,
                session_id=session_id,
                conversation_id=conversation_id,
                timestamp=timestamp,
                raw_hook_data=stdin_json,
            )

    def extract_files(self, tool_name: str, tool_input: Dict[str, Any]) -> List[str]:
        """Extract file paths from tool_input based on tool type."""
        files: List[str] = []

        if tool_name in ("Read", "Write"):
            path = tool_input.get("file_path")
            if path:
                files.append(path)

        elif tool_name == "Edit":
            path = tool_input.get("file_path")
            if path:
                files.append(path)

        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            files.extend(self._extract_paths_from_command(command))

        elif tool_name == "NotebookEdit":
            path = tool_input.get("notebook_path")
            if path:
                files.append(path)

        return files

    def infer_success(self, tool_name: str, tool_response: Optional[Dict[str, Any]]) -> bool:
        """Infer whether a tool call succeeded based on response content."""
        if tool_response is None:
            return True

        response_str = json.dumps(tool_response).lower() if isinstance(tool_response, dict) else str(tool_response).lower()

        failure_indicators = [
            "error:", "traceback", "failed", "permission denied",
            "no such file", "command not found", "exitcode: 1",
        ]

        for indicator in failure_indicators:
            if indicator in response_str:
                return False

        return True

    def _infer_conversation_id(self, hook_type: str, stdin_json: Dict[str, Any]) -> str:
        """Infer conversation ID from hook data."""
        session_id = stdin_json.get("session_id", "unknown")

        if hook_type == "SubagentStop":
            subagent_id = stdin_json.get("conversation_id")
            if subagent_id:
                return subagent_id
            return f"{session_id}:sub"

        return stdin_json.get("conversation_id", session_id)

    def _extract_paths_from_command(self, command: str) -> List[str]:
        """Best-effort extraction of file paths from bash commands."""
        import shlex

        files: List[str] = []
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        for token in tokens:
            if "/" in token and not token.startswith("-"):
                if any(token.endswith(ext) for ext in (
                    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml",
                    ".txt", ".sh", ".toml", ".cfg", ".ini",
                    ".ts", ".tsx", ".js", ".jsx",
                )):
                    files.append(token)

        return files


def read_stdin() -> Optional[Dict[str, Any]]:
    """Read and parse JSON from stdin (hook input)."""
    try:
        data = sys.stdin.read()
        if not data.strip():
            return None
        return json.loads(data)
    except (json.JSONDecodeError, IOError):
        return None
