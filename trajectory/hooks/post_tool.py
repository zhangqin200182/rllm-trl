#!/usr/bin/env python3
"""PostToolUse hook — captures tool calls to events.jsonl.

Called by Claude Code after every tool use. Reads hook JSON from stdin,
converts via HooksAdapter, and appends to the session's events file.

Must complete within 1 second. Fails silently to avoid disrupting Claude Code.
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.adapter.hooks_adapter import HooksAdapter, read_stdin
from trajectory.store.writer import EventWriter
from trajectory.config import TrajectoryConfig


def detect_layer(stdin_json: Dict[str, Any]) -> str:
    """Detect layer from hook data based on skill name prefix."""
    tool_data = stdin_json.get("tool", {})
    tool_name = tool_data.get("name")

    if tool_name == "Skill":
        tool_input = tool_data.get("input", {})
        skill_name = tool_input.get("skill", "")

        if skill_name.startswith("rllm-"):
            return "rllm"
        elif skill_name.startswith("traj-"):
            return "traj"
        elif skill_name.startswith("meta-"):
            return "meta"

    return "rllm"


def main() -> None:
    try:
        stdin_json = read_stdin()
        if not stdin_json:
            return

        layer = detect_layer(stdin_json)
        config = TrajectoryConfig(layer=layer)

        adapter = HooksAdapter()
        event = adapter.adapt("PostToolUse", stdin_json)
        event.layer = layer

        writer = EventWriter(config)
        writer.write_event(event)
    except Exception:
        pass


if __name__ == "__main__":
    main()
