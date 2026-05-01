#!/usr/bin/env python3
"""Stop / SubagentStop hook — records turn_end and conversation_end events.

Called by Claude Code on Stop and SubagentStop events.
Usage:
  python trajectory/hooks/on_stop.py              # Stop (turn end)
  python trajectory/hooks/on_stop.py --subagent   # SubagentStop (sub-conversation end)

Must complete within 1 second. Fails silently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.adapter.hooks_adapter import HooksAdapter, read_stdin
from trajectory.store.writer import EventWriter
from trajectory.config import DEFAULT_CONFIG


def main() -> None:
    try:
        is_subagent = "--subagent" in sys.argv

        stdin_json = read_stdin()
        if not stdin_json:
            return

        hook_type = "SubagentStop" if is_subagent else "Stop"

        adapter = HooksAdapter()
        event = adapter.adapt(hook_type, stdin_json)

        writer = EventWriter(DEFAULT_CONFIG)
        writer.write_event(event)
    except Exception:
        pass


if __name__ == "__main__":
    main()
