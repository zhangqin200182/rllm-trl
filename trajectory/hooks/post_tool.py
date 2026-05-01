#!/usr/bin/env python3
"""PostToolUse hook — captures tool calls to events.jsonl.

Called by Claude Code after every tool use. Reads hook JSON from stdin,
converts via HooksAdapter, and appends to the session's events file.

Must complete within 1 second. Fails silently to avoid disrupting Claude Code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.adapter.hooks_adapter import HooksAdapter, read_stdin
from trajectory.store.writer import EventWriter
from trajectory.config import DEFAULT_CONFIG


def main() -> None:
    try:
        stdin_json = read_stdin()
        if not stdin_json:
            return

        adapter = HooksAdapter()
        event = adapter.adapt("PostToolUse", stdin_json)

        writer = EventWriter(DEFAULT_CONFIG)
        writer.write_event(event)
    except Exception:
        pass


if __name__ == "__main__":
    main()
