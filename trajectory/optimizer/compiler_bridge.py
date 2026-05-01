"""Compiler bridge — invokes skill-bank/compile.py to apply patches."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class CompilerBridge:
    """Calls skill-bank/compile.py to compile skills after patch application."""

    COMPILE_SCRIPT = "skill-bank/compile.py"

    def compile_skill(self, skill_name: str) -> subprocess.CompletedProcess:
        """Compile a single skill."""
        return subprocess.run(
            ["python", self.COMPILE_SCRIPT, skill_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def compile_group(self, group: str) -> subprocess.CompletedProcess:
        """Compile all skills in a group."""
        return subprocess.run(
            ["python", self.COMPILE_SCRIPT, "--group", group],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def diff_skill(self, skill_name: str) -> str:
        """Preview changes for a skill without compiling."""
        result = subprocess.run(
            ["python", self.COMPILE_SCRIPT, "--diff", skill_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    def status(self) -> str:
        """Get patch status summary."""
        result = subprocess.run(
            ["python", self.COMPILE_SCRIPT, "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
