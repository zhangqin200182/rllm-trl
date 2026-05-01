"""Report writer — formats and saves analysis reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from trajectory.adapter.schema import SkillOptimizationSuggestion
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class ReportWriter:
    """Writes analysis reports to output/reports/."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def write_report(self, content: str, prefix: str = "report") -> Path:
        """Write a markdown report and return its path."""
        reports_dir = self.config.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{prefix}.md"
        report_path = reports_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return report_path

    def format_suggestions_table(self, suggestions: List[SkillOptimizationSuggestion]) -> str:
        """Format optimization suggestions as a markdown table."""
        if not suggestions:
            return "No optimization suggestions.\n"

        lines = [
            "| Priority | Target Skill | Section | Action | Description |",
            "|----------|-------------|---------|--------|-------------|",
        ]

        for s in suggestions:
            lines.append(
                f"| {s.priority} | {s.skill_name} | {s.target_section} | {s.action} | {s.description} |"
            )

        return "\n".join(lines) + "\n"
