"""Patch generator — creates skill-bank patch files from optimization suggestions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from trajectory.adapter.schema import SkillOptimizationSuggestion
from trajectory.config import DEFAULT_CONFIG, TrajectoryConfig


class PatchGenerator:
    """Generates skill-bank patch files from SkillOptimizationSuggestions."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def generate_patch(self, suggestion: SkillOptimizationSuggestion) -> Path:
        """Generate a single patch file in the skill-bank directory.

        Returns the path to the created patch file.
        """
        skill_dir = Path("skill-bank") / self._find_group(suggestion.skill_name) / suggestion.skill_name
        patches_dir = skill_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        patch_id = f"traj-{timestamp}-{suggestion.target_section}"
        filename = f"{patch_id}.md"
        patch_path = patches_dir / filename

        content = self._format_patch(patch_id, suggestion)
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(content)

        return patch_path

    def generate_patches(self, suggestions: List[SkillOptimizationSuggestion]) -> List[Path]:
        """Generate patch files for all suggestions."""
        return [self.generate_patch(s) for s in suggestions]

    def _format_patch(self, patch_id: str, suggestion: SkillOptimizationSuggestion) -> str:
        """Format a patch in skill-bank patch format."""
        desc = self._yaml_quote(suggestion.description)
        lines = [
            "---",
            f"id: {patch_id}",
            f"target_section: {suggestion.target_section}",
            f"action: {suggestion.action}",
            f"description: {desc}",
            f"status: proposed",
            f"source: trajectory-analysis",
            f"source_sessions: {json.dumps(suggestion.source_sessions)}",
            "---",
            "",
            suggestion.patch_content,
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _yaml_quote(value: str) -> str:
        if any(c in value for c in ":#{}[]|>&*!%@`"):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _find_group(self, skill_name: str) -> str:
        """Determine the skill-bank group for a skill name."""
        if skill_name.startswith("rllm-"):
            return "rllm"
        if skill_name.startswith("traj-"):
            return "traj"
        return "general"
