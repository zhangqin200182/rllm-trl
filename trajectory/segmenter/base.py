"""Segmenter strategy interface — pluggable trajectory splitting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from trajectory.adapter.schema import TrajectoryEvent, Trajectory


class SegmenterStrategy(ABC):
    """Base class for trajectory segmentation strategies."""

    @abstractmethod
    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]:
        """Split a stream of events into meaningful trajectories."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Strategy identifier used for registry lookup."""
        ...
