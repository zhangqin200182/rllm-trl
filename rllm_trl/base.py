"""
Self-contained base abstractions inlined from rllm.
Provides BaseAgent, BaseEnv, Action, Step, Trajectory, ToolCall, ToolOutput.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    thought: str = ""
    action: Any = None
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0


@dataclass
class Action:
    action: Any = None


@dataclass
class Trajectory:
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self):
        return {
            "steps": [asdict(step) for step in self.steps],
            "reward": float(self.reward),
        }


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return []

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory()

    @abstractmethod
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Action:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        return

    def get_current_state(self) -> Step | None:
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]


class BaseEnv(ABC):
    @property
    def idx(self) -> Any:
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        pass

    def close(self):
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        raise NotImplementedError

    @staticmethod
    def is_multithread_safe() -> bool:
        return True


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolOutput:
    name: str = ""
    output: str | list | dict | None = None
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        elif self.output is None:
            return ""
        elif isinstance(self.output, (list, dict)):
            return json.dumps(self.output)
        else:
            return str(self.output)
