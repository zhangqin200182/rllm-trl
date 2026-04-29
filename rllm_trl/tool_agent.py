"""
Self-contained ToolAgent inlined from rllm.
"""

import copy
import json
import logging
import uuid
from typing import Any

from rllm_trl.base import Action, BaseAgent, Step, Trajectory
from rllm_trl.parsers import QwenToolParser

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    def __init__(self, system_prompt="", parser_name="qwen", tool_map=None):
        self.system_prompt = system_prompt
        self.tool_parser = QwenToolParser()

        if tool_map is not None:
            self._tool_instances = {}
            for name, tool_cls in tool_map.items():
                self._tool_instances[name] = tool_cls()
            tools_json = [inst.json for inst in self._tool_instances.values()]
        else:
            self._tool_instances = {}
            tools_json = []

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(tools_json, indent=2))
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append({
                        "role": "tool",
                        "content": tool_output_str,
                        "tool_call_id": tool_call_id,
                    })
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})
        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        tool_calls_dict = []
        assistant_content = response
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {e}")
            tool_calls_dict = []

        assistant_message = {"role": "assistant", "content": assistant_content}
        if tool_calls_dict:
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {"response": assistant_content},
                    },
                }
            ]

        self.messages.append(assistant_message)
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)
        return Action(action=tool_calls_dict)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt + self.tools_prompt}]

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
