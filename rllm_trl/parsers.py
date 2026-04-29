"""
Self-contained chat template and tool parsers inlined from rllm.
Provides QwenChatTemplateParser, QwenToolParser, and helper functions
for converting messages to tokens and masks.
"""

import json
from typing import Any

from rllm_trl.base import ToolCall


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False) -> "ChatTemplateParser":
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            if "qwen" in model_name or "qwen" in tokenizer_cls:
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
        return ChatTemplateParser(tokenizer)


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token
        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"
        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token
        for message in messages:
            role = message["role"]
            if role == "system":
                result += self.system_token + message["content"] + self.eot_token
            elif role == "user":
                result += self.user_token + message["content"] + self.eot_token
            elif role == "assistant":
                result += self.assistant_token + message["content"] + self.eot_token
            elif role == "tool":
                result += self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token
        if add_generation_prompt:
            result += self.generation_prompt
        return result


class QwenToolParser:
    def __init__(self):
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"

    def parse(self, model_response: str) -> list[ToolCall]:
        tool_calls_dicts = self._parse_tool_calls(model_response)
        return [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        if self.tool_call_begin not in text:
            return tool_calls
        while self.tool_call_begin in text:
            start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = text.find(self.tool_call_end)
            if end == -1:
                break
            json_content = text[start:end].strip()
            try:
                call_data = json.loads(json_content)
                tool_calls.append({"name": call_data["name"], "arguments": call_data["arguments"]})
            except json.JSONDecodeError:
                text = text[end + len(self.tool_call_end):]
                continue
            text = text[end + len(self.tool_call_end):]
        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""


def get_recent_assistant_user_messages(chat_completions_messages):
    env_messages = []
    assistant_message = None
    seen_assistant_message = False
    for message in reversed(chat_completions_messages):
        role = message.get("role", None)
        if role == "assistant":
            if assistant_message:
                break
            seen_assistant_message = True
            assistant_message = message
        elif role in ["user", "tool"] and not seen_assistant_message:
            env_messages.append(message)
    env_messages = list(reversed(env_messages))
    return assistant_message, env_messages


def convert_messages_to_tokens_and_masks(messages, tokenizer, parser, contains_first_msg=False, contains_generation_msg=False):
    all_msg_tokens = []
    all_msg_masks = []

    def _convert(msg, first_msg=False, generation_msg=False):
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)
        if msg["role"] == "assistant" and msg_text.startswith(parser.assistant_token):
            msg_text = msg_text.replace(parser.assistant_token, "", 1)
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        return msg_tokens, [mask_value] * len(msg_tokens)

    for i, msg in enumerate(messages):
        tokens, mask = _convert(
            msg,
            first_msg=(contains_first_msg and i == 0),
            generation_msg=(contains_generation_msg and i == len(messages) - 1),
        )
        all_msg_tokens.extend(tokens)
        all_msg_masks.extend(mask)

    return all_msg_tokens, all_msg_masks
