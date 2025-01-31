from enum import Enum
from typing import Union, Optional
from functools import lru_cache
from dataclasses import dataclass

import tiktoken
from tokenizers import Tokenizer

from src.config import LogConfiguration
from src.router.log import get_logger
from src.token.func import _format_function_definitions
from src.model.message import ChatMessageValues

DEFAULT_IMAGE_TOKEN_COUNT = 250

CL100K_BASE = tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(encoding, messages, tokens_per_message, tokens_per_name):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value, disallowed_special=()))
                if key == "name":
                    num_tokens += tokens_per_name
            elif isinstance(value, list):
                num_tokens += _count_nested_list(encoding, value)
    return num_tokens


def _count_nested_list(encoding, content_list):
    num_tokens = 0
    for content in content_list:
        if content["type"] == "text":
            num_tokens += len(encoding.encode(content["text"], disallowed_special=()))
        elif content["type"] == "image_url":
            num_tokens += DEFAULT_IMAGE_TOKEN_COUNT
    return num_tokens


def _process_messages(messages):
    text = ""
    is_tool_call = False
    for message in messages:
        content = message.get("content")
        if content:
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text += item["text"]
        if message.get("tool_calls"):
            is_tool_call = True
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    text += tool_call["function"]["arguments"]
    return text, is_tool_call


class TokenizerType(Enum):
    HuggingFace = "huggingface"
    OpenAI = "openai"


@dataclass
class TokenCounterFunc:
    tokenizer: Tokenizer
    tokenizer_type: TokenizerType


class TokenCounter:
    def __init__(self, log_cfg: LogConfiguration):
        self.logger = get_logger(__name__, log_cfg=log_cfg)

    def _get_encoding(self, model):
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            self.logger.error(f"Could not automatically map {model} to a tokeniser. ")
            return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    @lru_cache(maxsize=128)
    def _select_tokenizer_helper(model: str):
        if "llama-3" in model.lower():
            tokenizer = Tokenizer.from_pretrained("Xenova/llama-3-tokenizer")
            return TokenCounterFunc(tokenizer=tokenizer, tokenizer_type=TokenizerType.HuggingFace)
        else:
            return TokenCounterFunc(tokenizer=CL100K_BASE, tokenizer_type=TokenizerType.OpenAI)

    def _openai_token_counter(
        self,
        model: str,
        messages=None,
        text=None,
        is_tool_call=False,
        tools=None,
        tool_choice=None,
        count_response_tokens=False,
    ):
        encoding = self._get_encoding(model)
        tokens_per_message, tokens_per_name = (4, -1) if model == "gpt-3.5-turbo-0301" else (3, 1)
        num_tokens = 0
        includes_system_message = False
        if is_tool_call and text:
            num_tokens += len(encoding.encode(text, disallowed_special=()))
        elif messages:
            includes_system_message = any(message.get("role") == "system" for message in messages)
            num_tokens += _count_message_tokens(encoding, messages, tokens_per_message, tokens_per_name)
        elif text:
            num_tokens += len(encoding.encode(text, disallowed_special=()))
            if count_response_tokens:
                return num_tokens
        num_tokens += 3
        if tools:
            num_tokens += len(encoding.encode(_format_function_definitions(tools))) + 9
            if includes_system_message:
                num_tokens -= 4
        if tool_choice:
            if tool_choice == "none":
                num_tokens += 1
            elif isinstance(tool_choice, dict) and "function" in tool_choice and "name" in tool_choice["function"]:
                num_tokens += 7 + len(encoding.encode(tool_choice["function"]["name"]))
        return num_tokens

    def token_counter(
        self,
        model: str = "",
        text: Optional[str] = None,
        messages: Optional[list[ChatMessageValues]] = None,
        count_response_tokens: bool = False,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> int:
        if text is None and messages is None:
            raise ValueError("text and messages cannot both be None")
        is_tool_call = False
        if messages is not None:
            text, is_tool_call = _process_messages(messages)
        else:
            count_response_tokens = True
        if model:
            tokenizer = self._select_tokenizer_helper(model=model)
            if tokenizer.tokenizer_type == TokenizerType.HuggingFace:
                num_tokens = len(tokenizer.tokenizer.encode(text).ids)
            else:  # tokenizer.tokenizer_type == TokenizerType.OpenAI:
                num_tokens = self._openai_token_counter(
                    text=text,
                    model=model,
                    messages=messages,
                    is_tool_call=is_tool_call,
                    count_response_tokens=count_response_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                )
        else:
            num_tokens = len(CL100K_BASE.encode(text, disallowed_special=()))
        return num_tokens
