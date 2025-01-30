from typing import Union, Literal, Iterable, Optional, Required, TypedDict

from openai.types.chat import ChatCompletionContentPartInputAudioParam


class OpenAIChatCompletionTextObject(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ChatCompletionTextObject(OpenAIChatCompletionTextObject, total=False):
    cache_control: ChatCompletionCachedContent


class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str


class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionImageUrlObject]


class ChatCompletionAudioObject(ChatCompletionContentPartInputAudioParam):
    pass


OpenAIMessageContentListBlock = Union[
    ChatCompletionTextObject,
    ChatCompletionImageObject,
    ChatCompletionAudioObject,
]

OpenAIMessageContent = Union[
    str,
    Iterable[OpenAIMessageContentListBlock],
]


class OpenAIChatCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: OpenAIMessageContent


class ChatCompletionUserMessage(OpenAIChatCompletionUserMessage, total=False):
    cache_control: ChatCompletionCachedContent


class ChatCompletionToolCallFunctionChunk(TypedDict, total=False):
    name: Optional[str]
    arguments: str


class ChatCompletionAssistantToolCall(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class OpenAIChatCompletionAssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]]
    name: Optional[str]
    tool_calls: Optional[list[ChatCompletionAssistantToolCall]]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]


class ChatCompletionAssistantMessage(OpenAIChatCompletionAssistantMessage, total=False):
    cache_control: ChatCompletionCachedContent


class ChatCompletionToolMessage(TypedDict):
    role: Literal["tool"]
    content: Union[str, Iterable[ChatCompletionTextObject]]
    tool_call_id: str


class ChatCompletionFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]]
    name: str
    tool_call_id: Optional[str]


class OpenAIChatCompletionSystemMessage(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[Union[str, list]]
    name: str


class ChatCompletionSystemMessage(OpenAIChatCompletionSystemMessage, total=False):
    cache_control: ChatCompletionCachedContent


ChatMessageValues = Union[
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
    ChatCompletionFunctionMessage,
]
