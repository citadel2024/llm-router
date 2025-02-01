from typing import Optional
from dataclasses import dataclass

from src.config import RetryConfig, FallbackConfig
from src.model.message import ChatMessageValues
from src.exceptions.exceptions import InvalidInputError


@dataclass
class UserParams:
    model_group: str
    text: Optional[str] = None
    messages: Optional[list[ChatMessageValues]] = None

    def __post_init__(self):
        text = self.text
        messages = self.messages
        if not (text or messages):
            raise InvalidInputError("Either 'text' or 'messages' must be provided.")


@dataclass
class RouterParams(UserParams):
    # To distinguish with request, we use `RouterInput`.
    timeout_seconds: int = 30
    retry_config: Optional[RetryConfig] = None
    fallback_config: Optional[FallbackConfig] = None
