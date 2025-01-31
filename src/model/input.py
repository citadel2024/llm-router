from typing import Optional
from dataclasses import dataclass

from src.config import RetryConfig, FallbackConfig
from src.model.message import ChatMessageValues


@dataclass
class RouterInput:
    # To distinguish with request, we use `RouterInput`.
    model_group: str
    text: Optional[str] = None
    messages: Optional[list[ChatMessageValues]] = None
    max_attempt: Optional[int] = None
    timeout_seconds: int = 30

    retry_config: Optional[RetryConfig] = None
    fallback_config: Optional[FallbackConfig] = None
