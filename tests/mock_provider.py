from typing import Any

from src.message import ChatMessageValues
from src.providers.base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    async def completion(self, messages: list[ChatMessageValues], **kwargs) -> Any:
        pass

    async def embedding(self, texts: list[str], **kwargs) -> Any:
        pass
