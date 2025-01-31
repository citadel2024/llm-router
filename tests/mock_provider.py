from typing import Any

from src.model import ChatMessageValues
from src.router.base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    async def completion(self, messages: list[ChatMessageValues], **kwargs) -> Any:
        pass

    async def embedding(self, texts: list[str], **kwargs) -> Any:
        pass
