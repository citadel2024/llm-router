from typing import Any

from src.providers.base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    async def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> Any:
        pass

    async def embedding(self, texts: list[str], **kwargs) -> Any:
        pass
