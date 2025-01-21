from typing import Any

from src.providers.base_provider import BaseLLMProvider
from src.router.router_config import LLMProviderConfig, RouterConfig


class MockLLMProvider(BaseLLMProvider):
    async def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> Any:
        pass

    async def embedding(self, texts: list[str], **kwargs) -> Any:
        pass


def test_create_route_config():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_name="gpt3", impl=gpt3_impl, rpm=100, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_name="llama", impl=llama_impl, rpm=100, tpm=100)
    rc = RouterConfig(
        llm_providers={"gpt3-level-model": [gpt3, llama]}
    )

    assert len(rc.llm_providers) == 1
    assert len(rc.llm_providers["gpt3-level-model"]) == 2
    assert rc.llm_providers["gpt3-level-model"][0].model_name == "gpt3"
    assert rc.llm_providers["gpt3-level-model"][0].impl == gpt3_impl
    assert rc.llm_providers["gpt3-level-model"][0].rpm == 100
    assert rc.llm_providers["gpt3-level-model"][0].tpm == 100

    assert rc.llm_providers["gpt3-level-model"][1].model_name == "llama"
    assert rc.llm_providers["gpt3-level-model"][1].impl == llama_impl
    assert rc.llm_providers["gpt3-level-model"][1].rpm == 100
    assert rc.llm_providers["gpt3-level-model"][1].tpm == 100
