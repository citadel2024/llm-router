from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.model.input import UserParams
from src.router.base_provider import BaseLLMProvider


class ConcreteLLMProvider(BaseLLMProvider):
    async def completion(self, param: UserParams) -> Any:
        return await self.external_call(param)

    async def external_call(self, param):
        pass


def test_repr():
    class TestProvider(BaseLLMProvider):
        __module__ = "fake.module"

        async def completion(self, param):
            pass

    instance = TestProvider()
    assert repr(instance) == "<fake.module.TestProvider>"


@pytest.mark.asyncio
@patch.object(ConcreteLLMProvider, "external_call", new_callable=AsyncMock)
async def test_completion(mock_external_call):
    provider = ConcreteLLMProvider()
    mock_param = UserParams(model_group="m", text="t")
    mock_external_call.return_value = "mocked"

    result = await provider.completion(mock_param)

    assert result == "mocked"
    mock_external_call.assert_awaited_once_with(mock_param)
