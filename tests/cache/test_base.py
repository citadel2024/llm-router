from unittest.mock import AsyncMock, create_autospec

import pytest

from src.cache.base import BaseCache


@pytest.fixture
def mock_cache():
    mock = create_autospec(
        BaseCache,
        instance=True,
        async_set_value=AsyncMock(),
        async_get_value=AsyncMock(),
    )
    mock.default_ttl = 3600
    return mock


@pytest.mark.asyncio
async def test_async_set_value_default_ttl(mock_cache):
    await mock_cache.async_set_value("key", "value")
    mock_cache.async_set_value.assert_awaited_once_with("key", "value")


@pytest.mark.asyncio
async def test_async_set_value_custom_ttl(mock_cache):
    await mock_cache.async_set_value("key", "value", ttl=100)
    mock_cache.async_set_value.assert_awaited_once_with("key", "value", ttl=100)


@pytest.mark.asyncio
async def test_async_get_value(mock_cache):
    mock_cache.async_get_value.return_value = "cached_value"
    result = await mock_cache.async_get_value("key")
    assert result == "cached_value"
    mock_cache.async_get_value.assert_awaited_once_with("key")
