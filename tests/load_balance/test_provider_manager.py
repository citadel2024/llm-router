import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import LogConfiguration
from src.cache.base import BaseCache
from src.config.config import LLMProviderConfig
from src.config.cooldown import CooldownConfig, AllowedFailsPolicy
from tests.mock_provider import MockLLMProvider
from src.exceptions.exceptions import (
    RateLimitError,
    BadRequestError,
    ModelGroupNotFound,
    RequestTimeoutError,
)
from src.load_balance.provider_manager import CooldownState, ProviderStatusManager


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=BaseCache)
    cache.async_get_value = AsyncMock(return_value=None)
    cache.async_set_value = AsyncMock()
    return cache


@pytest.fixture
def mock_providers():
    provider1 = LLMProviderConfig(model_id="provider1", impl=MockLLMProvider())
    provider2 = LLMProviderConfig(model_id="provider2", impl=MockLLMProvider())
    provider1.id = "provider1"
    provider2.id = "provider2"
    return [provider1, provider2]


@pytest.fixture
def mock_bad_request():
    return BadRequestError(
        "Bad request error",
        response=httpx.Response(status_code=400, request=httpx.Request("GET", "")),
    )


@pytest.fixture
def mock_rate_limit():
    return RateLimitError(
        "Rate limit error",
        response=httpx.Response(status_code=429, request=httpx.Request("GET", "")),
    )


@pytest.fixture
def mock_request_timeout():
    return RequestTimeoutError(
        "Request timeout",
        response=httpx.Response(status_code=404, request=httpx.Request("GET", "")),
    )


@pytest.fixture
def mock_manager(mock_cache, mock_providers):
    provider_groups = {"group1": mock_providers}
    cooldown_config = CooldownConfig(
        allowed_fails_policy=AllowedFailsPolicy(
            BadRequestErrorAllowedFails=2,
            TimeoutErrorAllowedFails=2,
            RateLimitErrorAllowedFails=1,
            ContentPolicyViolationErrorAllowedFails=0,
            AuthenticationErrorAllowedFails=1,
        ),
        general_allowed_fails=2,
        cooldown_seconds=300,
    )
    return ProviderStatusManager(
        log_cfg=LogConfiguration(), provider_groups=provider_groups, cooldown_config=cooldown_config, cache=mock_cache
    )


@pytest.mark.asyncio
async def test_get_available_providers(mock_manager):
    providers = await mock_manager.get_available_providers("group1")
    assert len(providers) == 2
    assert [p.id for p in providers] == ["provider1", "provider2"]


@pytest.mark.asyncio
async def test_get_available_providers_with_cooldown(mock_manager, mock_cache):
    cooldown_data = CooldownState(
        exception="RateLimitError", timestamp=time.time() - 100, cooldown_seconds=300
    ).serialize()
    mock_cache.async_get_value.side_effect = [cooldown_data, None]
    providers = await mock_manager.get_available_providers("group1")
    assert len(providers) == 1
    assert "provider1" not in [p.id for p in providers]
    assert "provider2" in [p.id for p in providers]


@pytest.mark.asyncio
async def test_critical_exception_cooldown(mock_manager, mock_cache, mock_rate_limit):
    exception = mock_rate_limit
    await mock_manager.try_add_cooldown("provider1", exception)
    mock_cache.async_set_value.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_critical_exception_cooldown(mock_manager, mock_cache, mock_bad_request):
    exception = mock_bad_request
    mock_cache.async_get_value.return_value = 2
    result = await mock_manager._should_cooldown("provider1", exception)
    assert result is True


@pytest.mark.asyncio
async def test_temp_exception_no_cooldown(mock_manager, mock_request_timeout):
    exception = mock_request_timeout
    result = await mock_manager._should_cooldown("provider1", exception)
    assert result is False


@pytest.mark.asyncio
async def test_failure_count_increment(mock_manager, mock_cache, mock_request_timeout):
    exception = mock_request_timeout
    mock_cache.async_get_value.side_effect = [None, 1, 2]
    assert not await mock_manager._should_cooldown("provider1", exception)
    assert not await mock_manager._should_cooldown("provider1", exception)
    assert await mock_manager._should_cooldown("provider1", exception)
    assert mock_cache.async_set_value.await_count == 2


@pytest.mark.asyncio
async def test_cooldown_state_expiration():
    state = CooldownState(exception="Test", timestamp=time.time() - 400, cooldown_seconds=300)
    assert state.is_expired()


def test_allowed_fails_policy(mock_manager, mock_request_timeout):
    exception = mock_request_timeout
    allowed = mock_manager._get_allowed_fails_from_policy(exception)
    assert allowed == 2


def test_default_allowed_fails(mock_manager):
    class TestError(Exception):
        status_code = 418

    exception = TestError()
    allowed = mock_manager._get_allowed_fails_from_policy(exception)
    assert allowed == 2


@pytest.mark.asyncio
async def test_model_group_not_found(mock_manager):
    with pytest.raises(ModelGroupNotFound):
        mock_manager._get_healthy_providers("invalid_group")


@pytest.mark.asyncio
async def test_lock_mechanism(mock_manager):
    key = "test_key"
    lock1 = mock_manager._fetch_or_create_lock(key)
    lock2 = mock_manager._fetch_or_create_lock(key)
    assert lock1 is lock2
    assert key in mock_manager._locks
