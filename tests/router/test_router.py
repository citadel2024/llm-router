import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import RetryConfig, LogConfiguration, LoadBalancerConfig
from src.model.input import RouterParams
from src.config.config import RouterConfig, LoadBalancerStrategy
from src.router.router import Router
from src.exceptions.exceptions import (
    RateLimitError,
    InvalidInputError,
    NoProviderAvailableError,
    ContentPolicyViolationError,
)


@pytest.fixture
def mock_router_config():
    return RouterConfig(
        llm_provider_group={"group1": []},
        log_config=LogConfiguration(),
        load_balancer_config=LoadBalancerConfig(strategy=LoadBalancerStrategy.RANDOM),
        retry_config=RetryConfig(max_attempt=3),
        fallback_config=MagicMock(allow_fallback=True, degraded_map={"group1": ["group2"]}),
        cooldown_config=MagicMock(),
        timeout_seconds=30,
    )


@pytest.fixture
def router(mock_router_config):
    return Router(mock_router_config)


@pytest.mark.asyncio
async def test_async_completion_success(router):
    mock_provider = MagicMock()
    mock_provider.impl.completion = AsyncMock(return_value="success")

    router.provider_status_manager.get_available_providers = AsyncMock(return_value=[mock_provider])
    router.load_balancer.schedule_provider = AsyncMock(return_value=mock_provider)

    result = await router.async_completion(RouterParams(model_group="group1", text="t"))
    assert result == "success"
    router.load_balancer.schedule_provider.assert_awaited_once()
    mock_provider.impl.completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_completion_no_provider_available(router):
    router.provider_status_manager.get_available_providers = AsyncMock(return_value=[])

    with pytest.raises(NoProviderAvailableError):
        await router.async_completion(RouterParams(model_group="group1", text="t"))


@pytest.mark.asyncio
async def test_async_completion_fallback_triggered(caplog, router):
    mock_main_provider = MagicMock()
    mock_main_provider.impl.completion = AsyncMock(side_effect=InvalidInputError(message="invalid input"))
    mock_fallback_provider = MagicMock()
    mock_fallback_provider.impl.completion = AsyncMock(return_value="fallback_success")

    router.provider_status_manager.get_available_providers = AsyncMock(
        side_effect=[[mock_main_provider], [mock_fallback_provider]]
    )
    router.load_balancer.schedule_provider = AsyncMock(side_effect=[mock_main_provider, mock_fallback_provider])

    result = await router.async_completion(RouterParams(model_group="group1", text="t"))
    assert result == "fallback_success"
    assert router.load_balancer.schedule_provider.await_count == 2
    assert caplog.text.count("Trying fallback model") == 1
    assert caplog.text.count("No fallback model specified") == 0


@pytest.mark.asyncio
async def test_async_completion_fallback_failed(caplog, router):
    caplog.set_level(logging.INFO)
    mock_main_provider = MagicMock()
    mock_main_provider.impl.completion = AsyncMock(side_effect=InvalidInputError(message="invalid input"))
    mock_fallback_provider = MagicMock()
    mock_fallback_provider.impl.completion = AsyncMock(side_effect=InvalidInputError(message="invalid input"))

    router.provider_status_manager.get_available_providers = AsyncMock(
        side_effect=[[mock_main_provider], [mock_fallback_provider]]
    )
    router.load_balancer.schedule_provider = AsyncMock(side_effect=[mock_main_provider, mock_fallback_provider])

    with pytest.raises(InvalidInputError):
        await router.async_completion(RouterParams(model_group="group1", text="t"))
    assert caplog.text.count("Trying fallback model") == 1
    assert caplog.text.count("No fallback model specified") == 1


@pytest.mark.asyncio
async def test_async_completion_retry_mechanism(router):
    mock_provider = MagicMock()
    mock_provider.impl.completion = AsyncMock(side_effect=[RateLimitError(message="rate limit"), "success"])

    router.provider_status_manager.get_available_providers = AsyncMock(return_value=[mock_provider])
    router.load_balancer.schedule_provider = AsyncMock(return_value=mock_provider)

    result = await router.async_completion(RouterParams(model_group="group1", text="t"))
    assert result == "success"
    assert mock_provider.impl.completion.await_count == 2


@pytest.mark.asyncio
async def test_rpm_tpm_usage_updated_on_success(router):
    mock_provider = MagicMock(id="test_provider")
    mock_provider.impl.completion = AsyncMock(return_value="success")

    router.provider_status_manager.get_available_providers = AsyncMock(return_value=[mock_provider])
    router.load_balancer.schedule_provider = AsyncMock(return_value=mock_provider)
    router.rpm_tpm_manager.update_rpm_used_usage = AsyncMock()
    router.rpm_tpm_manager.update_tpm_used_usage = AsyncMock()

    await router.async_completion(RouterParams(model_group="group1", text="test"))
    router.rpm_tpm_manager.update_rpm_used_usage.assert_awaited_once()
    router.rpm_tpm_manager.update_tpm_used_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_rpm_tpm_released_on_failure(router):
    mock_provider = MagicMock(id="test_provider")
    mock_provider.impl.completion = AsyncMock(
        side_effect=ContentPolicyViolationError(message="content policy violation")
    )

    router.provider_status_manager.get_available_providers = AsyncMock(return_value=[mock_provider])
    router.load_balancer.schedule_provider = AsyncMock(return_value=mock_provider)
    router.rpm_tpm_manager.release_rpm_occupied = AsyncMock()
    router.rpm_tpm_manager.release_tpm_occupied = AsyncMock()

    with pytest.raises(ContentPolicyViolationError):
        await router.async_completion(RouterParams(model_group="group99", text="test"))
    router.rpm_tpm_manager.release_rpm_occupied.assert_awaited_once()
    router.rpm_tpm_manager.release_tpm_occupied.assert_awaited_once()
