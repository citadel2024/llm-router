from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.utils.context import RouterContext, router_context
from tests.mock_provider import MockLLMProvider
from src.load_balance.capacity_based import CapacityBasedBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


@pytest.fixture
def mock_lb_cache():
    return MagicMock(spec=BaseCache)


@pytest.fixture
def mock_log_cfg():
    return LogConfiguration()


@pytest.fixture
def mock_lb_config():
    config = MagicMock(spec=LoadBalancerConfig)
    config.capacity_dimension = "rpm"
    return config


@pytest.fixture
def mock_balancer(mock_lb_cache, mock_log_cfg, mock_lb_config):
    instance = CapacityBasedBalancer(mock_lb_cache, mock_log_cfg, mock_lb_config)
    instance.logger = MagicMock()
    return instance


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=LLMProviderConfig)
    return provider


@pytest.mark.asyncio
async def test_schedule_provider_returns_none_when_no_healthy_providers(mock_balancer):
    result = await mock_balancer.schedule_provider("test_group", [])
    assert result is None


@pytest.mark.asyncio
async def test_schedule_provider_returns_none_when_all_over_limit(mock_balancer, mock_provider):
    with patch.object(mock_balancer, "_filter_over_limit_providers", return_value=[]):
        result = await mock_balancer.schedule_provider("test_group", [mock_provider])
        assert result is None
        mock_balancer.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_provider_selects_provider_correctly(mock_balancer, mock_provider):
    with patch.object(mock_balancer, "_filter_over_limit_providers", return_value=[mock_provider]), patch.object(
        mock_balancer, "_select_weighted_provider", return_value=mock_provider
    ):
        result = await mock_balancer.schedule_provider("test_group", [mock_provider])
        assert result == mock_provider


@pytest.mark.asyncio
async def test_capacity_based_balancer_exhaust_rpm(mock_lb_cache):
    provider1 = LLMProviderConfig(model_id="model1", impl=MagicMock(), rpm=5)
    provider1.id = "provider1"
    provider2 = LLMProviderConfig(model_id="model2", impl=MagicMock(), rpm=3)
    provider2.id = "provider2"
    healthy_providers = [provider1, provider2]
    balancer = CapacityBasedBalancer(mock_lb_cache, LogConfiguration(), LoadBalancerConfig(capacity_dimension="rpm"))
    selected_providers = []
    u0 = RpmTpmManager.Usage(used=0, occupying=0).serialize()
    u3 = RpmTpmManager.Usage(used=3, occupying=0).serialize()
    u5 = RpmTpmManager.Usage(used=5, occupying=0).serialize()
    # The first 5 calls, make provider1 exceed the RPM limit, and the next 3 calls, make provider2 exceed the RPM limit
    mock_lb_cache.async_get_value = AsyncMock(
        side_effect=[u0, u3, u0, u3, u0, u3, u0, u3, u0, u3, u5, u0, u5, u0, u5, u0]
    )
    for _ in range(8):
        router_context.set(RouterContext())
        selected_provider = await balancer.schedule_provider("test_group", healthy_providers)
        if selected_provider:
            selected_providers.append(selected_provider.id)

    assert len(selected_providers) == 8
    assert selected_providers.count("provider1") == 5
    assert selected_providers.count("provider2") == 3


@patch("random.choices")
def test_select_weighted_provider_with_valid_weights(mock_choices, mock_balancer):
    mock_provider1 = LLMProviderConfig(model_id="model1", impl=MockLLMProvider(), weight=1)
    mock_provider2 = LLMProviderConfig(model_id="model2", impl=MockLLMProvider(), weight=2)
    mock_choices.return_value = [1]

    result = mock_balancer._select_weighted_provider([mock_provider1, mock_provider2], "test_model", "weight")
    assert result == mock_provider2
    mock_choices.assert_called_once_with(range(2), weights=[0.3333333333333333, 0.6666666666666666])


@patch("random.choice")
def test_select_weighted_provider_with_zero_weights(mock_choice, mock_balancer):
    provider1 = LLMProviderConfig(model_id="zero1", weight=0, impl=MockLLMProvider())
    provider2 = LLMProviderConfig(model_id="zero2", weight=0, impl=MockLLMProvider())
    mock_choice.return_value = provider1

    result = mock_balancer._select_weighted_provider([provider1, provider2], "test_model", "weight")
    assert result == provider1
    mock_balancer.logger.debug.assert_called_with("All providers have 0 weight, selecting randomly.")


@pytest.mark.asyncio
async def test_filter_over_limit_providers_returns_valid_providers(mock_balancer):
    mock_balancer.mock_rpm_tpm_manager.rpm_usage_at_minute = AsyncMock(side_effect=[99, 49])
    overlimit_provider = MagicMock(spec=LLMProviderConfig)
    overlimit_provider.id = "overlimit_provider"
    overlimit_provider.rpm = 99

    valid_provider = MagicMock(spec=LLMProviderConfig)
    valid_provider.id = "valid_provider"
    valid_provider.rpm = 100

    filtered = await mock_balancer._filter_over_limit_providers("test_group", [overlimit_provider, valid_provider])
    assert filtered == [valid_provider]
