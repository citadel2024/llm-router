from unittest.mock import MagicMock, patch

import pytest

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.load_balance.capacity_based import CapacityBasedBalancer
from tests.mock_provider import MockLLMProvider


@pytest.fixture
def mock_lb_cache():
    return MagicMock(spec=BaseCache)


@pytest.fixture
def mock_log_cfg():
    return LogConfiguration()


@pytest.fixture
def mock_lb_config():
    config = MagicMock(spec=LoadBalancerConfig)
    config.capacity_dimension = "weight"
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


def test_schedule_provider_returns_none_when_no_healthy_providers(mock_balancer):
    result = mock_balancer.schedule_provider("test_group", [])
    assert result is None


def test_schedule_provider_returns_none_when_all_over_limit(mock_balancer, mock_provider):
    with patch.object(mock_balancer, '_filter_over_limit_providers', return_value=[]):
        result = mock_balancer.schedule_provider("test_group", [mock_provider])
        assert result is None
        mock_balancer.logger.warning.assert_called_once()


def test_schedule_provider_selects_provider_correctly(mock_balancer, mock_provider):
    with patch.object(mock_balancer, '_filter_over_limit_providers', return_value=[mock_provider]), \
            patch.object(mock_balancer, '_select_weighted_provider', return_value=mock_provider):
        result = mock_balancer.schedule_provider("test_group", [mock_provider])
        assert result == mock_provider


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


@patch("src.load_balance.capacity_based.datetime")
def test_filter_over_limit_providers_returns_valid_providers(mock_datetime, mock_balancer, mock_lb_cache):
    mock_datetime.now.return_value.strftime.return_value = "12-00"
    mock_lb_cache.get_cache.return_value = {
        "overlimit_provider": 99,
        "valid_provider": 50
    }

    overlimit_provider = MagicMock(spec=LLMProviderConfig)
    overlimit_provider.id = "overlimit_provider"
    overlimit_provider.rpm = 99

    valid_provider = MagicMock(spec=LLMProviderConfig)
    valid_provider.id = "valid_provider"
    valid_provider.rpm = 100

    filtered = mock_balancer._filter_over_limit_providers("test_group", [overlimit_provider, valid_provider])

    assert filtered == [valid_provider]
    mock_lb_cache.get_cache.assert_called_once_with("test_group:rpm:12-00")
