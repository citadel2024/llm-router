from unittest.mock import MagicMock, patch

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.load_balance.capacity_based import CapacityBasedBalancer
from tests.mock_provider import MockLLMProvider


class TestCapacityBasedBalancer:
    def setup_method(self):
        self.mock_lb_cache = MagicMock(spec=BaseCache)
        self.mock_log_cfg = LogConfiguration()
        self.mock_load_balancer_config = MagicMock(spec=LoadBalancerConfig)
        self.mock_load_balancer_config.capacity_dimension = "weight"

        self.balancer = CapacityBasedBalancer(
            self.mock_lb_cache,
            self.mock_log_cfg,
            self.mock_load_balancer_config
        )
        self.balancer.logger = MagicMock()

    def test_schedule_provider_returns_none_when_no_healthy_providers(self):
        result = self.balancer.schedule_provider("test_group", [])
        assert result is None

    def test_schedule_provider_returns_none_when_all_over_limit(self):
        self.balancer._filter_over_limit_providers = MagicMock(return_value=[])
        providers = [MagicMock(spec=LLMProviderConfig)]

        result = self.balancer.schedule_provider("test_group", providers)
        assert result is None
        self.balancer.logger.warning.assert_called_once()

    def test_schedule_provider_selects_provider_correctly(self):
        mock_provider = MagicMock(spec=LLMProviderConfig)
        self.balancer._filter_over_limit_providers = MagicMock(return_value=[mock_provider])
        self.balancer._select_weighted_provider = MagicMock(return_value=mock_provider)

        result = self.balancer.schedule_provider("test_group", [mock_provider])
        assert result == mock_provider

    @patch("random.choices")
    def test_select_weighted_provider_with_valid_weights(self, mock_choices):
        mock_provider1 = LLMProviderConfig(model_id="model1", impl=MockLLMProvider(), weight=1)
        mock_provider2 = LLMProviderConfig(model_id="model2", impl=MockLLMProvider(), weight=2)
        mock_choices.return_value = [1]

        result = self.balancer._select_weighted_provider(
            [mock_provider1, mock_provider2], "test_model", "weight"
        )
        assert result == mock_provider2

    @patch("random.choice")
    def test_select_weighted_provider_with_zero_weights(self, mock_choice):
        mock_provider = MagicMock(spec=LLMProviderConfig)
        mock_provider.weight = 0
        mock_choice.return_value = mock_provider

        result = self.balancer._select_weighted_provider(
            [mock_provider, mock_provider], "test_model", "weight"
        )
        assert result == mock_provider
        self.balancer.logger.debug.assert_any_call("All providers have 0 weight, selecting randomly.")

    @patch("src.load_balance.capacity_based.datetime")
    def test_filter_over_limit_providers_returns_valid_providers(self, mock_datetime):
        mock_datetime.now.return_value.strftime.return_value = "12-00"
        self.mock_lb_cache.get_cache.return_value = {
            "overlimit_provider": 99,
            "valid_provider": 50
        }

        overlimit_provider = MagicMock(spec=LLMProviderConfig)
        overlimit_provider.id = "overlimit_provider"
        overlimit_provider.rpm = 99

        valid_provider = MagicMock(spec=LLMProviderConfig)
        valid_provider.id = "valid_provider"
        valid_provider.rpm = 100

        providers = [overlimit_provider, valid_provider]
        filtered = self.balancer._filter_over_limit_providers("test_group", providers)

        assert filtered == [valid_provider]
        self.mock_lb_cache.get_cache.assert_called_once_with("test_group:rpm:12-00")
