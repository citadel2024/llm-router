import math
from unittest.mock import MagicMock, Mock

import pytest

from src.config import LogConfiguration
from src.config.config import LLMProviderConfig, LoadBalancerConfig
from src.load_balance.lowest_tpm import LowestTPMBalancer
from src.load_balance.strategy import LoadBalancerStrategy


class TestLowestTPMBalancer:
    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.get_cache = MagicMock()
        return cache

    @pytest.fixture
    def mock_logger(self):
        logger = LogConfiguration()
        return logger

    @pytest.fixture
    def mock_load_balancer_config(self):
        return LoadBalancerConfig(strategy=LoadBalancerStrategy.LOWEST_TPM_BALANCER)

    @pytest.fixture
    def mock_balancer(self, mock_cache, mock_logger, mock_load_balancer_config):
        return LowestTPMBalancer(mock_cache, mock_logger, mock_load_balancer_config)

    @pytest.fixture
    def mock_providers(self, mock_balancer):
        return [
            LLMProviderConfig("model-1", mock_balancer, 10, 100),
            LLMProviderConfig("model-2", mock_balancer, 20, 200),
        ]

    def test_select_provider_with_no_providers(self, mock_balancer):
        result = mock_balancer.schedule_provider("test-group", [], text="text")
        assert result is None

    def test_select_provider_no_usage_data(self, mock_balancer, mock_providers, mock_cache):
        # first call to get_cache returns None, treat all providers tpm as zero, choose first provider
        mock_cache.get_cache.return_value = None
        messages = [{"role": "user", "content": "test"}]

        result = mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

        assert result == mock_providers[0]
        assert mock_cache.get_cache.call_count == 2

    def test_get_usage_data(self, mock_balancer, mock_providers):
        # first provider has usage data, second provider does not, expect second provider to have zero usage
        cache_keys = {
            "tpm": "test:tpm:00-00",
            "rpm": "test:rpm:00-00"
        }
        existing_tpm = {mock_providers[0].id: 50}
        existing_rpm = {mock_providers[0].id: 10}
        mock_balancer.lb_cache.get_cache.side_effect = [existing_tpm, existing_rpm]

        result = mock_balancer._get_usage_data(cache_keys)

        assert "tpm" in result
        assert "rpm" in result
        assert result["tpm"][mock_providers[0].id] == 50
        assert mock_providers[1].id not in result["tpm"]
        assert result["rpm"][mock_providers[0].id] == 10
        assert mock_providers[1].id not in result["rpm"]

    @pytest.mark.parametrize("current_tpm,input_tokens,expected_id", [
        (10, 20, "model-1"),  # model-1 has lower current usage
        (90, 20, "model-2"),  # model-1 would exceed TPM limit
        (195, 10, None),  # both would exceed limits
    ])
    def test_find_optimal_provider(self, mock_balancer, mock_providers, current_tpm, input_tokens, expected_id):
        usage_data = {
            "tpm": {mock_providers[0].id: current_tpm, mock_providers[1].id: current_tpm + 10},
            "rpm": {mock_providers[0].id: 5, mock_providers[1].id: 5}
        }
        result = mock_balancer._find_optimal_provider(mock_providers, usage_data, input_tokens)

        if expected_id:
            assert result.model_id == expected_id
        else:
            assert result is None

    def test_select_lowest_tpm(self, mock_balancer, mock_providers, mock_cache):
        tpm_data = {"model-1": 30, "model-2": 31}
        rpm_data = {mock_providers[0].id: 5, mock_providers[1].id: 8}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

        assert result is not None
        assert result.model_id == "model-1"
        assert mock_cache.get_cache.call_count == 2

    def test_select_from_one_candidate(self, mock_balancer, mock_providers, mock_cache):
        tpm_data = {"model-1": 30, "model-2": 31}
        rpm_data = {mock_providers[0].id: 10, mock_providers[1].id: 8}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

        assert result is not None
        assert result.model_id == "model-2"
        assert mock_cache.get_cache.call_count == 2

    def test_select_from_empty_rpm(self, mock_balancer, mock_cache):
        providers = [
            LLMProviderConfig("model-1", mock_balancer, tpm=100),
            LLMProviderConfig("model-2", mock_balancer, tpm=200),
        ]
        tpm_data = {"model-1": 30, "model-2": 31}
        rpm_data = {providers[0].id: 1000000, providers[1].id: 1000000}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = mock_balancer.schedule_provider("test-group", providers, messages=messages)

        assert result is not None
        assert result.model_id == "model-1"
        assert mock_cache.get_cache.call_count == 2

    def test_select_from_empty_tpm(self, mock_balancer, mock_cache):
        providers = [
            LLMProviderConfig("model-1", mock_balancer, rpm=10),
            LLMProviderConfig("model-2", mock_balancer, rpm=10),
        ]
        tpm_data = {"model-1": 1000000, "model-2": 1000000}
        rpm_data = {providers[0].id: 10, providers[1].id: 8}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = mock_balancer.schedule_provider("test-group", providers, messages=messages)

        assert result is not None
        assert result.model_id == "model-2"
        assert mock_cache.get_cache.call_count == 2

    @pytest.mark.parametrize("model_id, max_tpm, max_rpm, rpm_dict, current_tpm, input_tokens, expected", [
        ("model-1", 100, 10, {"model-1": 5}, 50, 30, True),
        ("model-1", 100, 10, {}, 80, 30, False),
        ("model-1", 100, 10, {}, 70, 30, True),
        ("model-1", 100, 10, {"model-1": 8}, 50, 30, True),
        ("model-1", 100, 10, {}, 50, 30, True),
        ("model-1", 100, 10, {"model-2": 5}, 50, 30, True),
        ("model-1", math.inf, math.inf, {"model-1": 1000000}, 1000000, 1000000, True),
        ("model-1", 100, 10, {"model-1": 0}, 0, 0, True)
    ])
    def test_is_model_available(
            self,
            model_id: str,
            max_tpm: float,
            max_rpm: float,
            rpm_dict: dict,
            current_tpm: int,
            input_tokens: int,
            expected: bool
    ):
        result = LowestTPMBalancer._is_model_available(
            provider_id=model_id,
            max_tpm=max_tpm,
            max_rpm=max_rpm,
            rpm_dict=rpm_dict,
            current_tpm=current_tpm,
            input_tokens=input_tokens
        )
        assert result == expected, (
            f"Failed for model_id={model_id}, max_tpm={max_tpm}, max_rpm={max_rpm}, "
            f"rpm_dict={rpm_dict}, current_tpm={current_tpm}, input_tokens={input_tokens}"
        )
