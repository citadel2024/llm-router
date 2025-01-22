import math
from unittest.mock import MagicMock, Mock

import pytest

from src.config import LogConfiguration
from src.config.config import LLMProviderConfig
from src.load_balance.lowest_tpm import LowestTPMBalancer


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
    def balancer(self, mock_cache, mock_logger):
        return LowestTPMBalancer(mock_cache, mock_logger)

    @pytest.fixture
    def sample_providers(self, balancer):
        return [
            LLMProviderConfig("model-1", balancer, 10, 100),
            LLMProviderConfig("model-2", balancer, 20, 200),
        ]

    def test_select_provider_with_no_providers(self, balancer):
        result = balancer.schedule_provider("test-group", [])
        assert result is None

    def test_select_provider_no_usage_data(self, balancer, sample_providers, mock_cache):
        mock_cache.get_cache.return_value = None
        messages = [{"role": "user", "content": "test"}]

        result = balancer.schedule_provider("test-group", sample_providers, messages)

        assert result == sample_providers[0]
        assert mock_cache.get_cache.call_count == 2

    def test_get_usage_data(self, balancer, sample_providers):
        cache_keys = {
            "tpm": "test:tpm:00-00",
            "rpm": "test:rpm:00-00"
        }
        existing_tpm = {sample_providers[0].id: 50}
        balancer.lb_cache.get_cache.side_effect = [existing_tpm, {}]

        result = balancer._get_usage_data(cache_keys, sample_providers)

        assert "tpm" in result
        assert "rpm" in result
        assert result["tpm"][sample_providers[0].id] == 50
        assert result["tpm"][sample_providers[1].id] == 0

    @pytest.mark.parametrize("current_tpm,input_tokens,expected_id", [
        (10, 20, "model-1"),  # model-1 has lower current usage
        (90, 20, "model-2"),  # model-1 would exceed TPM limit
        (195, 10, None),  # both would exceed limits
    ])
    def test_find_optimal_provider(self, balancer, sample_providers, current_tpm, input_tokens, expected_id):
        usage_data = {
            "tpm": {"model-1": current_tpm, "model-2": current_tpm + 10},
            "rpm": {"model-1": 5, "model-2": 5}
        }
        result = balancer._find_optimal_provider(sample_providers, usage_data, input_tokens)

        if expected_id:
            assert result.model_id == expected_id
        else:
            assert result is None

    def test_select_lowest_tpm(self, balancer, sample_providers, mock_cache):
        tpm_data = {"model-1": 30, "model-2": 31}
        rpm_data = {"model-1": 5, "model-2": 8}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = balancer.schedule_provider("test-group", sample_providers, messages)

        assert result is not None
        assert result.model_id == "model-1"
        assert mock_cache.get_cache.call_count == 2

    def test_select_from_one_candidate(self, balancer, sample_providers, mock_cache):
        tpm_data = {"model-1": 30, "model-2": 31}
        rpm_data = {"model-1": 10, "model-2": 8}

        mock_cache.get_cache.side_effect = [tpm_data, rpm_data]

        messages = [{"role": "user", "content": "test message"}]
        result = balancer.schedule_provider("test-group", sample_providers, messages)

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
            model_id=model_id,
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
