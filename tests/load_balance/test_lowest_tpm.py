import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import LogConfiguration
from src.config.config import LLMProviderConfig, LoadBalancerConfig
from src.utils.context import RouterContext, router_context
from src.load_balance.strategy import LoadBalancerStrategy
from src.load_balance.lowest_tpm import LowestTPMBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    return cache


@pytest.fixture
def mock_logger():
    logger = LogConfiguration()
    return logger


@pytest.fixture
def mock_load_balancer_config():
    return LoadBalancerConfig(strategy=LoadBalancerStrategy.LOWEST_TPM_BALANCER)


@pytest.fixture
def mock_balancer(mock_cache, mock_logger, mock_load_balancer_config):
    return LowestTPMBalancer(mock_cache, mock_logger, mock_load_balancer_config)


@pytest.fixture
def mock_providers(mock_balancer):
    return [
        LLMProviderConfig("model-1", mock_balancer, 10, 100),
        LLMProviderConfig("model-2", mock_balancer, 20, 200),
    ]


@pytest.mark.asyncio
async def test_select_provider_with_no_providers(mock_balancer):
    result = await mock_balancer.schedule_provider("test-group", [], text="text")
    assert result is None


@pytest.mark.asyncio
async def test_select_provider_no_usage_data(mock_balancer, mock_providers, mock_cache):
    # first call to get_cache returns None, treat all providers tpm as zero, choose first provider
    tpm_data1 = RpmTpmManager.Usage(used=0, occupying=0).to_json()
    rpm_data1 = RpmTpmManager.Usage(used=0, occupying=0).to_json()
    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data1, tpm_data1, rpm_data1])
    messages = [{"role": "user", "content": "test"}]
    router_context.set(RouterContext())
    result = await mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

    assert result == mock_providers[0]
    assert mock_cache.async_get_value.call_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_tpm,input_tokens,expected_id",
    [
        (10, 20, "model-1"),  # model-1 has lower current usage
        (90, 20, "model-2"),  # model-1 would exceed TPM limit
        (195, 10, None),  # both would exceed limits
    ],
)
async def test_find_optimal_provider(mock_balancer, mock_cache, mock_providers, current_tpm, input_tokens, expected_id):
    tpm_data1 = RpmTpmManager.Usage(used=current_tpm, occupying=0).to_json()
    tpm_data2 = RpmTpmManager.Usage(used=current_tpm + 10, occupying=0).to_json()
    rpm_data = RpmTpmManager.Usage(used=5, occupying=0).to_json()
    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data, tpm_data2, rpm_data])
    router_context.set(RouterContext())
    result = await mock_balancer._find_optimal_provider("group", mock_providers, input_tokens)

    if expected_id:
        assert result.model_id == expected_id
    else:
        assert result is None


@pytest.mark.asyncio
async def test_select_lowest_tpm(mock_balancer, mock_providers, mock_cache):
    tpm_data1 = RpmTpmManager.Usage(used=30, occupying=0).to_json()
    tpm_data2 = RpmTpmManager.Usage(used=31, occupying=0).to_json()
    rpm_data1 = RpmTpmManager.Usage(used=5, occupying=0).to_json()
    rpm_data2 = RpmTpmManager.Usage(used=8, occupying=0).to_json()
    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data1, tpm_data2, rpm_data2])
    messages = [{"role": "user", "content": "test message"}]
    router_context.set(RouterContext())
    result = await mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

    assert result is not None
    assert result.model_id == "model-1"
    assert mock_cache.async_get_value.call_count == 4


@pytest.mark.asyncio
async def test_select_from_one_candidate(mock_balancer, mock_providers, mock_cache):
    tpm_data1 = RpmTpmManager.Usage(used=30, occupying=0).to_json()
    tpm_data2 = RpmTpmManager.Usage(used=31, occupying=0).to_json()
    rpm_data1 = RpmTpmManager.Usage(used=10, occupying=0).to_json()
    rpm_data2 = RpmTpmManager.Usage(used=8, occupying=0).to_json()
    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data1, tpm_data2, rpm_data2])

    messages = [{"role": "user", "content": "test message"}]
    router_context.set(RouterContext())
    result = await mock_balancer.schedule_provider("test-group", mock_providers, messages=messages)

    assert result is not None
    assert result.model_id == "model-2"
    assert mock_cache.async_get_value.call_count == 4


@pytest.mark.asyncio
async def test_select_from_empty_rpm(mock_balancer, mock_cache):
    providers = [
        LLMProviderConfig("model-1", mock_balancer, tpm=100),
        LLMProviderConfig("model-2", mock_balancer, tpm=200),
    ]
    rpm_data = RpmTpmManager.Usage(used=1000000, occupying=0).to_json()
    tpm_data1 = RpmTpmManager.Usage(used=30, occupying=0).to_json()
    tpm_data2 = RpmTpmManager.Usage(used=31, occupying=0).to_json()
    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data, tpm_data2, rpm_data])

    messages = [{"role": "user", "content": "test message"}]
    router_context.set(RouterContext())
    result = await mock_balancer.schedule_provider("test-group", providers, messages=messages)

    assert result is not None
    assert result.model_id == "model-1"
    assert mock_cache.async_get_value.call_count == 4


@pytest.mark.asyncio
async def test_select_from_empty_tpm(mock_balancer, mock_cache):
    providers = [
        LLMProviderConfig("model-1", mock_balancer, rpm=10),
        LLMProviderConfig("model-2", mock_balancer, rpm=10),
    ]
    rpm_data1 = RpmTpmManager.Usage(used=10, occupying=0).to_json()
    rpm_data2 = RpmTpmManager.Usage(used=8, occupying=0).to_json()
    tpm_data1 = RpmTpmManager.Usage(used=1000000, occupying=0).to_json()

    mock_cache.async_get_value = AsyncMock(side_effect=[tpm_data1, rpm_data1, tpm_data1, rpm_data2])

    messages = [{"role": "user", "content": "test message"}]
    router_context.set(RouterContext())
    result = await mock_balancer.schedule_provider("test-group", providers, messages=messages)

    assert result is not None
    assert result.model_id == "model-2"
    assert mock_cache.async_get_value.call_count == 4


@pytest.mark.parametrize(
    "max_tpm, max_rpm, rpm, current_tpm, input_tokens, expected",
    [
        (100, 10, 5, 50, 30, True),
        (100, 10, 0, 80, 30, False),
        (100, 10, 0, 70, 30, True),
        (100, 10, 8, 50, 30, True),
        (100, 10, 0, 50, 30, True),
        (100, 10, 5, 50, 30, True),
        (math.inf, math.inf, 1000000, 1000000, 1000000, True),
        (100, 10, 0, 0, 0, True),
    ],
)
def test_is_model_available(
    max_tpm: float, max_rpm: float, rpm: int, current_tpm: int, input_tokens: int, expected: bool
):
    result = LowestTPMBalancer._is_model_available(
        max_tpm=max_tpm,
        max_rpm=max_rpm,
        rpm=rpm,
        current_tpm=current_tpm,
        input_tokens=input_tokens,
    )
    assert result == expected, (
        f"Failed for max_tpm={max_tpm}, max_rpm={max_rpm}, "
        f"rpm={rpm}, current_tpm={current_tpm}, input_tokens={input_tokens}"
    )
