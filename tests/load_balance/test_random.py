import random
from unittest.mock import Mock, AsyncMock

import pytest

from src.config import LogConfiguration, LoadBalancerConfig
from src.cache.base import BaseCache
from src.model.message import ChatCompletionUserMessage
from src.load_balance.random import RandomBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    return cache


@pytest.fixture
def mock_rpm_tpm_manager(mock_cache):
    return RpmTpmManager(mock_cache, LogConfiguration())


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for RandomBalancer."""
    lb_cache = Mock(spec=BaseCache)
    log_cfg = LogConfiguration()
    load_balancer_config = Mock(spec=LoadBalancerConfig)
    return lb_cache, log_cfg, load_balancer_config


def test_random_balancer_initialization(mock_dependencies, mock_rpm_tpm_manager):
    """Test that RandomBalancer can be initialized correctly."""
    lb_cache, log_cfg, load_balancer_config = mock_dependencies
    balancer = RandomBalancer(lb_cache, log_cfg, load_balancer_config, mock_rpm_tpm_manager)

    assert balancer is not None
    assert balancer.lb_cache == lb_cache
    assert balancer.logger is not None


@pytest.mark.asyncio
async def test_schedule_provider_empty_providers(mock_dependencies, mock_rpm_tpm_manager):
    """Test scheduling when no healthy providers are available."""
    lb_cache, log_cfg, load_balancer_config = mock_dependencies
    balancer = RandomBalancer(lb_cache, log_cfg, load_balancer_config, mock_rpm_tpm_manager)

    result = await balancer.schedule_provider("test_group", [])
    assert result is None


@pytest.mark.asyncio
async def test_schedule_provider_single_provider(mock_dependencies, mock_rpm_tpm_manager):
    """Test scheduling when only one provider is available."""
    lb_cache, log_cfg, load_balancer_config = mock_dependencies
    balancer = RandomBalancer(lb_cache, log_cfg, load_balancer_config, mock_rpm_tpm_manager)

    providers = [{"name": "provider1", "endpoint": "http://test1.com"}]
    result = await balancer.schedule_provider("test_group", providers)

    assert result == providers[0]


@pytest.mark.asyncio
async def test_schedule_provider_multiple_providers(mock_dependencies, mock_rpm_tpm_manager):
    """Test scheduling with multiple providers."""
    lb_cache, log_cfg, load_balancer_config = mock_dependencies
    balancer = RandomBalancer(lb_cache, log_cfg, load_balancer_config, mock_rpm_tpm_manager)

    providers = [
        {"name": "provider1", "endpoint": "http://test1.com"},
        {"name": "provider2", "endpoint": "http://test2.com"},
        {"name": "provider3", "endpoint": "http://test3.com"},
    ]
    random.seed(42)
    # To test randomness, we'll run multiple times and ensure
    # different providers can be selected
    selected_providers = set()
    for _ in range(10):
        result = await balancer.schedule_provider("test_group", providers)
        selected_providers.add(result["name"])

    # Ensure all providers can be selected
    assert len(selected_providers) == len(providers)


@pytest.mark.asyncio
async def test_schedule_provider_with_messages(mock_dependencies, mock_rpm_tpm_manager):
    """Test scheduling with additional messages' parameter."""
    lb_cache, log_cfg, load_balancer_config = mock_dependencies
    balancer = RandomBalancer(lb_cache, log_cfg, load_balancer_config, mock_rpm_tpm_manager)

    providers = [{"name": "provider1", "endpoint": "http://test1.com"}]
    messages = [ChatCompletionUserMessage(role="user", content="Test message")]

    result = await balancer.schedule_provider("test_group", providers, messages)

    # Messages should not affect the random selection
    assert result == providers[0]
