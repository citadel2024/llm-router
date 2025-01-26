import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import LogConfiguration
from src.cache.base import BaseCache
from src.utils.context import RouterContext, router_context
from src.load_balance.rpm_tpm_manager import Dimension, RpmTpmManager


@pytest.fixture
def mock_cache():
    return AsyncMock(spec=BaseCache)


@pytest.fixture
def mock_rpm_tpm_manager(mock_cache):
    manager = RpmTpmManager(mock_cache, LogConfiguration())
    manager.logger = MagicMock()
    return manager


@pytest.mark.asyncio
async def test_increase_rpm_occupied(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.increase_rpm_occupied("group1", "provider1", 2)

    expected_key = "rpm:group1:provider1:202310101200"
    expected_usage = json.dumps({"used": 0, "occupying": 2})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, expected_usage, ttl=86400)
    assert expected_key in mock_rpm_tpm_manager.locks


@pytest.mark.asyncio
async def test_increase_tpm_occupied(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.increase_tpm_occupied("group1", "provider1", 5)

    expected_key = "rpm:group1:provider1:202310101200"
    expected_usage = json.dumps({"used": 0, "occupying": 5})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, expected_usage, ttl=86400)


@pytest.mark.asyncio
async def test_update_rpm_used_usage(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    initial_usage = {"used": 0, "occupying": 3}
    mock_cache.async_get_value.return_value = json.dumps(initial_usage)

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.update_rpm_used_usage("group1", "provider1", 2)

    expected_key = "rpm:group1:provider1:202310101200"
    updated_usage = json.dumps({"used": 2, "occupying": 1})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, updated_usage)


@pytest.mark.asyncio
async def test_update_tpm_used_usage(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    initial_usage = {"used": 10, "occupying": 5}
    mock_cache.async_get_value.return_value = json.dumps(initial_usage)

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.update_tpm_used_usage("group1", "provider1", 3)

    expected_key = "tpm:group1:provider1:202310101200"
    updated_usage = json.dumps({"used": 13, "occupying": 2})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, updated_usage)


@pytest.mark.asyncio
async def test_release_rpm_occupied(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    initial_usage = {"used": 2, "occupying": 5}
    mock_cache.async_get_value.return_value = json.dumps(initial_usage)

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.release_rpm_occupied("group1", "provider1", 3)

    expected_key = "rpm:group1:provider1:202310101200"
    updated_usage = json.dumps({"used": 2, "occupying": 2})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, updated_usage)


@pytest.mark.asyncio
async def test_release_tpm_occupied(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    initial_usage = {"used": 8, "occupying": 4}
    mock_cache.async_get_value.return_value = json.dumps(initial_usage)

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.release_tpm_occupied("group1", "provider1", 2)

    expected_key = "tpm:group1:provider1:202310101200"
    updated_usage = json.dumps({"used": 8, "occupying": 2})
    mock_cache.async_set_value.assert_awaited_once_with(expected_key, updated_usage)


@pytest.mark.asyncio
async def test_rpm_usage_at_minute_exists(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    mock_cache.async_get_value.return_value = json.dumps({"used": 5, "occupying": 3})

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        total = await mock_rpm_tpm_manager.rpm_usage_at_minute("group1", "provider1")

    assert total == 8


@pytest.mark.asyncio
async def test_rpm_usage_at_minute_not_exists(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    mock_cache.async_get_value.return_value = None

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        total = await mock_rpm_tpm_manager.rpm_usage_at_minute("group1", "provider1")

    assert total == 0


@pytest.mark.asyncio
async def test_lock_reuse(mock_rpm_tpm_manager):
    router_context.set(RouterContext())
    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        key1, lock1 = mock_rpm_tpm_manager._fetch_or_create_lock(Dimension.RPM, "group1", "provider1")
        key2, lock2 = mock_rpm_tpm_manager._fetch_or_create_lock(Dimension.RPM, "group1", "provider1")

    assert key1 == key2
    assert lock1 is lock2


@pytest.mark.asyncio
async def test_update_usage_no_data_logs_error(mock_rpm_tpm_manager, mock_cache):
    router_context.set(RouterContext())
    mock_cache.async_get_value.return_value = None

    with patch.object(router_context.get(), "start_minute_str", return_value="202310101200"):
        await mock_rpm_tpm_manager.update_rpm_used_usage("group1", "provider1")

    mock_rpm_tpm_manager.logger.error.assert_called_once_with(
        "No usage data found for rpm:group1:provider1:202310101200"
    )
