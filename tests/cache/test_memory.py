import time
import asyncio

import pytest
import pytest_asyncio

from src.config import LogConfiguration
from src.cache.memory import MemoryCache


@pytest.fixture
def mock_log_cfg():
    return LogConfiguration()


@pytest_asyncio.fixture
async def mock_cache(mock_log_cfg):
    instance = MemoryCache(
        log_cfg=mock_log_cfg, max_size_in_memory=2, default_ttl=3600, cleanup_interval=5, num_buckets=1
    )
    await instance.start_cleanup_task()
    yield instance
    await instance.stop_cleanup_task()


@pytest_asyncio.fixture
async def mock_cache_small(mock_log_cfg):
    instance = MemoryCache(log_cfg=mock_log_cfg, max_size_in_memory=2, default_ttl=1, cleanup_interval=1, num_buckets=1)
    await instance.start_cleanup_task()
    yield instance
    await instance.stop_cleanup_task()


@pytest.mark.asyncio
async def test_async_set_get(mock_cache):
    await mock_cache.async_set_value("key", "value")
    assert await mock_cache.async_get_value("key") == "value"


@pytest.mark.asyncio
async def test_async_get_expired(mock_cache, monkeypatch):
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    await mock_cache.async_set_value("key", "value", ttl=1)
    monkeypatch.setattr(time, "time", lambda: now + 2)
    assert await mock_cache.async_get_value("key") is None


@pytest.mark.asyncio
async def test_bucket_full(mock_cache_small, caplog):
    await mock_cache_small.async_set_value("key1", "v1")
    await mock_cache_small.async_set_value("key2", "v2")
    await mock_cache_small.async_set_value("key3", "v3")
    assert "bucket 0 is full" in caplog.text


@pytest.mark.asyncio
async def test_evict_expired_entries(mock_cache, monkeypatch):
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    await mock_cache.async_set_value("key", "value", ttl=1)
    monkeypatch.setattr(time, "time", lambda: now + 2)
    await mock_cache._evict_expired_entries(0)
    assert await mock_cache.async_get_value("key") is None


@pytest.mark.asyncio
async def test_periodic_cleanup(mock_cache_small):
    await mock_cache_small.async_set_value("key", "value", ttl=0.5)
    await asyncio.sleep(1.5)
    assert await mock_cache_small.async_get_value("key") is None


@pytest.mark.asyncio
async def test_get_bucket_index(mock_cache):
    key = "test_key"
    idx = mock_cache._get_bucket_index(key)
    assert 0 <= idx < mock_cache.num_buckets
    assert mock_cache._get_bucket_index(key) == idx


@pytest.mark.asyncio
async def test_concurrent_writes(mock_cache):
    async def write(key, value):
        await mock_cache.async_set_value(key, value)

    tasks = [write("key", i) for i in range(10)]
    await asyncio.gather(*tasks)
    assert await mock_cache.async_get_value("key") == 9
