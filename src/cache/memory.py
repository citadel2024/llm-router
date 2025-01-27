import time
import asyncio
from typing import Any, Optional

from src.config import LogConfiguration
from src.cache.base import BaseCache
from src.router.log import get_logger


class MemoryCache(BaseCache):
    def __init__(
        self,
        log_cfg: LogConfiguration,
        max_size_in_memory: int = 10 * 64 * 60,
        default_ttl: int = 60 * 60,
        cleanup_interval: int = 60 * 5,
        num_buckets: int = 64,
    ):
        """
        Memory cache
        :param max_size_in_memory: let's suppose we got 10 types of keys, and we want to store 64 keys for each type in 60 minutes.
        :param default_ttl: 60 minutes
        :param cleanup_interval: 5 minutes
        """
        super().__init__(default_ttl)
        self.logger = get_logger(__name__, log_cfg)
        self.max_size_in_memory = max_size_in_memory
        self.cleanup_interval = cleanup_interval
        self.num_buckets = num_buckets
        self.locks = [asyncio.Lock() for _ in range(num_buckets)]
        self.cache_buckets: list[dict[str, Any]] = [dict() for _ in range(num_buckets)]
        self.ttl_buckets: list[dict[str, float]] = [dict() for _ in range(num_buckets)]
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self):
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

    def _get_bucket_index(self, key: str) -> int:
        """
        Hash based on the key
        :param key:
        :return:
        """
        return abs(hash(key)) % self.num_buckets

    async def _evict_expired_entries(self, bucket_idx: Optional[int] = None):
        """
        Evict expired entries from the cache
        :param bucket_idx:
        :return:
        """
        if bucket_idx is not None:
            async with self.locks[bucket_idx]:
                self._clean_bucket(bucket_idx)
        else:
            for idx in range(self.num_buckets):
                async with self.locks[idx]:
                    self._clean_bucket(idx)

    def _clean_bucket(self, bucket_idx: int):
        """
        Clean the bucket
        :param bucket_idx:
        :return:
        """
        now = time.time()
        cache = self.cache_buckets[bucket_idx]
        ttl = self.ttl_buckets[bucket_idx]
        to_remove = [k for k, v in ttl.items() if v < now]
        for k in to_remove:
            del cache[k]
            del ttl[k]

    async def async_set_value(self, key: str, value: Any, ttl: Optional[int] = None, **_kwargs):
        """
        Set value in the cache
        :param key:
        :param value:
        :param ttl:
        :param _kwargs:
        :return:
        """
        bucket_idx = self._get_bucket_index(key)
        max_per_bucket = self.max_size_in_memory // self.num_buckets
        async with self.locks[bucket_idx]:
            cache = self.cache_buckets[bucket_idx]
            ttl_dict = self.ttl_buckets[bucket_idx]
            if len(cache) >= max_per_bucket:
                self._clean_bucket(bucket_idx)
                if len(cache) >= max_per_bucket:
                    self.logger.warning(f"bucket {bucket_idx} is full")
            cache[key] = value
            ttl_dict[key] = time.time() + (ttl if ttl is not None else self.default_ttl)

    async def async_get_value(self, key: str, **_kwargs) -> Any:
        """
        Get value from the cache
        :param key:
        :param _kwargs:
        :return:
        """
        bucket_idx = self._get_bucket_index(key)
        async with self.locks[bucket_idx]:
            cache = self.cache_buckets[bucket_idx]
            ttl_dict = self.ttl_buckets[bucket_idx]
            if key not in cache:
                return None
            if ttl_dict[key] < time.time():
                del cache[key]
                del ttl_dict[key]
                return None
            return cache[key]

    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self._evict_expired_entries()
