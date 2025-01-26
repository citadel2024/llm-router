import json
import asyncio
from enum import Enum
from dataclasses import asdict, dataclass

from src.config import LogConfiguration
from src.cache.base import BaseCache
from src.router.log import get_logger
from src.utils.context import RouterContext, router_context


class Dimension(Enum):
    RPM = "rpm"
    TPM = "tpm"


class RpmTpmManager:
    """
    Manages RPM and TPM limits for models.

    Before invoking a provider, the RPM 'occupying' for the provider is incremented by 1 (parameter default value, user can change it).
    If the call is successful, the RPM 'occupying' is decremented by 1, and the 'used' count for the provider is incremented by 1.
    If the call fails, the RPM 'occupying' is still decremented by 1, but the 'used' count remains the same.

    This process ensures that the RPM usage is accurately tracked in real time, even when the call may take time to complete.
    In high concurrency situations, locks are used to avoid race conditions when updating the RPM usage for each provider.

    The RPM usage is tracked in a Redis Hash like structure(TPM is same as RPM):
    - The key is formatted as `rpm:{group_name}:{provider_id}:{minute}`.
    - The value `Usage(used, occupying)` represents the current RPM usage for that provider in that minute.
    """

    DEFAULT_TTL = 60 * 60 * 24

    @dataclass
    class Usage:
        used: int
        occupying: int

        def total(self):
            return self.used + self.occupying

        def serialize(self):
            return json.dumps(asdict(self))

        @classmethod
        def deserialize(cls, data):
            return cls(**json.loads(data))

    def __init__(self, cache: BaseCache, log_cfg: LogConfiguration):
        self.cache = cache
        self.logger = get_logger(__name__, log_cfg)
        self.locks: dict[str, asyncio.Lock] = {}

    async def _increase_occupied(self, dimension: Dimension, group: str, provider_id: str, value: int):
        """
        Before invoking a provider, the RPM 'occupying' for the provider is incremented. We will insert initialized usage data here.
        :param dimension:
        :param group:
        :param provider_id:
        :param value: For RPM, default is 1, for TPM, no default.
        :return:
        """
        key, lock = self._fetch_or_create_lock(dimension, group, provider_id)
        async with lock:
            usage = self.Usage(used=0, occupying=value)
            await self.cache.async_set_value(key, usage.serialize(), ttl=self.DEFAULT_TTL)

    async def increase_rpm_occupied(self, group: str, provider_id: str, value: int = 1):
        return await self._increase_occupied(Dimension.RPM, group, provider_id, value)

    async def increase_tpm_occupied(self, group: str, provider_id: str, value: int):
        return await self._increase_occupied(Dimension.RPM, group, provider_id, value)

    async def _update_used_usage(self, dimension: Dimension, group: str, provider_id: str, value: int):
        """
        If the call succeeds, increase the 'used' and 'occupying' for the provider.
        :param dimension:
        :param group:
        :param provider_id:
        :param value:
        :return:
        """
        key, lock = self._fetch_or_create_lock(dimension, group, provider_id)
        async with lock:
            data = await self.cache.async_get_value(key)
            if not data:
                self.logger.error(f"No usage data found for {key}")
                return
            usage = self.Usage.deserialize(data)
            usage.used += value
            usage.occupying -= value
            await self.cache.async_set_value(key, usage.serialize())

    async def update_rpm_used_usage(self, group: str, provider_id: str, value: int = 1):
        return await self._update_used_usage(Dimension.RPM, group, provider_id, value)

    async def update_tpm_used_usage(self, group: str, provider_id: str, value: int):
        return await self._update_used_usage(Dimension.TPM, group, provider_id, value)

    async def _release_occupied(self, dimension: Dimension, group: str, provider_id: str, value: int):
        """
        If the call fails, release the occupying usage.
        :param dimension:
        :param group:
        :param provider_id:
        :param value:
        :return:
        """
        key, lock = self._fetch_or_create_lock(dimension, group, provider_id)
        async with lock:
            data = await self.cache.async_get_value(key)
            if not data:
                self.logger.error(f"No usage data found for {key}")
                return
            usage = self.Usage.deserialize(data)
            usage.occupying -= value
            await self.cache.async_set_value(key, usage.serialize())

    async def release_rpm_occupied(self, group: str, provider_id: str, value: int = 1):
        return await self._release_occupied(Dimension.RPM, group, provider_id, value)

    async def release_tpm_occupied(self, group: str, provider_id: str, value: int):
        return await self._release_occupied(Dimension.TPM, group, provider_id, value)

    async def _usage_at_minute(self, dimension: Dimension, group: str, provider_id: str) -> int:
        """
        Get the usage for a provider at the current minute.
        :param dimension:
        :param group:
        :param provider_id:
        :return:
        """
        key, lock = self._fetch_or_create_lock(dimension, group, provider_id)
        async with lock:
            data = await self.cache.async_get_value(key)
            if not data:
                self.logger.debug(f"No usage data found for {key}")
                return 0
            self.logger.debug(f"Usage data found for {key}: {data}")
            usage = self.Usage.deserialize(data)
            return usage.total()

    async def rpm_usage_at_minute(self, group: str, provider_id: str):
        return await self._usage_at_minute(Dimension.RPM, group, provider_id)

    async def tpm_usage_at_minute(self, group: str, provider_id: str):
        return await self._usage_at_minute(Dimension.TPM, group, provider_id)

    @staticmethod
    def _build_rpm_tpm_key(dimension: Dimension, group: str, provider_id: str):
        ctx: RouterContext = router_context.get()
        minute = ctx.start_minute_str()
        key = f"{dimension.value}:{group}:{provider_id}:{minute}"
        return key

    def _fetch_or_create_lock(self, dimension: Dimension, group: str, provider_id: str):
        """
        Build the key, and lock for a provider's if not exists.
        :param dimension:
        :param group:
        :param provider_id:
        :return:
        """
        key = RpmTpmManager._build_rpm_tpm_key(dimension, group, provider_id)
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return key, self.locks[key]
