import json
import time
import asyncio
from datetime import datetime
from dataclasses import asdict, dataclass

from src.config import LogConfiguration
from src.cache.base import BaseCache
from src.router.log import get_logger
from src.config.config import LLMProviderConfig
from src.config.cooldown import CooldownConfig
from src.exceptions.exceptions import (
    NotFoundError,
    APIStatusError,
    RateLimitError,
    BadRequestError,
    ModelGroupNotFound,
    AuthenticationError,
    RequestTimeoutError,
    ContentPolicyViolationError,
)

DEFAULT_CACHE_EXPIRED_SECONDS = 60 * 60
CLIENT_ERROR_MIN_STATUS = 400
CLIENT_ERROR_MAX_STATUS = 500


@dataclass
class CooldownState:
    exception: str
    timestamp: float
    cooldown_seconds: float

    def is_expired(self) -> bool:
        return time.time() > self.timestamp + self.cooldown_seconds

    def serialize(self):
        return json.dumps(asdict(self))

    @classmethod
    def deserialize(cls, data):
        return cls(**json.loads(data))


class ProviderStatusManager:
    _CRITICAL_EXCEPTIONS = (RateLimitError, AuthenticationError, NotFoundError)
    _TEMPORARY_EXCEPTIONS = (RequestTimeoutError,)

    def __init__(
        self,
        log_cfg: LogConfiguration,
        provider_groups: dict[str, list[LLMProviderConfig]],
        cooldown_config: CooldownConfig,
        cache: BaseCache,
    ):
        self.cache = cache
        self.logger = get_logger(__name__, log_cfg)
        self.provider_groups = provider_groups
        self.allowed_fails_policy = cooldown_config.allowed_fails_policy
        self.general_allowed_fails = cooldown_config.general_allowed_fails
        self.cooldown_seconds = cooldown_config.cooldown_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_available_providers(self, model_group):
        """
        Get the available providers for the model group:
        1. Get the healthy providers in the model group
        2. Filter out the providers that are in cooldown
        :param model_group:
        :return:
        """
        healthy_providers = self._get_healthy_providers(model_group)
        r = []
        for p in healthy_providers:
            if not await self._is_in_cooldown(p.id):
                r.append(p)
        return r

    async def try_add_cooldown(self, provider_id: str, exception: APIStatusError):
        """
        Try to add a provider to the cooldown list based on the exception type and the allowed fails policy.
        :param provider_id:
        :param exception:
        :return:
        """
        if await self._should_cooldown(provider_id, exception):
            await self._add_cooldown(exception=exception.__class__.__name__, provider_id=provider_id)

    def _get_healthy_providers(self, model_group: str):
        """
        Maybe we need to add automatically health check.
        :param model_group:
        :return:
        """
        healthy = self.provider_groups.get(model_group)
        if not healthy:
            raise ModelGroupNotFound("Model group not found", model_group)
        self.logger.info(f"Healthy providers for group {model_group}: {healthy}")
        return healthy

    async def _add_cooldown(self, exception: str, provider_id: str):
        """
        Add a provider to the cooldown, we keep the data more than the cooldown time,
        so that we can monitor the cooldown metrics later.
        :param exception:
        :param provider_id:
        :return:
        """
        key = self._build_cooldown_key(provider_id)
        await self.cache.async_set_value(
            key,
            CooldownState(
                exception=exception,
                timestamp=time.time(),
                cooldown_seconds=self.cooldown_seconds,
            ).serialize(),
            ttl=DEFAULT_CACHE_EXPIRED_SECONDS,
        )
        self.logger.info(f"Provider {provider_id} added to cooldown due to '{exception}'")

    async def _is_in_cooldown(self, provider_id: str) -> bool:
        """
        Check if the provider is in cooldown. We don't use the `ttl` of the record to check it,
        we need to calculate the `timestamp` and `cooldown_seconds` to determine if the provider is in cooldown.
        :param provider_id:
        :return:
        """
        # Get the expiration time for the provider
        key = self._build_cooldown_key(provider_id)
        data = await self.cache.async_get_value(key)
        if data is None:
            return False
        cooldown_state = CooldownState.deserialize(data)
        return not cooldown_state.is_expired()

    async def _should_cooldown(self, provider_id: str, original_exception: APIStatusError) -> bool:
        """
        Check if the provider should be put in cooldown based on:
        1. Exception type.
        2. The allowed fails' policy.
        Update the failed calls cache if the provider should not be put in cooldown.
        :param provider_id:
        :param original_exception:
        :return:
        """
        if self._is_cooldown_required_for_exception(original_exception):
            return True

        allowed_fails = self._get_allowed_fails_from_policy(exception=original_exception)
        key = self._build_fail_calls_key(provider_id)
        lock = self._fetch_or_create_lock(key)
        async with lock:
            current_fails = await self.cache.async_get_value(key) or 0
            updated_fails = current_fails + 1

            if updated_fails > allowed_fails:
                return True
            await self.cache.async_set_value(key, updated_fails, ttl=DEFAULT_CACHE_EXPIRED_SECONDS)
            return False

    def _get_allowed_fails_from_policy(self, exception: APIStatusError):
        """
        Get the allowed failures for the provider
        :param exception:
        :return:
        """
        exception_map = {
            BadRequestError: self.allowed_fails_policy.BadRequestErrorAllowedFails,
            AuthenticationError: self.allowed_fails_policy.AuthenticationErrorAllowedFails,
            RequestTimeoutError: self.allowed_fails_policy.TimeoutErrorAllowedFails,
            RateLimitError: self.allowed_fails_policy.RateLimitErrorAllowedFails,
            ContentPolicyViolationError: self.allowed_fails_policy.ContentPolicyViolationErrorAllowedFails,
        }
        # Get the corresponding allowed fails if exists
        for exc_type, allowed_fails in exception_map.items():
            if isinstance(exception, exc_type) and allowed_fails is not None:
                return allowed_fails
        return self.general_allowed_fails

    @staticmethod
    def _build_fail_calls_key(provider_id: str) -> str:
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        return f"failed_calls:{provider_id}:{current_minute}"

    @staticmethod
    def _build_cooldown_key(provider_id: str) -> str:
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        return f"cooldown:{provider_id}:{current_minute}"

    def _is_critical_failure(self, exception: APIStatusError) -> bool:
        return isinstance(exception, self._CRITICAL_EXCEPTIONS)

    def _is_temporary_failure(self, exception: APIStatusError) -> bool:
        return isinstance(exception, self._TEMPORARY_EXCEPTIONS)

    def _is_cooldown_required_for_exception(self, exception: APIStatusError) -> bool:
        """
        Return True if exception is in critical exceptions, or http code >= 500
        :param exception:
        :return:
        """
        if self._is_critical_failure(exception):
            return True
        if self._is_temporary_failure(exception):
            return False
        exception_status = exception.status_code
        if CLIENT_ERROR_MIN_STATUS <= exception_status < CLIENT_ERROR_MAX_STATUS:
            return False
        return True

    def _fetch_or_create_lock(self, key: str):
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
