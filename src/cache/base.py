from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    def __init__(self, default_ttl: int):
        self.default_ttl = default_ttl

    @abstractmethod
    async def async_set_value(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def async_get_value(self, key: str, **kwargs) -> Any:
        raise NotImplementedError
