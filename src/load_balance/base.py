from abc import ABC, abstractmethod
from typing import Optional

from src.config import LogConfiguration, LoadBalancerConfig
from src.cache.base import BaseCache
from src.router.log import get_logger
from src.config.config import LLMProviderConfig
from src.model.message import ChatMessageValues


class BaseLoadBalancer(ABC):
    def __init__(
        self, lb_cache: BaseCache, module_name: str, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig
    ):
        self.lb_cache = lb_cache
        self.logger = get_logger(module_name, log_cfg)
        self.load_balancer_config = load_balancer_config

    @abstractmethod
    async def schedule_provider(
        self,
        group: str,
        healthy_providers: list[LLMProviderConfig],
        text: Optional[str] = None,
        messages: list[ChatMessageValues] = None,
    ) -> Optional[LLMProviderConfig]:
        """
        Choose a provider from the list of healthy providers based on the load balancing strategy.
        :param group:
        :param healthy_providers:
        :param text:
        :param messages:
        :return: If a provider is selected, return the provider, otherwise return None.
        """
        raise NotImplementedError
