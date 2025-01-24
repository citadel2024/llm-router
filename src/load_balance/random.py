import random
from typing import Optional

from src.cache.base import BaseCache
from src.config import LogConfiguration
from src.config.config import LoadBalancerConfig, LLMProviderConfig
from src.load_balance.base import BaseLoadBalancer
from src.message.message import ChatMessageValues


class RandomBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig):
        """
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config)

    def schedule_provider(self, group: str, healthy_providers: list[LLMProviderConfig], text: Optional[str] = None,
                          messages: list[ChatMessageValues] = None) -> Optional[LLMProviderConfig]:
        if not healthy_providers:
            return None
        return random.choice(healthy_providers)
