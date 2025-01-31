import random
from typing import Optional

from src.model import ChatMessageValues
from src.config import LogConfiguration, LoadBalancerConfig
from src.cache.base import BaseCache
from src.config.config import LLMProviderConfig
from src.load_balance.base import BaseLoadBalancer


class RandomBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig):
        """
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config)

    async def schedule_provider(
        self,
        _group: str,
        healthy_providers: list[LLMProviderConfig],
        _text: Optional[str] = None,
        _messages: list[ChatMessageValues] = None,
    ) -> Optional[LLMProviderConfig]:
        if not healthy_providers:
            return None
        return random.choice(healthy_providers)
