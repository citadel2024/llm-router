from abc import ABC, abstractmethod
from typing import Optional

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LoadBalancerConfig
from src.message.message import ChatMessageValues
from src.router.log import get_logger


class BaseLoadBalancer(ABC):
    def __init__(self, lb_cache: BaseCache, module_name: str, log_cfg: LogConfiguration,
                 load_balancer_config: LoadBalancerConfig):
        self.lb_cache = lb_cache
        self.logger = get_logger(module_name, log_cfg)
        self.load_balancer_config = load_balancer_config

    @abstractmethod
    def schedule_provider(self, model_group: str, healthy_providers: list[dict],
                          messages: list[ChatMessageValues] = None) -> Optional[dict]:
        raise NotImplementedError

