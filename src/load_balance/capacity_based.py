import random
from datetime import datetime
from typing import Optional

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.load_balance.base import BaseLoadBalancer
from src.message import ChatMessageValues


class CapacityBasedBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig):
        """
        If the user has specified a weight, rpm, or tpm for a provider, this balancer will select a provider based on the specified metric.
        If no metric is specified, it will return a random provider from the list of healthy providers.
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config)

    def schedule_provider(self, group: str, healthy_providers: list[LLMProviderConfig], text: Optional[str] = None,
                          messages: list[ChatMessageValues] = None) -> Optional[LLMProviderConfig]:
        if not healthy_providers:
            return None
        filtered_providers = self._filter_over_limit_providers(group, healthy_providers)
        if not filtered_providers:
            self.logger.warning("No providers available after filtering over RPM limits.")
            return None
        dimension = self.load_balancer_config.capacity_dimension
        return self._select_weighted_provider(filtered_providers, group, dimension)

    def _filter_over_limit_providers(self, group: str, healthy_providers: list[LLMProviderConfig]) -> list[
        LLMProviderConfig]:
        current_minute = datetime.now().strftime("%H-%M")
        rpm_cache_key = f"{group}:rpm:{current_minute}"
        rpm_dict = self.lb_cache.get_cache(rpm_cache_key) or {}
        filtered_providers = filter(lambda p: rpm_dict.get(p.id, 0) + 1 <= p.rpm, healthy_providers)
        return list(filtered_providers)

    def _select_weighted_provider(self, providers: list[LLMProviderConfig], model: str,
                                  dimension: str) -> Optional[LLMProviderConfig]:
        values = [getattr(p, dimension) for p in providers]
        total = sum(values)
        if total == 0:
            self.logger.debug("All providers have 0 weight, selecting randomly.")
            return random.choice(providers)
        weights = [value / total for value in values]
        selected_index = random.choices(range(len(weights)), weights=weights)[0]
        provider = providers[selected_index]
        self.logger.debug(f"Selected provider: {provider.id} for model: {model}")
        return provider
