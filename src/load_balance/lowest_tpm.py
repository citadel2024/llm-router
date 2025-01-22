import math
from datetime import datetime
from typing import Optional

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.load_balance.base import BaseLoadBalancer
from src.message import ChatMessageValues
from src.token.counter import token_counter


class LowestTPMBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig):
        """
        Load balancer that selects the provider with the lowest TPM and filters out providers that are not available in RPM.
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config)

    def schedule_provider(self, model_group: str, healthy_providers: list[LLMProviderConfig],
                          messages: list[ChatMessageValues] = None) -> Optional[LLMProviderConfig]:
        current_minute = datetime.now().strftime("%H-%M")
        cache_keys = {
            "tpm": f"{model_group}:tpm:{current_minute}",
            "rpm": f"{model_group}:rpm:{current_minute}"
        }
        usage_data = self._get_usage_data(cache_keys, healthy_providers)
        input_tokens = token_counter(messages=messages)
        self.logger.debug(f"input token: {input_tokens}")
        return self._find_optimal_provider(healthy_providers, usage_data, input_tokens)

    def _get_usage_data(self, cache_keys: dict[str, str], providers: list[LLMProviderConfig]) -> dict[str, dict]:
        tpm_dict = self.lb_cache.get_cache(cache_keys["tpm"]) or {}
        rpm_dict = self.lb_cache.get_cache(cache_keys["rpm"]) or {}
        # For tpm dict, we use zero as the default value for providers that are not in the cache
        # For rpm dict, we only check the usage if the provider is in the cache
        return {"tpm": tpm_dict, "rpm": rpm_dict}

    def _find_optimal_provider(self, providers: list[LLMProviderConfig], usage_data: dict[str, dict],
                               input_tokens: int) -> Optional[LLMProviderConfig]:
        lowest_tpm = math.inf
        optimal_provider = None
        for provider in providers:
            current_tpm = usage_data["tpm"].get(provider.id, 0)
            # If user does not have a tpm or rpm limit, we assume it is infinity
            if not self._is_model_available(
                    provider.id,
                    provider.tpm or math.inf,
                    provider.rpm or math.inf,
                    usage_data["rpm"],
                    current_tpm,
                    input_tokens
            ):
                self.logger.debug(f"Skipping provider {provider.model_id} as it is not available")
                continue
            if current_tpm < lowest_tpm:
                lowest_tpm = current_tpm
                optimal_provider = provider
        return optimal_provider

    @staticmethod
    def _is_model_available(provider_id: str, max_tpm: float, max_rpm: float, rpm_dict: dict, current_tpm: int,
                            input_tokens: int) -> bool:
        if current_tpm + input_tokens > max_tpm:
            return False
        if rpm_dict and provider_id in rpm_dict:
            if rpm_dict[provider_id] + 1 > max_rpm:
                return False
        return True
