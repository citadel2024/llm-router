import math
from datetime import datetime
from typing import Optional

from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig
from src.load_balance.base import BaseLoadBalancer
from src.message import ChatMessageValues
from src.token.counter import token_counter


class LowestTPMBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration):
        """
        Load balancer that selects the provider with the lowest TPM and filters out providers that are not available in RPM.
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg)

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

        for i in providers:
            model_id = i.id
            if model_id not in tpm_dict:
                tpm_dict[model_id] = 0

        return {"tpm": tpm_dict, "rpm": rpm_dict}

    def _find_optimal_provider(self, providers: list[LLMProviderConfig], usage_data: dict[str, dict],
                               input_tokens: int) -> Optional[LLMProviderConfig]:
        lowest_tpm = math.inf
        optimal_provider = None
        for provider in providers:
            current_tpm = usage_data["tpm"].get(provider.model_id, 0)
            if not self._is_model_available(
                    provider.model_id,
                    provider.tpm,
                    provider.rpm,
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
    def _is_model_available(model_id: str, max_tpm: float, max_rpm: float, rpm_dict: dict, current_tpm: int,
                            input_tokens: int) -> bool:
        if current_tpm + input_tokens > max_tpm:
            return False
        if rpm_dict and model_id in rpm_dict:
            if rpm_dict[model_id] + 1 > max_rpm:
                return False
        return True
