import math
from typing import Optional

from src.message import ChatMessageValues
from src.cache.base import BaseCache
from src.config.config import LogConfiguration, LLMProviderConfig, LoadBalancerConfig
from src.token.counter import TokenCounter
from src.load_balance.base import BaseLoadBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


class LowestTPMBalancer(BaseLoadBalancer):
    def __init__(self, lb_cache: BaseCache, log_cfg: LogConfiguration, load_balancer_config: LoadBalancerConfig):
        """
        Load balancer that selects the provider with the lowest TPM and filters out providers that are not available in RPM.
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config)
        self.tc = TokenCounter(log_cfg)
        self.rpm_tpm_manager = RpmTpmManager(lb_cache, log_cfg)

    async def schedule_provider(
        self,
        group: str,
        healthy_providers: list[LLMProviderConfig],
        text: Optional[str] = None,
        messages: list[ChatMessageValues] = None,
    ) -> Optional[LLMProviderConfig]:
        # Since we don't choose a model, we try to get the estimated token count from the messages
        input_tokens = self.tc.token_counter(messages=messages, text=text)
        self.logger.debug(f"input token: {input_tokens}")
        return await self._find_optimal_provider(group, healthy_providers, input_tokens)

    async def _find_optimal_provider(
        self, group: str, providers: list[LLMProviderConfig], input_tokens: int
    ) -> Optional[LLMProviderConfig]:
        lowest_tpm = math.inf
        optimal_provider = None
        for provider in providers:
            current_tpm = await self.rpm_tpm_manager.tpm_usage_at_minute(group, provider.id)
            # If user does not have a tpm or rpm limit, we assume it is infinity
            if not self._is_model_available(
                provider.tpm or math.inf,
                provider.rpm or math.inf,
                await self.rpm_tpm_manager.rpm_usage_at_minute(group, provider.id),
                current_tpm,
                input_tokens,
            ):
                self.logger.debug(f"Skipping provider {provider.model_id} as it is not available")
                continue
            if current_tpm < lowest_tpm:
                lowest_tpm = current_tpm
                optimal_provider = provider
        return optimal_provider

    @staticmethod
    def _is_model_available(max_tpm: float, max_rpm: float, rpm: int, current_tpm: int, input_tokens: int) -> bool:
        if current_tpm + input_tokens > max_tpm:
            return False
        if rpm + 1 > max_rpm:
            return False
        return True
