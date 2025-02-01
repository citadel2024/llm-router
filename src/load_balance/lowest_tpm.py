import math
from typing import Optional

from src.model import ChatMessageValues
from src.config import LogConfiguration, LoadBalancerConfig
from src.cache.base import BaseCache
from src.config.config import LLMProviderConfig
from src.utils.context import RouterContext, router_context
from src.load_balance.base import BaseLoadBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


class LowestTPMBalancer(BaseLoadBalancer):
    def __init__(
        self,
        lb_cache: BaseCache,
        log_cfg: LogConfiguration,
        load_balancer_config: LoadBalancerConfig,
        rpm_tpm_manager: RpmTpmManager,
    ):
        """
        Load balancer that selects the provider with the lowest TPM and filters out providers that are not available in RPM.
        :param lb_cache:
        :param log_cfg:
        """
        super().__init__(lb_cache, __name__, log_cfg, load_balancer_config, rpm_tpm_manager)

    async def schedule_provider(
        self,
        group: str,
        healthy_providers: list[LLMProviderConfig],
        text: Optional[str] = None,  # noqa
        messages: list[ChatMessageValues] = None,  # noqa
    ) -> Optional[LLMProviderConfig]:
        ctx: RouterContext = router_context.get()
        # Since we don't choose a model, we try to get the estimated token count from the messages
        self.logger.debug(f"input token: {ctx.token_count}")
        return await self._find_optimal_provider(group, healthy_providers, ctx.token_count)

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
