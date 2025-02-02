from copy import deepcopy
from typing import cast

from src.config import RetryConfig, FallbackConfig, LoadBalancerStrategy
from src.router.log import get_logger
from src.model.input import UserParams, RouterParams
from src.cache.memory import MemoryCache
from src.load_balance import RandomBalancer, ProviderStatusManager
from src.router.retry import RetryManager
from src.config.config import RouterConfig
from src.token.counter import TokenCounter
from src.utils.context import RouterContext, router_context
from src.exceptions.exceptions import SHOULD_FALLBACK_EXCEPTIONS, NoProviderAvailableError
from src.load_balance.lowest_tpm import LowestTPMBalancer
from src.load_balance.capacity_based import CapacityBasedBalancer
from src.load_balance.rpm_tpm_manager import RpmTpmManager


class Router:
    def __init__(self, cfg: RouterConfig):
        """ """
        self.log_cfg = cfg.log_config
        self.load_balancer_config = cfg.load_balancer_config
        self.retry_config = cfg.retry_config
        self.fallback_config = cfg.fallback_config
        self.cooldown_config = cfg.cooldown_config

        self.cache = MemoryCache(cfg.log_config)
        self.logger = get_logger(__name__, self.log_cfg)
        self.provider_status_manager = ProviderStatusManager(
            cfg.log_config, cfg.llm_provider_group, cooldown_config=cfg.cooldown_config, cache=self.cache
        )
        self.rpm_tpm_manager = RpmTpmManager(self.cache, self.log_cfg)
        self.load_balancer = self.routing_strategy_init(strategy=cfg.load_balancer_config.strategy)
        self.tc = TokenCounter(cfg.log_config)

    def routing_strategy_init(self, strategy: LoadBalancerStrategy):
        self.logger.info(f"Routing strategy: {strategy}")
        strategy_config = {
            LoadBalancerStrategy.LOWEST_TPM_BALANCER: LowestTPMBalancer,
            LoadBalancerStrategy.CAPACITY_BASED_BALANCER: CapacityBasedBalancer,
        }
        config = strategy_config.get(strategy, RandomBalancer)
        return config(
            lb_cache=self.cache,
            log_cfg=self.log_cfg,
            load_balancer_config=self.load_balancer_config,
            rpm_tpm_manager=self.rpm_tpm_manager,
        )

    def normalize_input(self, arg: RouterParams):
        """
        Normalize the input to the router
        :param arg:
        :return:
        """
        new_arg = deepcopy(arg)
        if arg.retry_config is None:
            new_arg.retry_config = self.retry_config
        if arg.fallback_config is None:
            new_arg.fallback_config = self.fallback_config
        self.logger.info(f"Normalized input: before {arg} after {new_arg}")
        return new_arg

    async def async_completion(
        self,
        arg: RouterParams,
    ):
        # create context for each request
        router_context.set(
            RouterContext(
                model_group=arg.model_group,
                token_count=self.tc.token_counter(messages=arg.messages, text=arg.text),
            )
        )
        new_arg = self.normalize_input(arg)
        try:

            async def run(*args, **kwargs):
                healthy_providers = await self.provider_status_manager.get_available_providers(new_arg.model_group)
                provider = await self.load_balancer.schedule_provider(
                    new_arg.model_group, healthy_providers, new_arg.text, new_arg.messages
                )
                if not provider:
                    raise NoProviderAvailableError("No provider available")
                ctx: RouterContext = router_context.get()
                # update current model group and provider_id, used in retry manager to update usage.
                ctx.update_model_group(arg.model_group)
                ctx.update_provider_id(provider.id)
                ctx.update_start_time()
                return await provider.impl.completion(*args, **kwargs)

            retryer = RetryManager(
                run,
                log_cfg=self.log_cfg,
                max_attempt=new_arg.retry_config.max_attempt,
                retry_policy=new_arg.retry_config.retry_policy,
                rpm_tpm_manager=self.rpm_tpm_manager,
            )
            result = await retryer.execute(cast(UserParams, new_arg))
            # handle result
            self.logger.info(f"Completion result: {result}")
            return result
        except SHOULD_FALLBACK_EXCEPTIONS as e:
            # try to fallback
            self.logger.warning(f"Should fallback: {e}")
            return await self._trigger_fallback(new_arg, e)
        except Exception as e:
            self.logger.error(f"Error in completion: {e}")
            raise e

    async def _trigger_fallback(self, arg: RouterParams, e: SHOULD_FALLBACK_EXCEPTIONS):
        """
        Fallback allows the user to specify a list of models to try if the primary model fails.
        We don't apply retry on fallback models, since we may have already retried the primary model.
        :param arg:
        :return:
        """
        if (
            not arg.fallback_config.allow_fallback
            or not self.fallback_config.degraded_map
            or arg.model_group not in self.fallback_config.degraded_map
        ):
            self.logger.info("No fallback model specified")
            raise e
        new_arg = deepcopy(arg)
        for i, fallback_group in enumerate(self.fallback_config.degraded_map[arg.model_group]):
            try:
                self.logger.info(f"Trying fallback model: {fallback_group}")
                new_arg.model_group = fallback_group
                new_arg.retry_config = RetryConfig(max_attempt=1)
                new_arg.fallback_config = FallbackConfig(allow_fallback=False)
                return await self.async_completion(new_arg)
            except Exception as e:
                if i == len(self.fallback_config.degraded_map[arg.model_group]) - 1:
                    raise e
                continue
