from __future__ import annotations

import json
from typing import Optional
from dataclasses import field, asdict, dataclass

from src.config.log import LogConfiguration
from src.utils.hash import generate_unique_id
from src.config.retry import RetryConfig
from src.config.cooldown import CooldownConfig
from src.config.fallback import FallbackConfig
from src.utils.validator import validate_integer
from src.config.load_balancer import LoadBalancerConfig, LoadBalancerStrategy
from src.router.base_provider import BaseLLMProvider


@dataclass
class LLMProviderConfig:
    id: str = field(init=False)  # We generate a unique id for each provider, and it's used to identify the provider.
    model_id: str  # e.g. qwen2.5:14b-instruct / gpt-4o-2024-11-20
    impl: BaseLLMProvider
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    weight: Optional[int] = None

    def __post_init__(self):
        # The provider can not be same.
        self.id = generate_unique_id(self.serialize())

    def serialize(self, indent: Optional[int] = None):
        """
        Convert the object to a compact JSON string ordered by keys.
        :param indent:
        :return:
        """
        return json.dumps(
            {
                "model_id": self.model_id,
                "impl": self.impl.__repr__(),
                "rpm": self.rpm,
                "tpm": self.tpm,
            },
            default=str,
            indent=indent,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass
class RouterConfig:
    llm_provider_group: dict[str, list[LLMProviderConfig]]
    log_config: LogConfiguration = field(default_factory=LogConfiguration)
    load_balancer_config: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    fallback_config: FallbackConfig = field(default_factory=FallbackConfig)
    cooldown_config: CooldownConfig = field(default_factory=CooldownConfig)
    timeout_seconds: int = 30

    def serialize(self, indent: Optional[int] = None):
        """
        Convert the object to a compact JSON string ordered by keys.
        :param indent:
        :return:
        """
        # setup separators to remove spaces
        return json.dumps(asdict(self), default=str, indent=indent, separators=(",", ":"), sort_keys=True)

    def __post_init__(self):
        if not self.llm_provider_group:
            raise ValueError("No provider group is specified.")
        for i in ["timeout_seconds"]:
            validate_integer(self, i)
        if self.load_balancer_config.strategy == LoadBalancerStrategy.CAPACITY_BASED_BALANCER:
            dimension = self.load_balancer_config.capacity_dimension
            if dimension is None:
                raise ValueError("Capacity dimension is required for capacity based balancer.")
            for providers in self.llm_provider_group.values():
                for p in providers:
                    if not hasattr(p, dimension) or getattr(p, dimension) is None:
                        raise ValueError(f"Capacity dimension {dimension} is not found.")
