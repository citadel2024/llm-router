from __future__ import annotations

import json
import logging
from typing import Union, Literal, Optional
from dataclasses import field, asdict, dataclass

from src.utils.hash import generate_unique_id
from src.router.retry import RetryPolicy
from src.load_balance.strategy import LoadBalancerStrategy
from src.providers.base_provider import BaseLLMProvider


@dataclass
class LogConfiguration:
    # In dev stage, we use colored logs.
    # In prod stage, we use json formatted logs.
    stage: Literal["dev", "prod"] = "dev"
    level: Union[logging.FATAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG] = logging.DEBUG
    log_dir: str = "logs"

    def __post_init__(self):
        if self.stage not in ["dev", "prod"]:
            raise ValueError(f"Invalid stage value: {self.stage}")


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
        self.id = generate_unique_id(self.to_json())

    def to_json(self, indent: Optional[int] = None):
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
class LoadBalancerConfig:
    strategy: LoadBalancerStrategy = LoadBalancerStrategy.CAPACITY_BASED_BALANCER
    capacity_dimension: Optional[Literal["rpm", "tpm", "weight"]] = None

    def __post_init__(self):
        if self.strategy == LoadBalancerStrategy.CAPACITY_BASED_BALANCER:
            if self.capacity_dimension not in ["rpm", "tpm", "weight"]:
                raise ValueError(f"Invalid capacity dimension: {self.capacity_dimension}")


@dataclass
class RouterConfig:
    # group name must be unique, so we use dict instead of list
    llm_provider_group: dict[str, list[LLMProviderConfig]]

    log_config: LogConfiguration = field(default_factory=LogConfiguration)
    load_balancer_config: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    # The cooldown duration (in seconds) for the provider if num_fails exceeds `allowed_fails`. Default is 60.
    cooldown_seconds: int = 60
    num_retries: int = 3
    retry_policy: Optional[RetryPolicy] = None
    timeout_seconds: int = 30

    def to_json(self, indent: Optional[int] = None):
        """
        Convert the object to a compact JSON string ordered by keys.
        :param indent:
        :return:
        """
        return json.dumps(asdict(self), default=str, indent=indent, separators=(",", ":"), sort_keys=True)

    def __post_init__(self):
        if not self.llm_provider_group:
            raise ValueError("No provider group is specified.")
        for i in ["cooldown_seconds", "num_retries", "timeout_seconds"]:
            self.validate_integer(i)
        if self.load_balancer_config.strategy == LoadBalancerStrategy.CAPACITY_BASED_BALANCER:
            dimension = self.load_balancer_config.capacity_dimension
            if dimension is None:
                raise ValueError("Capacity dimension is required for capacity based balancer.")
            for providers in self.llm_provider_group.values():
                for p in providers:
                    if not hasattr(p, dimension) or getattr(p, dimension) is None:
                        raise ValueError(f"Capacity dimension {dimension} is not found.")

    def validate_integer(self, filed: str):
        if getattr(self, filed) < 0:
            raise ValueError(f"Invalid {filed} value: {getattr(self, filed)}")
