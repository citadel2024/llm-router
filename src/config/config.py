from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Literal, Union, Optional

from src.load_balance.strategy import LoadBalanceStrategy
from src.providers.base_provider import BaseLLMProvider
from src.router.retry import RetryPolicy
from src.utils.hash import generate_unique_id


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
    model_id: str
    impl: BaseLLMProvider
    rpm: int
    tpm: int
    id: str = field(init=False)

    def __post_init__(self):
        self.id = generate_unique_id(self.to_json())

    def to_json(self, indent: Optional[int] = None):
        """
        Convert the object to a compact JSON string ordered by keys.
        :param indent:
        :return:
        """
        return json.dumps({
            "model_id": self.model_id,
            "impl": self.impl.__repr__(),
            "rpm": self.rpm,
            "tpm": self.tpm,
        }, default=str, indent=indent, separators=(",", ":"), sort_keys=True)


@dataclass
class RouterConfig:
    # group name must be unique, so we use dict instead of list
    llm_provider_group: dict[str, list[LLMProviderConfig]]

    log_config: LogConfiguration = field(default_factory=LogConfiguration)
    lb_strategy: LoadBalanceStrategy = LoadBalanceStrategy.CAPACITY_BASED_BALANCING
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
