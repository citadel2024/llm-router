from enum import Enum
from typing import Literal, Optional
from dataclasses import dataclass


class LoadBalancerStrategy(Enum):
    CAPACITY_BASED_BALANCER = "capacity-based-balancer"
    LOWEST_TPM_BALANCER = "lowest-tpm-balancer"
    LATENCY_BASED_BALANCER = "latency-based-balancer"
    COST_BASED_BALANCER = "cost-based-balancer"


@dataclass
class LoadBalancerConfig:
    strategy: LoadBalancerStrategy = LoadBalancerStrategy.CAPACITY_BASED_BALANCER
    capacity_dimension: Optional[Literal["rpm", "tpm", "weight"]] = None

    def __post_init__(self):
        if self.strategy == LoadBalancerStrategy.CAPACITY_BASED_BALANCER:
            if self.capacity_dimension not in ["rpm", "tpm", "weight"]:
                raise ValueError(f"Invalid capacity dimension: {self.capacity_dimension}")
