from enum import Enum


class LoadBalanceStrategy(Enum):
    CAPACITY_BASED_BALANCING = "capacity-based-balancing"
    LOWEST_TPM_BALANCING = "lowest-tpm-balancing"
    LATENCY_BASED_BALANCING = "latency-based-balancing"
    COST_BASED_BALANCING = "cost-based-balancing"
