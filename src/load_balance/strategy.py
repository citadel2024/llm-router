from enum import Enum


class LoadBalancerStrategy(Enum):
    CAPACITY_BASED_BALANCER = "capacity-based-balancer"
    LOWEST_TPM_BALANCER = "lowest-tpm-balancer"
    LATENCY_BASED_BALANCER = "latency-based-balancer"
    COST_BASED_BALANCER = "cost-based-balancer"
