from enum import Enum
from typing import Optional
from dataclasses import dataclass


@dataclass
class RetryPolicy:
    BadRequestErrorRetries: Optional[int] = None
    AuthenticationErrorRetries: Optional[int] = None
    TimeoutErrorRetries: Optional[int] = None
    RateLimitErrorRetries: Optional[int] = None
    ContentPolicyViolationErrorRetries: Optional[int] = None
    InternalServerErrorRetries: Optional[int] = None


class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff_retry"
    CONSTANT_INTERVAL = "constant_retry"


@dataclass
class RetryConfig:
    fallbacks: list[str] = None
    context_window_fallbacks: list[str] = None
    content_policy_fallbacks: list[str] = None
    retry_policy: dict = None
    model_group_retry_policy: dict = None
