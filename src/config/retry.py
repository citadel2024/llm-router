import json
from enum import Enum
from typing import Optional
from dataclasses import asdict, dataclass


@dataclass
class RetryPolicy:
    BadRequestErrorRetries: Optional[int] = None
    AuthenticationErrorRetries: Optional[int] = None
    TimeoutErrorRetries: Optional[int] = None
    RateLimitErrorRetries: Optional[int] = None
    ContentPolicyViolationErrorRetries: Optional[int] = None
    InternalServerErrorRetries: Optional[int] = None

    def serialize(self):
        return json.dumps(asdict(self))


class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff_retry"
    CONSTANT_INTERVAL = "constant_retry"


@dataclass
class RetryConfig:
    max_attempt: int = 3
    retry_policy: Optional[RetryPolicy] = None
