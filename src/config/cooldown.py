import json
from typing import Optional
from dataclasses import field, asdict, dataclass

from src.utils.validator import validate_integer


@dataclass
class AllowedFailsPolicy:
    """
    Use this to set a custom number of allowed fails/minute before cooling down.
    If `TimeoutErrorAllowedFails = 3`, then 3 TimeoutErrorAllowedFails will be allowed before cooling down.
    """

    BadRequestErrorAllowedFails: Optional[int] = None
    AuthenticationErrorAllowedFails: Optional[int] = None
    TimeoutErrorAllowedFails: Optional[int] = None
    RateLimitErrorAllowedFails: Optional[int] = None
    ContentPolicyViolationErrorAllowedFails: Optional[int] = None
    InternalServerErrorAllowedFails: Optional[int] = None

    def serialize(self):
        return json.dumps(asdict(self))


@dataclass
class CooldownConfig:
    """
    The cooldown duration (in seconds) for the provider if num_fails exceeds `allowed_fails`. Default is 60.
    """

    cooldown_seconds: int = 60
    general_allowed_fails: int = 3
    allowed_fails_policy: AllowedFailsPolicy = field(default_factory=AllowedFailsPolicy)

    def __post_init__(self):
        for i in ["cooldown_seconds", "general_allowed_fails"]:
            validate_integer(self, i)
