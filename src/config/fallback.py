from typing import Optional
from dataclasses import field, dataclass


@dataclass
class FallbackConfig:
    # fallback model group to another, {"GPT-4": ["GPT-3.5"]}
    degraded_map: dict[str, list[str]] = field(default_factory=dict)
    allow_fallback: Optional[bool] = None
