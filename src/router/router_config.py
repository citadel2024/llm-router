from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from typing_extensions import Optional

from src.providers.base_provider import BaseLLMProvider


@dataclass
class LLMProviderConfig:
    model_name: str
    impl: BaseLLMProvider
    rpm: int
    tpm: int


@dataclass
class RouterConfig:
    llm_providers: dict[str, list[LLMProviderConfig]]

    def to_json(self, indent: Optional[int] = None):
        return json.dumps(asdict(self), default=str, indent=indent)
