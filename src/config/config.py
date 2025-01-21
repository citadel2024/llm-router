from dataclasses import dataclass
from typing import Literal


@dataclass
class LogConfiguration:
    stage: Literal["dev", "prod"] = "dev"
    verbose: bool = False
    log_dir: str = "logs"

    def __post_init__(self):
        if self.stage not in ["dev", "prod"]:
            raise ValueError(f"Invalid stage value: {self.stage}")
