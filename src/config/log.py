import logging
from typing import Union, Literal
from dataclasses import dataclass


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
