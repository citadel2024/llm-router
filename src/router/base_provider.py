from abc import ABC, abstractmethod
from typing import Any

from src.model.input import UserParams


class BaseLLMProvider(ABC):
    def __repr__(self):
        class_name = self.__class__.__name__
        module_name = self.__module__
        return f"<{module_name}.{class_name}>"

    @abstractmethod
    async def completion(self, param: UserParams) -> Any:
        pass
