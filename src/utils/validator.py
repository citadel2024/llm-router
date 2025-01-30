from typing import Any
from functools import wraps

from src.model import RouterInput
from src.exceptions.exceptions import InvalidInputError


def validate_completion_inputs(func):
    @wraps(func)
    async def wrapper(arg: RouterInput):
        text = arg.text
        messages = arg.messages
        if not (text or messages):
            raise InvalidInputError("Either 'text' or 'messages' must be provided.")
        return await func(arg)

    return wrapper


def validate_integer(obj: Any, filed: str):
    if hasattr(obj, filed):
        v = getattr(obj, filed)
        if isinstance(v, int) and v < 0:
            raise ValueError(f"Invalid {filed} value: {v}")
    else:
        raise ValueError(f"Object does not have {filed} attribute")
