from functools import wraps

from src.model.input import RouterParams
from src.exceptions.exceptions import InvalidInputError


def validate_completion_inputs(func):
    @wraps(func)
    async def wrapper(arg: RouterParams):
        text = arg.text
        messages = arg.messages
        if not (text or messages):
            raise InvalidInputError("Either 'text' or 'messages' must be provided.")
        return await func(arg)

    return wrapper
