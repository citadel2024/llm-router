from unittest.mock import AsyncMock

import pytest

from src.model.input import RouterParams
from src.model.validator import validate_completion_inputs
from src.exceptions.exceptions import InvalidInputError


@pytest.mark.asyncio
async def test_validate_completion_inputs_valid_text():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterParams(text="test", model_group="")
    await decorated_func(arg)

    mock_async_func.assert_awaited_once_with(arg)


@pytest.mark.asyncio
async def test_validate_completion_inputs_valid_messages():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterParams(messages=[{"role": "user"}], model_group="")
    await decorated_func(arg)

    mock_async_func.assert_awaited_once_with(arg)


@pytest.mark.asyncio
async def test_validate_completion_inputs_invalid_input():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterParams(model_group="")

    with pytest.raises(InvalidInputError):
        await decorated_func(arg)
