from unittest.mock import AsyncMock, MagicMock

import pytest

from src.model import RouterInput
from src.utils.validator import validate_integer, validate_completion_inputs
from src.exceptions.exceptions import InvalidInputError


@pytest.mark.asyncio
async def test_validate_completion_inputs_valid_text():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterInput(text="test", model_group="")
    await decorated_func(arg)

    mock_async_func.assert_awaited_once_with(arg)


@pytest.mark.asyncio
async def test_validate_completion_inputs_valid_messages():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterInput(messages=[{"role": "user"}], model_group="")
    await decorated_func(arg)

    mock_async_func.assert_awaited_once_with(arg)


@pytest.mark.asyncio
async def test_validate_completion_inputs_invalid_input():
    mock_async_func = AsyncMock()
    decorated_func = validate_completion_inputs(mock_async_func)
    arg = RouterInput(model_group="")

    with pytest.raises(InvalidInputError):
        await decorated_func(arg)


def test_validate_integer_valid():
    mock_obj = MagicMock()
    mock_obj.test_field = 5
    validate_integer(mock_obj, "test_field")


def test_validate_integer_negative():
    mock_obj = MagicMock()
    mock_obj.test_field = -1

    with pytest.raises(ValueError) as excinfo:
        validate_integer(mock_obj, "test_field")
    assert "Invalid test_field value: -1" in str(excinfo.value)


def test_validate_integer_missing_attribute():
    mock_obj = MagicMock()
    del mock_obj.test_field

    with pytest.raises(ValueError) as excinfo:
        validate_integer(mock_obj, "test_field")
    assert "Object does not have test_field attribute" in str(excinfo.value)
