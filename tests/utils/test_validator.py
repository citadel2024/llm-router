from unittest.mock import MagicMock

import pytest

from src.utils.validator import validate_integer


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
