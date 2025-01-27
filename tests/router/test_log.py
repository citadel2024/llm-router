import sys
import json
import logging
from unittest.mock import Mock, MagicMock

import pytest

from src.router.log import ColorCodes, JsonFormatter, ColorFormatter, get_logger


@pytest.fixture
def mock_log_cfg():
    cfg = Mock()
    cfg.level = logging.DEBUG
    cfg.log_dir = "/fake/logs"
    cfg.stage = "dev"
    return cfg


class MockHandler(MagicMock):
    @property
    def level(self):
        return 0


def test_dev_env_logger_configuration(mock_log_cfg):
    logger = get_logger("test", mock_log_cfg)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert isinstance(handler.formatter, ColorFormatter)


def test_color_formatter_levels():
    formatter = ColorFormatter("%(message)s")
    levels = [
        (logging.DEBUG, ColorCodes.GREY),
        (logging.INFO, ColorCodes.BLUE),
        (logging.WARNING, ColorCodes.YELLOW),
        (logging.ERROR, ColorCodes.RED),
        (logging.CRITICAL, ColorCodes.BOLD_RED),
    ]

    for level, color in levels:
        record = logging.LogRecord(
            name="test", level=level, pathname=__file__, lineno=1, msg="test", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert formatted.startswith(color)
        assert formatted.endswith(ColorCodes.RESET)


def test_json_formatter_output():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1, msg="test message", args=(), exc_info=None
    )
    output = json.loads(formatter.format(record))
    assert output["message"] == "test message"
    assert output["level"] == "INFO"
    assert "timestamp" in output


def test_existing_logger_handling(caplog, mock_log_cfg):
    logger = logging.getLogger("test")
    existing_handler = logging.NullHandler()
    logger.addHandler(existing_handler)

    with caplog.at_level(logging.WARNING):
        get_logger("test", mock_log_cfg)

    assert "Logger already configured" in caplog.text


def test_json_formatter_exception():
    formatter = JsonFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1, msg="test error", args=(), exc_info=exc_info
    )
    output = json.loads(formatter.format(record))
    assert "stacktrace" in output
    assert "ValueError: test error" in output["stacktrace"]


def test_log_level_configuration(mock_log_cfg):
    mock_log_cfg.level = logging.WARNING
    logger = get_logger(__name__, mock_log_cfg)
    assert logger.level == logging.WARNING
