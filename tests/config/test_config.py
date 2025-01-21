import pytest

from src.config import LogConfiguration


def test_default_values():
    config = LogConfiguration()
    assert config.stage == "dev"
    assert config.verbose is False
    assert config.log_dir == "logs"


def test_custom_values():
    config = LogConfiguration(stage="prod", verbose=True, log_dir="custom_logs")
    assert config.stage == "prod"
    assert config.verbose is True
    assert config.log_dir == "custom_logs"


@pytest.mark.parametrize("stage", ["dev", "prod"])
def test_stage_param(stage):
    config = LogConfiguration(stage=stage)
    assert config.stage == stage


def test_invalid_stage():
    with pytest.raises(ValueError):
        LogConfiguration(stage="invalid")


def test_verbose():
    config = LogConfiguration(verbose=True)
    assert config.verbose is True
    config = LogConfiguration(verbose=False)
    assert config.verbose is False
