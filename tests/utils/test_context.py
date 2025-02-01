import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.utils.context import RouterContext, router_context


@pytest.fixture
def mock_datetime(monkeypatch):
    mock_dt = Mock()
    mock_dt.now.return_value = datetime(2023, 1, 1, 12, 0)
    monkeypatch.setattr("src.utils.context.datetime", mock_dt)
    return mock_dt


def test_router_context_init():
    ctx = RouterContext(model_group="test-group", token_count=100)
    assert ctx.model_group == "test-group"
    assert ctx.token_count == 100
    assert ctx.provider_id is None
    assert isinstance(ctx.request_id, str)


def test_start_minute_str():
    with patch("src.utils.context.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2023, 1, 1, 12, 0)
        ctx = RouterContext(model_group="test-group", token_count=100)
        assert ctx.start_minute_str() == "202301011200"


def test_serialize():
    ctx = RouterContext(model_group="test-group", token_count=100)
    serialized = ctx.serialize()
    data = json.loads(serialized)
    assert len(data["request_id"].split("-")) == 5
    assert data["model_group"] == "test-group"
    assert data["token_count"] == 100
    assert data["provider_id"] is None
    assert data["start_time"] is not None


def test_update_start_time(mock_datetime):
    ctx = RouterContext(model_group="test-group", token_count=100)
    mock_datetime.now.return_value = datetime(2023, 1, 1, 13, 0)
    ctx.update_start_time()
    assert ctx.start_time == datetime(2023, 1, 1, 13, 0)


def test_update_model_group():
    ctx = RouterContext(model_group="test-group", token_count=100)
    ctx.update_model_group("new-group")
    assert ctx.model_group == "new-group"


def test_update_provider_id():
    ctx = RouterContext(model_group="test-group", token_count=100)
    ctx.update_provider_id("new-provider")
    assert ctx.provider_id == "new-provider"


@pytest.fixture
def mock_router_context():
    ctx = RouterContext(model_group="test-group", token_count=100)
    token = router_context.set(ctx)
    yield ctx
    router_context.reset(token)


def test_router_context_var(mock_router_context):  # noqa
    ctx = router_context.get()
    assert ctx.model_group == "test-group"
    assert ctx.token_count == 100
