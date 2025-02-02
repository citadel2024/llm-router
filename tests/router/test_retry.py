from unittest.mock import Mock, AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryCallState

from src.config import RetryPolicy, LogConfiguration
from src.model.input import UserParams
from src.router.retry import RetryManager
from src.utils.context import RouterContext, router_context
from src.exceptions.exceptions import RateLimitError, AuthenticationError, RetryExhaustedError, NoProviderAvailableError
from src.load_balance.rpm_tpm_manager import RpmTpmManager


@pytest.fixture
def mock_wrapped_fn():
    return AsyncMock()


@pytest.fixture
def mock_rpm_tpm_manager():
    return Mock(spec=RpmTpmManager)


@pytest.fixture
def retry_manager(mock_wrapped_fn, mock_rpm_tpm_manager):
    return RetryManager(
        async_wrapped_fn=mock_wrapped_fn,
        log_cfg=LogConfiguration(),
        rpm_tpm_manager=mock_rpm_tpm_manager,
        max_attempt=3,
        max_delay=30,
        fix_wait_seconds=0.01,
        multiplier=0.01,
    )


@pytest.fixture
def retry_manager_with_policy(mock_wrapped_fn, mock_rpm_tpm_manager):
    return RetryManager(
        async_wrapped_fn=mock_wrapped_fn,
        log_cfg=LogConfiguration(),
        rpm_tpm_manager=mock_rpm_tpm_manager,
        retry_policy=RetryPolicy(RateLimitErrorRetries=2),
        fix_wait_seconds=0.01,
        multiplier=0.01,
    )


@pytest.mark.asyncio
async def test_execute_success(mock_wrapped_fn, retry_manager):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.return_value = "success"
    result = await retry_manager.execute(UserParams(model_group="test_group", text="text"))
    assert result == "success"
    mock_wrapped_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_until_success(mock_wrapped_fn, retry_manager):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.side_effect = AsyncMock(
        side_effect=[RateLimitError("rate limit"), RateLimitError("rate limit"), "success"]
    )
    result = await retry_manager.execute(UserParams(model_group="test_group", text="text"))
    assert result == "success"
    assert mock_wrapped_fn.await_count == 3


@pytest.mark.asyncio
async def test_stop_on_non_retryable_exception(mock_wrapped_fn, retry_manager):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.side_effect = AuthenticationError("auth error")
    with pytest.raises(AuthenticationError):
        await retry_manager.execute(UserParams(model_group="test_group", text="text"))
    mock_wrapped_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_attempt_exhausted(mock_wrapped_fn, retry_manager):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.side_effect = [RateLimitError("rate limit")] * 3
    with pytest.raises(RetryExhaustedError):
        await retry_manager.execute(UserParams(model_group="test_group", text="text"))
    assert mock_wrapped_fn.await_count == 3


@pytest.mark.asyncio
async def test_retry_policy_limits(mock_wrapped_fn, retry_manager_with_policy):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.side_effect = [RateLimitError("rate limit")] * 3
    retry_manager_with_policy.max_attempt = 5
    with pytest.raises(RetryExhaustedError):
        await retry_manager_with_policy.execute(UserParams(model_group="test_group", text="text"))
    assert mock_wrapped_fn.await_count == 2


@pytest.mark.asyncio
async def test_rpm_tpm_updates(mock_wrapped_fn, mock_rpm_tpm_manager, retry_manager):
    ctx = RouterContext(model_group="test_group", provider_id="test_provider", token_count=100)
    token = router_context.set(ctx)
    try:
        mock_wrapped_fn.side_effect = [RateLimitError("rate limit"), "success"]
        await retry_manager.execute(UserParams(model_group="test_group", text="text"))
        assert mock_rpm_tpm_manager.increase_rpm_occupied.call_count == 2
        assert mock_rpm_tpm_manager.release_rpm_occupied.call_count == 1
        assert mock_rpm_tpm_manager.update_rpm_used_usage.call_count == 1
    finally:
        router_context.reset(token)


@pytest.mark.asyncio
async def test_should_stop_exceptions(mock_wrapped_fn, retry_manager):
    router_context.set(RouterContext(model_group="m", provider_id="p", token_count=0))
    mock_wrapped_fn.side_effect = NoProviderAvailableError()
    with pytest.raises(NoProviderAvailableError):
        await retry_manager.execute(UserParams(model_group="test_group", text="text"))
    mock_wrapped_fn.assert_awaited_once()


@patch("src.router.retry.get_logger")
def test_log_retrying_msg(mock_get_logger, mock_wrapped_fn, mock_rpm_tpm_manager):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    retry_manager = RetryManager(
        async_wrapped_fn=mock_wrapped_fn,
        log_cfg=LogConfiguration(),
        rpm_tpm_manager=mock_rpm_tpm_manager,
        max_attempt=3,
        max_delay=30,
        fix_wait_seconds=0.01,
        multiplier=0.01,
    )
    retry_state = Mock(spec=RetryCallState)
    retry_state.attempt_number = 1
    retry_state.idle_for = 0.5
    retry_state.outcome = Mock(failed=False, result=lambda: "ok")
    retry_manager._log_retrying_msg("TEST", retry_state)
    mock_logger.info.assert_called()


def test_get_num_retries_from_policy():
    policy = RetryPolicy(RateLimitErrorRetries=3)
    assert RetryManager.get_num_retries_from_retry_policy(RateLimitError("rate limit"), policy) == 3
    assert RetryManager.get_num_retries_from_retry_policy(AuthenticationError("auth error"), policy) is None
