import httpx
import pytest

from src.exceptions.exceptions import (
    APIError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    BadRequestError,
    ModelGroupNotFound,
    RetryExhaustedError,
    APIConnectionRefusedError,
    APINetworkUnreachableError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
)


def test_api_network_unreachable_error(mocker):
    mock_request = mocker.Mock(spec=httpx.Request)
    error = APINetworkUnreachableError(mock_request)
    assert error.message == "Network is unreachable"
    assert error.request is mock_request
    assert error.is_retryable()


def test_api_connection_refused_error(mocker):
    mock_request = mocker.Mock(spec=httpx.Request)
    mock_request.url = "http://failed.url"
    error = APIConnectionRefusedError(request=mock_request)
    assert error.message == "Connection refused by http://failed.url"
    assert error.request is mock_request
    assert error.is_retryable()


def test_api_status_error_default_response():
    class TestError(APIStatusError):
        status_code = 999

    error = TestError("test")
    assert error.response.status_code == 999
    assert isinstance(error.response.request, httpx.Request)


def test_bad_request_error_validation(mocker):
    mock_response = mocker.Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.headers = {}
    error = BadRequestError("message", response=mock_response)
    assert error.status_code == 400
    assert not error.is_retryable()
    assert not error.is_fallback()


def test_context_window_error_validation(mocker):
    mock_response = mocker.Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.headers = {}
    error = ContextWindowExceededError("message", response=mock_response)
    assert error.status_code == 400
    assert not error.is_retryable()
    assert error.is_fallback()


def test_invalid_status_code_validation():
    with pytest.raises(ValueError):
        BadRequestError("message", response=httpx.Response(500))


def test_retry_exhausted_error_attributes():
    mock_exception = Exception("mock error")
    error = RetryExhaustedError(
        "exhausted",
        mock_exception,
        retry_count=3,
        response=httpx.Response(400, request=httpx.Request("GET", "http://test.url")),
    )
    assert error.last_exception is mock_exception
    assert error.retry_count == 3
    assert not error.is_retryable()
    assert not error.is_fallback()


def test_content_policy_error_inheritance():
    assert issubclass(ContentPolicyViolationError, BadRequestError)


def test_model_group_notfound_inheritance():
    assert issubclass(ModelGroupNotFound, NotFoundError)


def test_api_error_repr():
    mock_request = httpx.Request("GET", "http://error.url")
    error = APIError("test error", mock_request)
    assert "http://error.url" in repr(error)


@pytest.mark.asyncio
async def test_async_request_handling(mocker):
    mock_client = mocker.AsyncMock(spec=httpx.AsyncClient)
    mock_response = mocker.Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_client.get.return_value = mock_response

    async def make_request():
        response = await mock_client.get("http://test.url")
        if response.status_code == 429:
            raise RateLimitError("limit exceeded", response=response)

    with pytest.raises(RateLimitError) as exc_info:
        await make_request()

    assert exc_info.value.status_code == 429
    assert "limit exceeded" in str(exc_info.value)
