from typing import Any, Literal, Optional

import httpx


class RetryMixin:
    retryable: bool = False


class FallbackMixin:
    fallback: bool = False


class RouterError(Exception, RetryMixin, FallbackMixin):
    def is_retryable(self):
        return self.retryable

    # TODO not use
    def is_fallback(self):
        return self.fallback


class NoProviderAvailableError(RouterError):
    pass


class APIError(RouterError):
    message: str
    request: httpx.Request
    body: Optional[object]

    def __init__(self, message: str, request: httpx.Request, *, body: Optional[object] = None) -> None:
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body

    def __repr__(self) -> str:
        return f"Request to {self.request.url} failed: {self.message}"


class APIConnectionError(APIError):
    retryable = True

    def __init__(self, message: str = "Connection error", *, request: httpx.Request) -> None:
        super().__init__(message, request, body=None)


class APINetworkUnreachableError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        message = "Network is unreachable"
        super().__init__(message, request=request)


class APIConnectionResetError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        message = "Connection reset by peer"
        super().__init__(message, request=request)


class APIConnectionRefusedError(APIConnectionError):
    def __init__(self, *, request: httpx.Request) -> None:
        message = f"Connection refused by {request.url}"
        super().__init__(message, request=request)


class APIStatusError(APIError):
    response: httpx.Response
    status_code: int
    request_id: str | None
    router_request: Optional[Any]  # Unused

    def __init__(
        self,
        message: str,
        *,
        router_request: Optional[Any] = None,
        response: Optional[httpx.Response] = None,
        body: Optional[object] = None,
    ) -> None:
        self.router_request = router_request
        self.response = response or self._create_default_response()
        if self.response.status_code != self.status_code:
            raise ValueError(f"Expected status code {self.status_code}, got {self.response.status_code}")
        self.request_id = self.response.headers.get("x-request-id")
        super().__init__(message, self.response.request, body=body)

    def _create_default_response(self) -> httpx.Response:
        return httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(method="GET", url=""),
        )


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400


class InvalidInputError(BadRequestError):
    pass


class ContextWindowExceededError(BadRequestError):
    fallback = True


class ContentPolicyViolationError(BadRequestError):
    pass


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404


class ModelGroupNotFound(NotFoundError):
    pass


class RequestTimeoutError(APIStatusError):
    status_code: Literal[408] = 408
    retryable = True


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429
    retryable = True


class RetryExhaustedError(BadRequestError):
    last_exception: Exception
    attempt_number: int

    def __init__(
        self,
        message: str,
        last_exception: Exception,
        attempt_number: int = 0,
        *,
        router_request: Optional[Any] = None,
        response: Optional[httpx.Response] = None,
        body: Optional[object] = None,
    ):
        super().__init__(message, router_request=router_request, response=response, body=body)
        self.last_exception = last_exception
        self.attempt_number = attempt_number


class InternalServerError(APIStatusError):
    status_code: Literal[500] = 500
