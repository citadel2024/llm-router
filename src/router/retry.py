from typing import Any, Callable, Optional, Awaitable

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    wait_fixed,
    wait_random,
    wait_combine,
    wait_exponential,
    retry_if_exception,
)

from src.config import RetryPolicy, LogConfiguration
from src.router.log import get_logger
from src.model.input import UserParams
from src.utils.context import RouterContext, router_context
from src.exceptions.exceptions import (
    APIError,
    RateLimitError,
    BadRequestError,
    AuthenticationError,
    InternalServerError,
    RequestTimeoutError,
    RetryExhaustedError,
    NoProviderAvailableError,
    ContentPolicyViolationError,
)
from src.load_balance.rpm_tpm_manager import RpmTpmManager


class RetryManager:
    _SHOULD_STOP_EXCEPTIONS = (NoProviderAvailableError,)

    def __init__(
        self,
        wrapped_fn: Callable[..., Awaitable],
        log_cfg: LogConfiguration,
        rpm_tpm_manager: RpmTpmManager,
        max_attempt: int = 3,
        max_delay: int = 30,
        retry_policy: Optional[RetryPolicy] = None,
        fix_wait_seconds: float = 1,
        multiplier: float = 1,
    ):
        """
        1. Default Retry (Global Shared Count)
        All exceptions inside a request share a global retry count (num_retries). Each retry (regardless of error type) deducts from the total.
        2. Retry Policy (Error-Type Isolation)
        Under retry_policy, different error types have isolated retry counts. Each error type uses its own max_retries without affecting others.
        3. Global Cap (num_retries_per_request)
        Total retries across all errors never exceed num_retries_per_request. This acts as a hard upper limit for safety.
        :param wrapped_fn:
        :param log_cfg:
        :param max_attempt:
        :param max_delay:
        :param retry_policy:
        """
        self.wrapped_fn = wrapped_fn
        self.max_attempt = max_attempt
        self.max_delay = max_delay
        self.retry_policy = retry_policy
        self.rpm_tpm_manager = rpm_tpm_manager
        self.fix_wait_seconds = fix_wait_seconds
        self.multiplier = multiplier
        self.logger = get_logger(__name__, log_cfg)

    def retry_error_callback(self, retry_state: RetryCallState):
        """
        If the user configures fallback model, go to fallback logic
        Else, raise RetryExhaustedError
        :param retry_state:
        :return:
        """
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            raise RetryExhaustedError(
                f"Model call failed after retries {retry_state.attempt_number}",
                attempt_number=retry_state.attempt_number,
                last_exception=exc,
            )

    def _log_retrying_msg(self, step: str, retry_state: RetryCallState):
        if retry_state.outcome is None:
            result = "none"
        elif retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            result = f"failed ({exception.__class__.__name__} {exception})"
        else:
            result = f"returned {retry_state.outcome.result()}"
        slept = float(round(retry_state.idle_for, 2))
        self.logger.info(f"{step}: attempt #{retry_state.attempt_number}; slept for {slept}; last result: {result}")

    async def before(self, retry_state: RetryCallState):
        self._log_retrying_msg("Before", retry_state)
        ctx: RouterContext = router_context.get()
        await self.rpm_tpm_manager.increase_rpm_occupied(ctx.model_group, ctx.provider_id)
        await self.rpm_tpm_manager.increase_tpm_occupied(ctx.model_group, ctx.provider_id, ctx.token_count)

    async def after(self, retry_state: RetryCallState):
        # If the retry succeeded, this `after` function will not be called
        # So we need to release the occupied resources here and update cost at the end of `execute` function
        self._log_retrying_msg("After", retry_state)
        ctx: RouterContext = router_context.get()
        if retry_state.outcome.failed:
            self.logger.error(f"Model call failed, {retry_state.outcome.exception()}")
            await self.rpm_tpm_manager.release_rpm_occupied(ctx.model_group, ctx.provider_id)
            await self.rpm_tpm_manager.release_tpm_occupied(ctx.model_group, ctx.provider_id, ctx.token_count)

    @staticmethod
    def should_retry(exc: BaseException) -> bool:
        return isinstance(exc, APIError) and exc.is_retryable()

    def should_stop(self, retry_state: RetryCallState) -> bool:
        # Global upper limit
        if retry_state.attempt_number >= self.max_attempt:
            return True
        # Global delay limit
        if retry_state.idle_for >= self.max_delay:
            return True
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            # If the exception is in the list of exceptions that should stop immediately
            if isinstance(exc, self._SHOULD_STOP_EXCEPTIONS):
                return True
            # Retry policy limit
            policy_max = self.get_num_retries_from_retry_policy(exc, self.retry_policy)
            if policy_max is not None:
                effective_max = min(policy_max, self.max_attempt)
                return retry_state.attempt_number >= effective_max
        return False

    async def execute(self, val: UserParams) -> Any:
        exponential_backoff = wait_combine(wait_exponential(multiplier=self.multiplier, max=10), wait_random(0, 1))
        fixed_wait = wait_fixed(self.fix_wait_seconds)

        def custom_wait(retry_state: RetryCallState) -> float:
            if retry_state.outcome and retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                if isinstance(exc, RateLimitError):
                    return exponential_backoff(retry_state)
            return fixed_wait(retry_state)

        retryer = AsyncRetrying(
            wait=custom_wait,
            stop=self.should_stop,
            retry=retry_if_exception(self.should_retry),
            before=self.before,
            after=self.after,
            reraise=True,
            retry_error_callback=self.retry_error_callback,
        )
        result = await retryer(self.wrapped_fn, val)
        # update cost if succeed
        ctx: RouterContext = router_context.get()
        self.logger.debug(f"Model call succeeded")
        await self.rpm_tpm_manager.update_rpm_used_usage(ctx.model_group, ctx.provider_id)
        await self.rpm_tpm_manager.update_tpm_used_usage(ctx.model_group, ctx.provider_id, ctx.token_count)
        return result

    @staticmethod
    def get_num_retries_from_retry_policy(
        exception: Exception,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Optional[int]:
        if not retry_policy:
            return None
        exception_mapping = {
            BadRequestError: "BadRequestErrorRetries",
            AuthenticationError: "AuthenticationErrorRetries",
            RequestTimeoutError: "TimeoutErrorRetries",
            RateLimitError: "RateLimitErrorRetries",
            ContentPolicyViolationError: "ContentPolicyViolationErrorRetries",
            InternalServerError: "InternalServerErrorRetries",
        }
        for exc_type, policy_attr in exception_mapping.items():
            if isinstance(exception, exc_type):
                return getattr(retry_policy, policy_attr, None)
        return None
