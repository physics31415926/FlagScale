"""Retry with exponential backoff for LLM API calls."""

import logging
import time

from typing import Callable, Optional

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)

RETRYABLE_EXCEPTION_NAMES = (
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    "APIConnectionError",
    "APITimeoutError",
)

CONTEXT_LIMIT_KEYWORDS = (
    "context length",
    "context window",
    "maximum context",
    "token limit",
    "too many tokens",
    "max_tokens",
    "prompt is too long",
    "input is too long",
    "request too large",
)


def _is_context_limit_error(exc: Exception) -> bool:
    """Check if a 400 error is specifically about context/token limits."""
    msg = str(exc).lower()
    return any(kw in msg for kw in CONTEXT_LIMIT_KEYWORDS)


def retry_with_backoff(
    fn,
    max_retries=3,
    base_delay=1.0,
    on_context_overflow: Optional[Callable[[], bool]] = None,
):
    """Call fn(), retrying on transient API errors with exponential backoff.

    Args:
        fn: The function to call.
        max_retries: Maximum number of retries.
        base_delay: Base delay in seconds for exponential backoff.
        on_context_overflow: Callback for 400 context-limit errors. Called with
            no arguments; should compact the context and return True if retry
            is safe, False otherwise. Only invoked once per call.
    """
    last_exc = None
    context_recovery_attempted = False
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                raise

            status = _extract_status(e)

            if (status == 400 and not context_recovery_attempted
                    and on_context_overflow and _is_context_limit_error(e)):
                context_recovery_attempted = True
                logger.warning("Context limit exceeded (400), attempting recovery...")
                try:
                    if on_context_overflow():
                        logger.info("Context compacted, retrying LLM call...")
                        continue
                except Exception as recovery_err:
                    logger.error("Context recovery failed: %s", recovery_err)
                raise

            if status and status in RETRYABLE_STATUS_CODES:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (status=%s), retrying in %.1fs (%d/%d): %s",
                    status, delay, attempt + 1, max_retries, e,
                )
                time.sleep(delay)
                continue

            if _is_retryable_exception(e):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (%s), retrying in %.1fs (%d/%d): %s",
                    type(e).__name__, delay, attempt + 1, max_retries, e,
                )
                time.sleep(delay)
                continue

            raise
    raise last_exc


def _extract_status(exc):
    """Try to extract HTTP status code from common SDK exceptions."""
    for attr in ("status_code", "status", "http_status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_retryable_exception(exc):
    """Check if exception type matches known retryable network errors."""
    for cls in type(exc).__mro__:
        if cls.__name__ in RETRYABLE_EXCEPTION_NAMES:
            return True
    return False
