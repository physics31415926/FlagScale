"""Retry with exponential backoff for LLM API calls."""

import logging
import time

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)


def retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    """Call fn(), retrying on transient API errors with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            status = _extract_status(e)
            if status and status in RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (status=%s), retrying in %.1fs (%d/%d): %s",
                    status, delay, attempt + 1, max_retries, e,
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
