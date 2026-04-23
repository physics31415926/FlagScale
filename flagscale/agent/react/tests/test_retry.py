"""Tests for retry_with_backoff."""

import pytest

from flagscale.agent.react.retry import retry_with_backoff


class FakeAPIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"API error {status_code}")


class TestRetryWithBackoff:
    def test_success_no_retry(self):
        calls = []
        def fn():
            calls.append(1)
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(calls) == 1

    def test_retry_on_429(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise FakeAPIError(429)
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 3

    def test_no_retry_on_400(self):
        def fn():
            raise FakeAPIError(400)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)

    def test_exhausted_retries(self):
        def fn():
            raise FakeAPIError(500)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)

    def test_non_api_error_no_retry(self):
        def fn():
            raise ValueError("bad input")
        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
