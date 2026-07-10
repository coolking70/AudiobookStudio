import pytest

from tools.openai_retry import call_with_404_backoff


class ProviderError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_404_retry_uses_exponential_backoff_then_succeeds():
    attempts = 0
    delays = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError(404)
        return "ok"

    assert call_with_404_backoff(operation, retries=3, base_delay=0.25, sleep=delays.append) == "ok"
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_non_404_error_is_not_retried():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise ProviderError(401)

    with pytest.raises(ProviderError):
        call_with_404_backoff(operation, sleep=lambda _delay: None)
    assert attempts == 1
