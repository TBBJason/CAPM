"""Tests for the yfinance resilience helpers (retry, cache, rate-limit detection)."""
import time

import pytest

from yf_utils import (
    RateLimitedError,
    TTLCache,
    is_rate_limit_error,
    ttl_cache,
    with_retry,
)


def test_is_rate_limit_error_detects_message_markers():
    assert is_rate_limit_error(Exception("Too Many Requests. Rate limited."))
    assert is_rate_limit_error(RuntimeError("429 client error"))
    assert not is_rate_limit_error(ValueError("bad expiry date"))


def test_is_rate_limit_error_detects_typed_exception():
    # YFRateLimitError takes no message argument, so detection must rely on the
    # exception *type*, not its string.
    from yfinance.exceptions import YFRateLimitError
    assert is_rate_limit_error(YFRateLimitError())


def test_with_retry_returns_value_without_error():
    calls = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0)
    def ok():
        calls["n"] += 1
        return "value"

    assert ok() == "value"
    assert calls["n"] == 1  # no retries needed


def test_with_retry_recovers_after_transient_rate_limit():
    calls = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Too Many Requests. Rate limited.")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_with_retry_raises_rate_limited_after_exhausting_attempts():
    @with_retry(max_attempts=2, base_delay=0)
    def always_limited():
        raise Exception("429 Too Many Requests")

    with pytest.raises(RateLimitedError):
        always_limited()


def test_with_retry_does_not_retry_other_errors():
    calls = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0)
    def boom():
        calls["n"] += 1
        raise ValueError("invalid expiry date")

    with pytest.raises(ValueError):
        boom()
    assert calls["n"] == 1  # non-rate-limit errors are not retried


def test_ttl_cache_serves_from_cache_within_ttl():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=60)
    def fetch(x):
        calls["n"] += 1
        return x * 2

    assert fetch(3) == 6
    assert fetch(3) == 6  # served from cache
    assert calls["n"] == 1
    assert fetch(4) == 8  # different arg -> a fresh call
    assert calls["n"] == 2


def test_ttl_cache_expires():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=0.05)
    def fetch():
        calls["n"] += 1
        return "v"

    fetch()
    time.sleep(0.1)
    fetch()
    assert calls["n"] == 2  # cache entry expired, so it re-fetched


def test_ttl_cache_does_not_cache_falsy_results():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=60)
    def fetch():
        calls["n"] += 1
        return None  # a transient failure returning empty must not be pinned

    fetch()
    fetch()
    assert calls["n"] == 2


def test_ttlcache_direct_get_set():
    c = TTLCache(ttl_seconds=60)
    assert c.get("k") is None
    c.set("k", 123)
    assert c.get("k") == 123
    c.clear()
    assert c.get("k") is None
