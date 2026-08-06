"""Resilience helpers for yfinance calls.

Yahoo Finance aggressively rate-limits by IP, so shared hosting (Render, Heroku,
etc.) frequently gets "Too Many Requests" (HTTP 429) responses. This module
provides two mitigations, applied to every network-facing data function:

* ``ttl_cache`` -- short-lived in-process caching so repeated requests for the
  same ticker/expiry don't hit Yahoo again, cutting request volume (the main
  lever for staying under the limit).
* ``with_retry`` -- exponential-backoff retries that transparently ride out
  transient rate-limit blips before giving up.

When retries are exhausted a :class:`RateLimitedError` is raised so callers can
translate it into a clean 429 with a helpful message.
"""
import functools
import threading
import time


class RateLimitedError(Exception):
    """Raised when Yahoo Finance rate-limits us after exhausting retries."""


# Substrings Yahoo/yfinance use in rate-limit messages (matched case-insensitively).
_RATE_LIMIT_MARKERS = ("too many requests", "rate limit", "429")


def is_rate_limit_error(exc):
    """Return True if ``exc`` looks like a Yahoo Finance rate-limit error."""
    # Prefer the typed exception when the installed yfinance exposes it.
    try:
        from yfinance.exceptions import YFRateLimitError
        if isinstance(exc, YFRateLimitError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def with_retry(max_attempts=3, base_delay=0.8, backoff=2.0):
    """Retry a function on rate-limit errors with exponential backoff.

    Only rate-limit errors are retried; any other exception propagates
    immediately. After the final attempt a :class:`RateLimitedError` is raised.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    if not is_rate_limit_error(exc):
                        raise
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    time.sleep(delay)
                    delay *= backoff
            raise RateLimitedError(
                "Yahoo Finance is rate-limiting requests. Please wait a minute "
                "and try again."
            ) from last_exc
        return wrapper
    return decorator


class TTLCache:
    """A minimal thread-safe time-to-live cache."""

    def __init__(self, ttl_seconds):
        self.ttl = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            timestamp, value = item
            if time.time() - timestamp > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key, value):
        with self._lock:
            self._store[key] = (time.time(), value)

    def clear(self):
        with self._lock:
            self._store.clear()


def ttl_cache(ttl_seconds):
    """Decorator caching a function's return value per-arguments for ``ttl_seconds``.

    Only positional args and simple keyword args (hashable) are supported as the
    cache key, which is all the data functions here need. The underlying
    :class:`TTLCache` is exposed as ``fn.cache`` for tests and manual eviction.
    """
    cache = TTLCache(ttl_seconds)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            cached = cache.get(key)
            if cached is not None:
                return cached
            value = fn(*args, **kwargs)
            # Never cache falsy/empty results so a transient failure that
            # returns None/[] isn't pinned for the whole TTL.
            if value:
                cache.set(key, value)
            return value

        wrapper.cache = cache
        return wrapper
    return decorator
