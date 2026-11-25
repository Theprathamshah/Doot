"""Async per-domain rate limiter with configurable jitter."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse


class DomainRateLimiter:
    """Enforces a minimum interval + random jitter per target hostname."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 1.0,
        jitter_range: tuple[float, float] = (0.2, 1.0),
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.jitter_range = jitter_range
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request_at: dict[str, float] = {}

    async def acquire(self, url_or_domain: str) -> None:
        """Wait until a request to the given URL's domain is permitted."""
        parsed = urlparse(url_or_domain)
        domain = (parsed.hostname or url_or_domain).lower()
        lock = self._locks.setdefault(domain, asyncio.Lock())

        async with lock:
            now = self._time_fn()
            last = self._last_request_at.get(domain)
            if last is not None:
                jitter = (
                    random.uniform(self.jitter_range[0], self.jitter_range[1])
                    if self.jitter_range[1] > 0
                    else 0.0
                )
                wait_for = (self.min_interval_seconds + jitter) - (now - last)
                if wait_for > 0:
                    await self._sleep_fn(wait_for)
            self._last_request_at[domain] = self._time_fn()
