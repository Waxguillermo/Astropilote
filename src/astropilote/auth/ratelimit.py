"""Thread-safe sliding-window rate limiter keyed by user ID or client IP."""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from astropilote.config import settings


class SlidingWindowRateLimiter:
    """Per-key 60-second sliding-window request limiter."""

    def __init__(self, *, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """Record a hit for `key`. Returns `(allowed, retry_after_seconds)`."""
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_seconds
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (ts - bucket[0])) + 1)
                return False, retry_after
            bucket.append(ts)
            return True, 0

    def reset(self) -> None:
        """Clear all recorded buckets (primarily for tests)."""
        with self._lock:
            self._hits.clear()


_DEFAULT_LIMITER = SlidingWindowRateLimiter(max_requests=settings.rate_limit_rpm)


def enforce_rate_limit(
    request: Request,
    *,
    subject_key: str | None = None,
    limiter: SlidingWindowRateLimiter | None = None,
) -> None:
    """Enforce rate-limiting in `cloud` mode (no-op in `local` solo mode)."""
    if settings.auth_mode != "cloud":
        return
    active = limiter or _DEFAULT_LIMITER
    client_host = request.client.host if request.client else "unknown"
    key = subject_key or f"ip:{client_host}"
    allowed, retry_after = active.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending more requests.",
            headers={"Retry-After": str(retry_after)},
        )
