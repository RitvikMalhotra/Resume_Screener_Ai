"""
Phase 4 – Rate Limiter

Simple sliding-window rate limiter keyed by client IP.

Why sliding window over fixed window
-------------------------------------
  Fixed window: 100 req/min resets at :00 each minute.
  A client can send 100 at :59 and 100 at :01 → 200 in 2 seconds.
  Sliding window: looks at the last 60 seconds always → no burst exploit.

Interview talking point:
  "We use a sliding window rate limiter keyed by IP. In production
   we'd back this with Redis so limits are shared across workers.
   The current in-process implementation works for single-worker
   deployments and degrades gracefully — if the store fills up
   we evict the oldest entries rather than crashing."
"""
from __future__ import annotations
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60    # for /rank (heavy)
    requests_per_minute_light: int = 300  # for /health, /index/stats etc
    burst_multiplier: float = 1.5    # allow short bursts up to this * rpm
    window_seconds: int = 60


# ── Sliding window store ───────────────────────────────────────────────────

class SlidingWindowStore:
    """
    Per-key sliding window request log.
    Stores timestamps of recent requests per key (IP address).
    """

    def __init__(self, max_keys: int = 10000):
        self._store: dict[str, deque[float]] = defaultdict(deque)
        self._max_keys = max_keys

    def record_and_check(
        self,
        key: str,
        limit: int,
        window_s: int = 60,
    ) -> tuple[bool, int, int]:
        """
        Record a request and check if limit is exceeded.

        Returns
        -------
        (allowed, current_count, limit)
        """
        now    = time.monotonic()
        cutoff = now - window_s

        # evict old entries
        q = self._store[key]
        while q and q[0] < cutoff:
            q.popleft()

        count = len(q)

        if count >= limit:
            return False, count, limit

        q.append(now)
        return True, count + 1, limit

    def reset(self, key: str) -> None:
        if key in self._store:
            self._store[key].clear()

    def cleanup(self, window_s: int = 60) -> int:
        """Remove keys with no recent requests. Returns number removed."""
        now    = time.monotonic()
        cutoff = now - window_s
        empty  = [
            k for k, q in self._store.items()
            if not q or q[-1] < cutoff
        ]
        for k in empty:
            del self._store[k]
        return len(empty)

    def stats(self) -> dict:
        return {"tracked_ips": len(self._store)}


# ── Rate limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """
    FastAPI-compatible rate limiter.

    Usage
    -----
    limiter = RateLimiter()

    # In endpoint:
    limiter.check(request, heavy=True)   # raises 429 if exceeded
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.cfg   = config or RateLimitConfig()
        self._store = SlidingWindowStore()
        self._total_blocked = 0

    def _get_client_ip(self, request: Request) -> str:
        """Extract real IP, respecting X-Forwarded-For for proxied setups."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request, heavy: bool = False) -> dict:
        """
        Check rate limit for this request.
        Raises HTTP 429 if exceeded.
        Returns rate limit headers dict on success.
        """
        ip    = self._get_client_ip(request)
        limit = (
            self.cfg.requests_per_minute
            if heavy
            else self.cfg.requests_per_minute_light
        )

        allowed, count, max_count = self._store.record_and_check(
            key      = ip,
            limit    = int(limit * self.cfg.burst_multiplier),
            window_s = self.cfg.window_seconds,
        )

        headers = {
            "X-RateLimit-Limit":     str(limit),
            "X-RateLimit-Remaining": str(max(0, limit - count)),
            "X-RateLimit-Window":    f"{self.cfg.window_seconds}s",
        }

        if not allowed:
            self._total_blocked += 1
            logger.warning(
                "Rate limit exceeded for IP %s (%d/%d requests in %ds)",
                ip, count, limit, self.cfg.window_seconds,
            )
            raise HTTPException(
                status_code = 429,
                detail      = {
                    "error":       "Rate limit exceeded.",
                    "limit":       limit,
                    "window_s":    self.cfg.window_seconds,
                    "retry_after": self.cfg.window_seconds,
                },
                headers = {"Retry-After": str(self.cfg.window_seconds)},
            )

        return headers

    def stats(self) -> dict:
        return {
            "total_blocked": self._total_blocked,
            **self._store.stats(),
        }