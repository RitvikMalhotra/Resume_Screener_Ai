"""
Two-level caching:
  1. In-process LRU cache (zero latency, process-local)
  2. Optional Redis cache (shared across workers, survives restarts)

Both layers use a content-hash as the key so identical texts always
hit the cache regardless of how they arrived.
"""
from __future__ import annotations
import hashlib
import pickle
import time
import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sha256_key(*parts: str) -> str:
    """Deterministic cache key from one or more strings."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


class LRUCache:
    """
    Thread-unsafe in-process LRU cache with optional TTL.
    For a multithreaded FastAPI app, wrap with a lock or use
    a thread-safe alternative like cachetools.TTLCache.
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float | None = 3600):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_size    = max_size
        self.ttl_seconds = ttl_seconds
        self._hits   = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        if key not in self._store:
            self._misses += 1
            return False, None
        value, ts = self._store[key]
        if self.ttl_seconds and (time.monotonic() - ts) > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return False, None
        # move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return True, value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.monotonic())
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)  # evict LRU

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "size": len(self._store),
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = 0


class RedisCache:
    """
    Optional Redis-backed cache. Falls back silently if Redis is unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        self._ttl = ttl
        self._client = None
        try:
            import redis
            client = redis.Redis(host=host, port=port, socket_connect_timeout=1)
            client.ping()
            self._client = client
            logger.info("Redis cache connected at %s:%d", host, port)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) – using in-process cache only", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> tuple[bool, Any]:
        if not self._client:
            return False, None
        try:
            raw = self._client.get(key)
            if raw is None:
                return False, None
            return True, pickle.loads(raw)
        except Exception:
            return False, None

    def set(self, key: str, value: Any) -> None:
        if not self._client:
            return
        try:
            self._client.setex(key, self._ttl, pickle.dumps(value))
        except Exception:
            pass


class TieredCache:
    """
    Combines LRU (L1) + Redis (L2). Write-through on set.
    """

    def __init__(
        self,
        lru_size: int = 512,
        ttl_seconds: int = 3600,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_redis: bool = False,
    ):
        self.l1 = LRUCache(max_size=lru_size, ttl_seconds=ttl_seconds)
        self.l2: Optional[RedisCache] = (
            RedisCache(host=redis_host, port=redis_port, ttl=ttl_seconds)
            if enable_redis
            else None
        )

    def get(self, key: str) -> tuple[bool, Any]:
        hit, val = self.l1.get(key)
        if hit:
            return True, val
        if self.l2:
            hit, val = self.l2.get(key)
            if hit:
                self.l1.set(key, val)   # warm L1
                return True, val
        return False, None

    def set(self, key: str, value: Any) -> None:
        self.l1.set(key, value)
        if self.l2:
            self.l2.set(key, value)

    def stats(self) -> dict:
        return {
            "l1": self.l1.stats(),
            "l2_available": self.l2.available if self.l2 else False,
        }
