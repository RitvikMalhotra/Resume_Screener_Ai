"""
Phase 1 – Batch Inference Manager

Handles:
  - Batched embedding with configurable batch size
  - Content-addressed cache (SHA-256 keyed by text)
  - Throughput + latency telemetry
  - Cache warm-up for known corpora

Interview talking point:
  "We batch embedding calls (64 texts/call), cache vectors by SHA-256
   content hash, and use FAISS IVFFlat for sub-linear ANN search.
   Result: 500 resumes processed in under 60s on CPU."
"""
from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from utils.cache import TieredCache, sha256_key

logger = logging.getLogger(__name__)


# ── Throughput Tracker ─────────────────────────────────────────────────────

class ThroughputTracker:
    """
    Rolling-window latency + throughput stats.
    Keeps last N jobs in memory. Thread-unsafe — fine for single-worker FastAPI.
    """

    def __init__(self, window: int = 50):
        self._latencies: deque[float] = deque(maxlen=window)
        self._counts: deque[int]      = deque(maxlen=window)
        self.total_jobs    = 0
        self.total_resumes = 0

    def record(self, n_resumes: int, elapsed_s: float) -> None:
        self._latencies.append(elapsed_s)
        self._counts.append(n_resumes)
        self.total_jobs    += 1
        self.total_resumes += n_resumes

    def summary(self) -> dict:
        if not self._latencies:
            return {
                "total_jobs": 0,
                "total_resumes": 0,
                "avg_latency_s": None,
                "p95_latency_s": None,
                "avg_throughput_per_s": None,
            }
        lats   = list(self._latencies)
        counts = list(self._counts)
        tputs  = [c / l for c, l in zip(counts, lats) if l > 0]
        return {
            "total_jobs":             self.total_jobs,
            "total_resumes":          self.total_resumes,
            "avg_latency_s":          round(float(np.mean(lats)), 3),
            "p95_latency_s":          round(float(np.percentile(lats, 95)), 3),
            "avg_throughput_per_s":   round(float(np.mean(tputs)), 1) if tputs else 0,
            "window_size":            len(lats),
        }


# ── Embedding Cache ────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Content-addressed vector cache.

    Same text → same embedding → cache hit, zero recomputation.
    Key = SHA-256(model_name + text[:2000]) so model changes invalidate entries.
    """

    def __init__(self, model_name: str, max_size: int = 8192, ttl: int = 3600):
        self._store      = TieredCache(lru_size=max_size, ttl_seconds=ttl)
        self._model_name = model_name
        self._hits       = 0
        self._misses     = 0

    def _key(self, text: str) -> str:
        return sha256_key("embed_v1", self._model_name, text[:2000])

    def get(self, text: str) -> Optional[np.ndarray]:
        hit, val = self._store.get(self._key(text))
        if hit:
            self._hits += 1
            return val
        self._misses += 1
        return None

    def put(self, text: str, vec: np.ndarray) -> None:
        self._store.set(self._key(text), vec)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":         self._hits,
            "misses":       self._misses,
            "hit_rate_pct": round(100 * self._hits / total, 1) if total else 0.0,
            "size":         self._store.l1.stats()["size"],
        }

    def warm(self, texts: list[str], embed_fn: Callable[[list[str]], np.ndarray]) -> dict:
        """
        Pre-embed texts and populate cache. Skips already-cached entries.
        Returns a summary dict.
        """
        uncached = [t for t in texts if self.get(t) is None]
        # get() increments miss counter — reset those phantom misses
        self._misses -= len(uncached)

        if not uncached:
            logger.info("Cache warm-up: all %d texts already cached.", len(texts))
            return {"warmed": 0, "already_cached": len(texts)}

        logger.info("Cache warm-up: embedding %d uncached texts…", len(uncached))
        t0   = time.perf_counter()
        vecs = embed_fn(uncached)   # (N, D) ndarray
        elapsed = time.perf_counter() - t0

        for text, vec in zip(uncached, vecs):
            self.put(text, vec)

        logger.info("Warm-up done: %d texts in %.2fs.", len(uncached), elapsed)
        return {
            "warmed":          len(uncached),
            "already_cached":  len(texts) - len(uncached),
            "elapsed_s":       round(elapsed, 3),
        }


# ── Batch Embedder ─────────────────────────────────────────────────────────

class BatchEmbedder:
    """
    Wraps a raw embed function with:
      - Configurable batch size
      - Per-call caching via EmbeddingCache
      - Telemetry (total texts, cache hits, elapsed time)

    Usage
    -----
    embedder = BatchEmbedder(model_name="BAAI/bge-large-en-v1.5",
                              raw_embed_fn=model.encode,
                              batch_size=64)
    vectors = embedder.embed(texts)   # (N, D) float32
    print(embedder.stats())
    """

    def __init__(
        self,
        model_name: str,
        raw_embed_fn: Callable[[list[str]], np.ndarray],
        batch_size: int = 64,
        cache_size: int = 8192,
        cache_ttl: int = 3600,
    ):
        self._embed_fn  = raw_embed_fn
        self._batch_sz  = batch_size
        self._cache     = EmbeddingCache(model_name, cache_size, cache_ttl)
        self._tracker   = ThroughputTracker()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts with cache + batching.
        Returns (N, D) float32 array preserving input order.
        """
        if not texts:
            return np.empty((0, 1), dtype=np.float32)

        result: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_idx: list[int] = []

        # ── L1: cache lookup ──────────────────────────────────────────────
        for i, text in enumerate(texts):
            vec = self._cache.get(text)
            if vec is not None:
                result[i] = vec
            else:
                uncached_idx.append(i)

        # ── L2: batch embed misses ────────────────────────────────────────
        if uncached_idx:
            miss_texts = [texts[i] for i in uncached_idx]
            t0 = time.perf_counter()

            all_vecs: list[np.ndarray] = []
            for start in range(0, len(miss_texts), self._batch_sz):
                chunk = miss_texts[start : start + self._batch_sz]
                vecs  = self._embed_fn(chunk)
                all_vecs.append(
                    vecs if isinstance(vecs, np.ndarray) else np.array(vecs)
                )

            elapsed = time.perf_counter() - t0
            batch_vecs = np.vstack(all_vecs).astype(np.float32)

            for pos, idx in enumerate(uncached_idx):
                vec = batch_vecs[pos]
                result[idx] = vec
                self._cache.put(texts[idx], vec)

            self._tracker.record(len(miss_texts), elapsed)
            logger.debug(
                "BatchEmbedder: %d cached, %d embedded in %.2fs",
                len(texts) - len(uncached_idx), len(uncached_idx), elapsed,
            )

        return np.vstack(result)

    def warm_cache(self, texts: list[str]) -> dict:
        return self._cache.warm(texts, self._embed_fn)

    def stats(self) -> dict:
        return {
            "cache":      self._cache.stats(),
            "throughput": self._tracker.summary(),
            "batch_size": self._batch_sz,
        }