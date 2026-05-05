"""
Batching helpers for embedding and reranking inference.

Key ideas:
  - Dynamic batching: group items into batches of <= max_size
  - Adaptive sizing: reduce batch when OOM, increase back after success
  - Parallel embedding with ThreadPoolExecutor for CPU models
"""
from __future__ import annotations
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


def chunks(lst: list[T], n: int) -> Generator[list[T], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def flatten(nested: list[list[T]]) -> list[T]:
    return [item for sub in nested for item in sub]


class AdaptiveBatcher:
    """
    Wraps a batch-inference function and adapts the batch size at runtime.

    If the underlying function raises a MemoryError or RuntimeError (OOM),
    the batch size is halved and the batch is retried. On success, size
    slowly recovers toward the original max.

    Parameters
    ----------
    fn          : callable(batch) → list of results, same length as batch
    max_batch   : initial / maximum batch size
    min_batch   : floor; raises if we go below this
    recover_step: how much to add back after each successful batch
    """

    def __init__(
        self,
        fn: Callable[[list], list],
        max_batch: int = 64,
        min_batch: int = 4,
        recover_step: int = 4,
    ):
        self._fn          = fn
        self._max_batch   = max_batch
        self._min_batch   = min_batch
        self._recover_step = recover_step
        self._current     = max_batch

    def run(self, items: list) -> list:
        """Process all items, returning results in original order."""
        results: list = [None] * len(items)
        idx = 0

        while idx < len(items):
            batch = items[idx : idx + self._current]
            try:
                batch_results = self._fn(batch)
                for j, r in enumerate(batch_results):
                    results[idx + j] = r
                idx += self._current
                # gentle recovery
                self._current = min(
                    self._max_batch, self._current + self._recover_step
                )
            except (MemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
                    new_size = self._current // 2
                    if new_size < self._min_batch:
                        raise RuntimeError(
                            f"Batch size fell below minimum ({self._min_batch}). "
                            "Cannot recover from OOM."
                        ) from exc
                    logger.warning(
                        "OOM detected; reducing batch size %d → %d",
                        self._current, new_size,
                    )
                    self._current = new_size
                else:
                    raise

        return results


def parallel_embed(
    texts: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    batch_size: int = 64,
    n_workers: int = 4,
) -> np.ndarray:
    """
    Split texts into batches, embed in parallel threads (useful for CPU).
    Returns (N, D) float32 array.
    """
    batches = list(chunks(texts, batch_size))
    batch_embeddings: list[np.ndarray | None] = [None] * len(batches)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(embed_fn, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            i = futures[fut]
            batch_embeddings[i] = fut.result()

    elapsed = time.perf_counter() - t0
    result = np.vstack([e for e in batch_embeddings if e is not None])
    logger.debug(
        "Embedded %d texts in %.2fs (%.0f texts/sec)",
        len(texts), elapsed, len(texts) / elapsed if elapsed > 0 else 0,
    )
    return result


def estimate_throughput(n_items: int, elapsed_s: float) -> dict:
    """Return a simple throughput summary dict."""
    return {
        "n_items": n_items,
        "elapsed_s": round(elapsed_s, 3),
        "items_per_second": round(n_items / elapsed_s, 1) if elapsed_s > 0 else 0,
        "ms_per_item": round(elapsed_s * 1000 / n_items, 2) if n_items > 0 else 0,
    }
