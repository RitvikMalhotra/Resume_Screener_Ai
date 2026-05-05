"""
Stage 1 – Bi-encoder retrieval.

Architecture
------------
  - SentenceTransformers bi-encoder for embeddings
  - FAISS IVFFlat index (L2, with inner-product renormalization → cosine)
  - Persistent index: save/load from disk between restarts
  - Tiered cache: embedding results cached by text hash

Design notes
------------
  FAISS IVFFlat chosen over Flat because:
    - IVFFlat with nlist=100, nprobe=10 gives ~95% recall at 10x speed on 10k+ items
    - Switch to IndexHNSWFlat for even better recall if memory allows
  
  For 500 resumes we also support brute-force Flat index (simpler, fast enough).
"""
from __future__ import annotations
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── lazy imports (expensive) ───────────────────────────────────────────────
try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False
    logger.warning("FAISS not installed – retrieval disabled")

try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except ImportError:
    _ST = False
    logger.warning("sentence-transformers not installed")

from utils.cache import TieredCache, sha256_key
from utils.batching import chunks
from utils.config import RetrieverConfig, get_config


@dataclass
class RetrievalResult:
    resume_id: str
    text: str
    embedding_score: float          # cosine similarity [0, 1]
    rank: int


@dataclass
class IndexEntry:
    resume_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class Retriever:
    """
    Manages bi-encoder embedding + FAISS index lifecycle.

    Typical flow
    ------------
    r = Retriever()
    r.build_index(entries)           # build from scratch
    results = r.retrieve(jd_text, top_k=50)

    Persistence
    -----------
    r.save_index()                   # writes .index + .pkl files
    r.load_index()                   # restores from disk
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.cfg = config or get_config().retriever
        self._model: Optional[SentenceTransformer] = None
        self._index = None              # faiss.Index
        self._entries: list[IndexEntry] = []
        self._dim: int = 0
        self._cache = TieredCache(lru_size=4096, ttl_seconds=3600)

    # ── model ──────────────────────────────────────────────────────────────

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            if not _ST:
                raise RuntimeError("sentence-transformers is required")
            logger.info("Loading embedding model: %s", self.cfg.model_name)
            self._model = SentenceTransformer(
                self.cfg.model_name, device=self.cfg.device
            )
            self._model.max_seq_length = self.cfg.max_seq_len
        return self._model

    # ── embedding ──────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts. Results are cached by content hash.
        Returns (N, D) float32 array with L2-normalized vectors.
        """
        if not texts:
            return np.empty((0, self._dim or 1), dtype=np.float32)

        model = self._get_model()
        results: list[np.ndarray | None] = [None] * len(texts)
        to_embed_idx: list[int] = []

        # L1 cache lookup
        for i, text in enumerate(texts):
            key = sha256_key("embed", self.cfg.model_name, text[:2000])
            hit, val = self._cache.get(key)
            if hit:
                results[i] = val
            else:
                to_embed_idx.append(i)

        # batch-embed cache misses
        if to_embed_idx:
            miss_texts = [texts[i] for i in to_embed_idx]
            t0 = time.perf_counter()
            embeddings = model.encode(
                miss_texts,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=True,      # cosine via inner product
                show_progress_bar=len(miss_texts) > 100,
                convert_to_numpy=True,
            ).astype(np.float32)
            elapsed = time.perf_counter() - t0
            logger.debug(
                "Embedded %d texts in %.2fs", len(miss_texts), elapsed
            )

            for pos, idx in enumerate(to_embed_idx):
                vec = embeddings[pos]
                results[idx] = vec
                key = sha256_key("embed", self.cfg.model_name, texts[idx][:2000])
                self._cache.set(key, vec)

            self._dim = embeddings.shape[1]

        return np.vstack(results)

    # ── index ──────────────────────────────────────────────────────────────

    def build_index(self, entries: list[IndexEntry]) -> None:
        """Build FAISS index from a list of IndexEntry objects."""
        if not _FAISS:
            raise RuntimeError("faiss-cpu / faiss-gpu is required")
        if not entries:
            raise ValueError("Cannot build index from empty entry list")

        logger.info("Building FAISS index for %d entries …", len(entries))
        texts = [e.text for e in entries]
        embeddings = self.embed(texts)          # (N, D)
        N, D = embeddings.shape
        self._entries = list(entries)

        if N < 1000:
            # small corpus → brute-force flat index
            logger.info("Using IndexFlatIP (N=%d < 1000)", N)
            index = faiss.IndexFlatIP(D)
        else:
            # IVF for large corpora
            nlist = min(self.cfg.faiss_nlist, N // 10)
            logger.info(
                "Using IndexIVFFlat (N=%d, nlist=%d, nprobe=%d)",
                N, nlist, self.cfg.faiss_nprobe,
            )
            quantizer = faiss.IndexFlatIP(D)
            index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.cfg.faiss_nprobe
            index.train(embeddings)

        index.add(embeddings)
        self._index = index
        self._dim = D
        logger.info("Index built: %d vectors, dim=%d", index.ntotal, D)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K most similar entries to the query.
        Returns list of RetrievalResult sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index().")

        k = min(top_k or self.cfg.top_k, len(self._entries))
        query_vec = self.embed([query])         # (1, D)

        scores, indices = self._index.search(query_vec, k)  # (1, k)
        scores, indices = scores[0], indices[0]

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:                         # FAISS returns -1 for padding
                continue
            entry = self._entries[idx]
            results.append(
                RetrievalResult(
                    resume_id=entry.resume_id,
                    text=entry.text,
                    embedding_score=float(score),
                    rank=rank,
                )
            )
        return results

    # ── persistence ────────────────────────────────────────────────────────

    def save_index(self) -> None:
        if self._index is None:
            logger.warning("No index to save.")
            return
        idx_path  = self.cfg.faiss_index_path
        meta_path = self.cfg.faiss_meta_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"entries": self._entries, "dim": self._dim}, f
            )
        logger.info("Index saved → %s", idx_path)

    def load_index(self) -> bool:
        """Returns True if loaded successfully, False if files not found."""
        idx_path  = self.cfg.faiss_index_path
        meta_path = self.cfg.faiss_meta_path
        if not idx_path.exists() or not meta_path.exists():
            logger.info("No persisted index found at %s", idx_path)
            return False
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._entries = meta["entries"]
        self._dim     = meta["dim"]
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.cfg.faiss_nprobe
        logger.info(
            "Loaded index: %d vectors, dim=%d", self._index.ntotal, self._dim
        )
        return True

    def add_to_index(self, entries: list[IndexEntry]) -> None:
        """Incrementally add new entries to an existing index."""
        if self._index is None:
            return self.build_index(entries)
        texts = [e.text for e in entries]
        embeddings = self.embed(texts).astype(np.float32)
        self._index.add(embeddings)
        self._entries.extend(entries)
        logger.info(
            "Added %d entries; index now has %d total",
            len(entries), self._index.ntotal,
        )

    # ── utils ──────────────────────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0

    def cache_stats(self) -> dict:
        return self._cache.stats()
