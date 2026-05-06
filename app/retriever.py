"""
Stage 1 – Bi-encoder retrieval.

Phase 1 changes
---------------
  - Replaced raw SentenceTransformer.encode() calls with BatchEmbedder
    → automatic caching + batching + telemetry in one place
  - Added FAISSIndexInfo dataclass: exposes index type, size, dim to API
  - Added explicit logging of index type chosen (Flat vs IVFFlat)
  - Cache stats now surfaced via .stats() for the /index/stats endpoint

FAISS index selection
---------------------
  N < 1000  → IndexFlatIP   (brute force, exact, fastest for small N)
  N >= 1000 → IndexIVFFlat  (approximate, nlist clusters, nprobe searched)

Interview talking point:
  "FAISS IVFFlat partitions the vector space into nlist Voronoi cells.
   At query time we only search nprobe cells (~10% of index), giving
   ~95% recall at 10x the speed of brute-force on large corpora."
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

from app.batch_manager import BatchEmbedder
from utils.cache import sha256_key
from utils.batching import chunks
from utils.config import RetrieverConfig, get_config


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    resume_id: str
    text: str
    embedding_score: float
    rank: int


@dataclass
class IndexEntry:
    resume_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FAISSIndexInfo:
    """
    Snapshot of FAISS index metadata.
    Returned by /index/stats — makes the system auditable.
    """
    index_type: str        # "FlatIP" or "IVFFlat"
    n_vectors: int
    dimension: int
    nlist: Optional[int]   # IVFFlat only
    nprobe: Optional[int]  # IVFFlat only
    index_size_mb: float

    def to_dict(self) -> dict:
        return {
            "index_type":    self.index_type,
            "n_vectors":     self.n_vectors,
            "dimension":     self.dimension,
            "nlist":         self.nlist,
            "nprobe":        self.nprobe,
            "index_size_mb": self.index_size_mb,
        }


# ── Retriever ──────────────────────────────────────────────────────────────

class Retriever:
    """
    Bi-encoder + FAISS retrieval with Phase 1 upgrades:
      - BatchEmbedder handles all embed calls (cached + batched)
      - FAISSIndexInfo exposed for observability
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.cfg      = config or get_config().retriever
        self._model: Optional[SentenceTransformer] = None
        self._embedder: Optional[BatchEmbedder]    = None
        self._index   = None
        self._entries: list[IndexEntry] = []
        self._dim: int = 0

    # ── model + embedder ───────────────────────────────────────────────────

    def _get_embedder(self) -> BatchEmbedder:
        if self._embedder is None:
            if not _ST:
                raise RuntimeError("sentence-transformers is required")
            logger.info("Loading embedding model: %s", self.cfg.model_name)
            model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
            model.max_seq_length = self.cfg.max_seq_len
            self._model = model

            def _raw_embed(texts: list[str]) -> np.ndarray:
                return model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

            self._embedder = BatchEmbedder(
                model_name=self.cfg.model_name,
                raw_embed_fn=_raw_embed,
                batch_size=self.cfg.batch_size,
            )
        return self._embedder

    # ── embedding ──────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._get_embedder().embed(texts)
        if vecs.shape[0] > 0:
            self._dim = vecs.shape[1]
        return vecs

    # ── index ──────────────────────────────────────────────────────────────

    def build_index(self, entries: list[IndexEntry]) -> None:
        if not _FAISS:
            raise RuntimeError("faiss-cpu is required")
        if not entries:
            raise ValueError("Cannot build index from empty entry list")

        logger.info("Building FAISS index for %d entries…", len(entries))
        texts      = [e.text for e in entries]
        embeddings = self.embed(texts).astype(np.float32)
        N, D       = embeddings.shape
        self._entries = list(entries)
        self._dim     = D

        if N < 1000:
            logger.info("Index type: FlatIP (N=%d < 1000, exact search)", N)
            index = faiss.IndexFlatIP(D)
            self._index_type = "FlatIP"
        else:
            nlist = min(self.cfg.faiss_nlist, N // 10)
            logger.info(
                "Index type: IVFFlat (N=%d, nlist=%d, nprobe=%d)",
                N, nlist, self.cfg.faiss_nprobe,
            )
            quantizer = faiss.IndexFlatIP(D)
            index     = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.cfg.faiss_nprobe
            index.train(embeddings)
            self._index_type = "IVFFlat"

        index.add(embeddings)
        self._index = index
        logger.info("Index ready: %d vectors, dim=%d", index.ntotal, D)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k         = min(top_k or self.cfg.top_k, len(self._entries))
        query_vec = self.embed([query])
        scores, indices = self._index.search(query_vec, k)
        scores, indices = scores[0], indices[0]

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
                continue
            entry = self._entries[idx]
            results.append(RetrievalResult(
                resume_id=entry.resume_id,
                text=entry.text,
                embedding_score=float(score),
                rank=rank,
            ))
        return results

    # ── persistence ────────────────────────────────────────────────────────

    def save_index(self) -> None:
        if self._index is None:
            return
        idx_path  = self.cfg.faiss_index_path
        meta_path = self.cfg.faiss_meta_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump({
                "entries":     self._entries,
                "dim":         self._dim,
                "index_type":  getattr(self, "_index_type", "unknown"),
            }, f)
        logger.info("Index saved → %s", idx_path)

    def load_index(self) -> bool:
        idx_path  = self.cfg.faiss_index_path
        meta_path = self.cfg.faiss_meta_path
        if not idx_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._entries    = meta["entries"]
        self._dim        = meta["dim"]
        self._index_type = meta.get("index_type", "unknown")
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.cfg.faiss_nprobe
        logger.info("Loaded index: %d vectors, dim=%d", self._index.ntotal, self._dim)
        return True

    def add_to_index(self, entries: list[IndexEntry]) -> None:
        if self._index is None:
            return self.build_index(entries)
        texts      = [e.text for e in entries]
        embeddings = self.embed(texts).astype(np.float32)
        self._index.add(embeddings)
        self._entries.extend(entries)

    # ── observability ──────────────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0

    def index_info(self) -> Optional[FAISSIndexInfo]:
        if self._index is None:
            return None
        size_mb = (self._dim * self.index_size * 4) / (1024 ** 2)
        return FAISSIndexInfo(
            index_type  = getattr(self, "_index_type", "unknown"),
            n_vectors   = self.index_size,
            dimension   = self._dim,
            nlist       = getattr(self._index, "nlist", None),
            nprobe      = getattr(self._index, "nprobe", None),
            index_size_mb = round(size_mb, 2),
        )

    def cache_stats(self) -> dict:
        if self._embedder is None:
            return {}
        return self._embedder.stats()

    def warm_cache(self, texts: list[str]) -> dict:
        return self._get_embedder().warm_cache(texts)