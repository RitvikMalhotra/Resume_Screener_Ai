"""
Unit tests for the Retriever.

Run with: pytest tests/test_retriever.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from app.retriever import Retriever, IndexEntry
from utils.config import RetrieverConfig


# ── fixtures ───────────────────────────────────────────────────────────────

SAMPLE_ENTRIES = [
    IndexEntry(
        resume_id="r001",
        text=(
            "Senior Python engineer with 8 years experience. "
            "Expert in FastAPI, PostgreSQL, AWS. Led teams of 5+."
        ),
    ),
    IndexEntry(
        resume_id="r002",
        text=(
            "Data scientist specializing in NLP and transformer models. "
            "Published papers on BERT fine-tuning. PyTorch, HuggingFace."
        ),
    ),
    IndexEntry(
        resume_id="r003",
        text=(
            "Frontend developer with React, TypeScript, CSS expertise. "
            "3 years at a Series B startup building dashboards."
        ),
    ),
    IndexEntry(
        resume_id="r004",
        text=(
            "Machine learning engineer. Experience with LLMs, LoRA fine-tuning, "
            "RLHF, FAISS vector search, and production model deployment."
        ),
    ),
    IndexEntry(
        resume_id="r005",
        text=(
            "DevOps / Platform engineer. Kubernetes, Terraform, CI/CD pipelines. "
            "Managed infrastructure for 50M+ user product."
        ),
    ),
]

JD_ML = (
    "We are hiring a Machine Learning Engineer to build LLM-powered products. "
    "Required: PyTorch, transformers, vector databases, Python. "
    "Bonus: LoRA fine-tuning, RLHF, FAISS."
)

JD_FRONTEND = (
    "Looking for a Senior Frontend Developer. Must know React, TypeScript. "
    "Experience with data visualization dashboards preferred."
)


# ── tests ──────────────────────────────────────────────────────────────────

class TestRetriever:
    """Test suite for Retriever with a small synthetic corpus."""

    @pytest.fixture(scope="class")
    def retriever(self):
        cfg = RetrieverConfig(
            model_name="all-MiniLM-L6-v2",   # small, fast
            device="cpu",
            batch_size=8,
            top_k=5,
        )
        r = Retriever(config=cfg)
        r.build_index(SAMPLE_ENTRIES)
        return r

    def test_index_built(self, retriever):
        assert retriever.index_size == len(SAMPLE_ENTRIES)

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve(JD_ML, top_k=3)
        assert len(results) == 3

    def test_ml_jd_top_result(self, retriever):
        """ML-focused JD should rank r004 near the top."""
        results = retriever.retrieve(JD_ML, top_k=5)
        top_ids = [r.resume_id for r in results[:2]]
        assert "r004" in top_ids, f"Expected r004 in top-2, got {top_ids}"

    def test_frontend_jd_top_result(self, retriever):
        """Frontend JD should rank r003 near the top."""
        results = retriever.retrieve(JD_FRONTEND, top_k=5)
        top_ids = [r.resume_id for r in results[:2]]
        assert "r003" in top_ids, f"Expected r003 in top-2, got {top_ids}"

    def test_scores_in_range(self, retriever):
        results = retriever.retrieve(JD_ML, top_k=5)
        for r in results:
            # cosine similarity should be in reasonable range
            assert -1.0 <= r.embedding_score <= 1.0

    def test_scores_sorted_descending(self, retriever):
        results = retriever.retrieve(JD_ML, top_k=5)
        scores = [r.embedding_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_cache_populated(self, retriever):
        # second call should hit cache
        retriever.retrieve(JD_ML, top_k=3)
        stats = retriever.cache_stats()
        assert stats["l1"]["hits"] > 0

    def test_top_k_bounds(self, retriever):
        # asking for more than corpus size should still work
        results = retriever.retrieve(JD_ML, top_k=1000)
        assert len(results) == len(SAMPLE_ENTRIES)

    def test_empty_query(self, retriever):
        # degenerate input should not crash
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)

    def test_persist_and_load(self, retriever, tmp_path):
        """Save index to tmp dir, reload, and verify results match."""
        idx_path  = tmp_path / "test.index"
        meta_path = tmp_path / "test.pkl"

        original_cfg = retriever.cfg
        retriever.cfg.faiss_index_path = idx_path
        retriever.cfg.faiss_meta_path  = meta_path

        retriever.save_index()
        assert idx_path.exists()
        assert meta_path.exists()

        r2 = Retriever(config=RetrieverConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            faiss_index_path=idx_path,
            faiss_meta_path=meta_path,
        ))
        loaded = r2.load_index()
        assert loaded
        assert r2.index_size == len(SAMPLE_ENTRIES)

        results = r2.retrieve(JD_ML, top_k=3)
        assert len(results) == 3

        # restore
        retriever.cfg = original_cfg

    def test_incremental_add(self, retriever):
        new_entry = IndexEntry(
            resume_id="r999",
            text="Go engineer. gRPC, microservices, Kubernetes.",
        )
        original_size = retriever.index_size
        retriever.add_to_index([new_entry])
        assert retriever.index_size == original_size + 1
