"""
Unit tests for CrossEncoderReranker and evaluation metrics.

Run with: pytest tests/test_reranker.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app.retriever import RetrievalResult
from app.reranker import CrossEncoderReranker
from app.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k,
    average_precision_at_k, mean_reciprocal_rank,
    spearman_rho, evaluate, batch_evaluate,
)
from utils.config import RerankerConfig


# ── fixtures ───────────────────────────────────────────────────────────────

JD = (
    "Hiring ML Engineer with PyTorch, LLM fine-tuning, and FAISS experience."
)

CANDIDATES = [
    RetrievalResult("r001", "Python ML engineer, LLM fine-tuning, FAISS, PyTorch.", 0.91, 0),
    RetrievalResult("r002", "Data scientist, scikit-learn, pandas, SQL.", 0.72, 1),
    RetrievalResult("r003", "Frontend developer, React, TypeScript, CSS.", 0.55, 2),
    RetrievalResult("r004", "ML researcher, transformers, LoRA, RLHF.", 0.85, 3),
    RetrievalResult("r005", "DevOps engineer, Kubernetes, Terraform.", 0.48, 4),
]

RELEVANT = {"r001", "r004"}


# ── reranker tests ─────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    @pytest.fixture(scope="class")
    def reranker(self):
        cfg = RerankerConfig(
            use_llm_reranker=False,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_batch_size=4,
            top_n=3,
        )
        return CrossEncoderReranker(cfg)

    def test_returns_top_n(self, reranker):
        results = reranker.rerank(JD, CANDIDATES, top_n=3)
        assert len(results) == 3

    def test_scores_are_floats(self, reranker):
        results = reranker.rerank(JD, CANDIDATES)
        for r in results:
            assert isinstance(r.final_score, float)
            assert isinstance(r.rerank_score, float)

    def test_sorted_descending(self, reranker):
        results = reranker.rerank(JD, CANDIDATES)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, reranker):
        results = reranker.rerank(JD, CANDIDATES, top_n=5)
        for i, r in enumerate(results):
            assert r.rank == i

    def test_empty_candidates(self, reranker):
        results = reranker.rerank(JD, [])
        assert results == []

    def test_relevant_near_top(self, reranker):
        """r001 and r004 (ML-focused) should rank above r003 and r005."""
        results = reranker.rerank(JD, CANDIDATES, top_n=5)
        top2_ids = {r.resume_id for r in results[:2]}
        # at least one of the relevant IDs should be in top-2
        assert len(top2_ids & RELEVANT) >= 1


# ── metrics tests ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_precision_at_k_perfect(self):
        assert precision_at_k({"a", "b"}, ["a", "b", "c"], 2) == 1.0

    def test_precision_at_k_zero(self):
        assert precision_at_k({"x"}, ["a", "b", "c"], 3) == 0.0

    def test_precision_at_k_partial(self):
        p = precision_at_k({"a", "d"}, ["a", "b", "c", "d"], 4)
        assert abs(p - 0.5) < 1e-6

    def test_recall_at_k_perfect(self):
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], 2) == 1.0

    def test_recall_at_k_partial(self):
        r = recall_at_k({"a", "b", "c"}, ["a", "b"], 2)
        assert abs(r - 2 / 3) < 1e-6

    def test_ndcg_perfect_ranking(self):
        # ideal: all relevant first
        ndcg = ndcg_at_k({"a", "b"}, ["a", "b", "c", "d"], 4)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_worst_ranking(self):
        ndcg = ndcg_at_k({"c", "d"}, ["a", "b", "c", "d"], 2)
        assert ndcg == 0.0   # no relevant in top-2

    def test_mrr_first_hit(self):
        mrr = mean_reciprocal_rank({"a"}, ["a", "b", "c"])
        assert abs(mrr - 1.0) < 1e-6

    def test_mrr_second_hit(self):
        mrr = mean_reciprocal_rank({"b"}, ["a", "b", "c"])
        assert abs(mrr - 0.5) < 1e-6

    def test_mrr_no_hit(self):
        mrr = mean_reciprocal_rank({"z"}, ["a", "b", "c"])
        assert mrr == 0.0

    def test_ap_at_k(self):
        # a, b both relevant; perfect order
        ap = average_precision_at_k({"a", "b"}, ["a", "b", "c"], 3)
        assert abs(ap - 1.0) < 1e-6

    def test_spearman_perfect(self):
        gold = {"a": 1, "b": 2, "c": 3}
        pred = {"a": 1, "b": 2, "c": 3}
        assert abs(spearman_rho(gold, pred) - 1.0) < 1e-6

    def test_spearman_inverse(self):
        gold = {"a": 1, "b": 2, "c": 3}
        pred = {"a": 3, "b": 2, "c": 1}
        assert abs(spearman_rho(gold, pred) - (-1.0)) < 1e-6

    def test_evaluate_returns_all_keys(self):
        result = evaluate({"a", "b"}, ["a", "c", "b", "d"], k_values=[1, 3])
        expected_keys = {
            "precision@1", "recall@1", "ap@1", "ndcg@1", "f1@1",
            "precision@3", "recall@3", "ap@3", "ndcg@3", "f1@3",
            "mrr",
        }
        assert expected_keys <= result.keys()

    def test_batch_evaluate_averages(self):
        queries = [
            {"relevant_ids": {"a"}, "ranked_ids": ["a", "b"]},
            {"relevant_ids": {"b"}, "ranked_ids": ["a", "b"]},
        ]
        result = batch_evaluate(queries, k_values=[1])
        # p@1: first query = 1.0, second = 0.0 → avg = 0.5
        assert abs(result["precision@1"] - 0.5) < 1e-6
        assert result["num_queries"] == 2
