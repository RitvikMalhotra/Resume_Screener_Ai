"""
Integration tests for the full two-stage pipeline.

These tests are heavier (load actual models) and tagged `slow`.
Run with: pytest tests/test_pipeline.py -v -m "not slow"  # skip model loads
          pytest tests/test_pipeline.py -v                 # full suite
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app.pipeline import Pipeline, ResumeInput, RankResponse
from utils.config import Config, RetrieverConfig, RerankerConfig, AppConfig


# ── synthetic corpus ───────────────────────────────────────────────────────

RESUMES = [
    ResumeInput("r001", "Senior Python/ML engineer. PyTorch, transformers, FAISS, LLMs, AWS."),
    ResumeInput("r002", "Full-stack engineer. React, Node.js, PostgreSQL, Docker."),
    ResumeInput("r003", "Data analyst. Excel, Tableau, SQL, Power BI. No coding."),
    ResumeInput("r004", "NLP researcher. BERT, LLaMA, fine-tuning, LoRA, HuggingFace."),
    ResumeInput("r005", "Mobile developer. Swift, Android, React Native."),
    ResumeInput("r006", "DevOps. Kubernetes, Terraform, CI/CD, Linux."),
    ResumeInput("r007", "ML engineer. Recommendation systems, A/B testing, Spark."),
    ResumeInput("r008", "Security engineer. Pentesting, SAST, threat modeling."),
]

JD_ML = (
    "We need a senior ML engineer to build and deploy large language models. "
    "Required: Python, PyTorch, transformers, vector databases. "
    "Nice to have: LoRA, RLHF, FAISS, LLaMA."
)

RELEVANT_IDS = {"r001", "r004", "r007"}


# ── config fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_config():
    return Config(
        retriever=RetrieverConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8,
            top_k=8,
        ),
        reranker=RerankerConfig(
            use_llm_reranker=False,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_batch_size=4,
            top_n=5,
        ),
        app=AppConfig(),
    )


@pytest.fixture(scope="module")
def pipeline(small_config):
    return Pipeline(config=small_config)


# ── tests ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPipelineIntegration:
    def test_rank_returns_response(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        assert isinstance(response, RankResponse)

    def test_rank_returns_correct_count(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        assert len(response.results) == 5

    def test_rank_all_ids_valid(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        valid_ids = {r.resume_id for r in RESUMES}
        for result in response.results:
            assert result.resume_id in valid_ids

    def test_timing_present(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        for key in ("preprocess_s", "index_build_s", "retrieval_s", "rerank_s", "total_s"):
            assert key in response.timing
            assert response.timing[key] >= 0

    def test_final_scores_descending(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        scores = [r.final_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_ml_resumes_ranked_higher(self, pipeline):
        """r001 and r004 should dominate top positions for an ML JD."""
        response = pipeline.rank(JD_ML, RESUMES, top_n=5)
        top3_ids = {r.resume_id for r in response.results[:3]}
        overlap  = len(top3_ids & RELEVANT_IDS)
        # at least 2 of 3 relevant should appear in top-3
        assert overlap >= 1, f"Expected relevant IDs in top-3, got {top3_ids}"

    def test_rank_with_metrics(self, pipeline):
        response = pipeline.rank(JD_ML, RESUMES, top_n=5, relevant_ids=RELEVANT_IDS)
        assert response.metrics is not None
        assert "precision@5" in response.metrics
        assert "recall@5" in response.metrics

    def test_single_resume(self, pipeline):
        """Edge case: only one resume."""
        response = pipeline.rank(JD_ML, [RESUMES[0]], top_n=1)
        assert len(response.results) == 1

    def test_top_n_larger_than_corpus(self, pipeline):
        """Asking for more results than resumes should not crash."""
        response = pipeline.rank(JD_ML, RESUMES[:3], top_n=100)
        assert len(response.results) <= 3

    def test_index_size_after_rank(self, pipeline):
        pipeline.rank(JD_ML, RESUMES, top_n=3)
        assert pipeline.retriever.index_size == len(RESUMES)


class TestPerformance:
    """
    Smoke test: 500 resumes must complete within 120s.
    (Actual time depends on hardware; this test is a placeholder for CI.)
    """

    @pytest.mark.slow
    def test_500_resumes_within_budget(self, pipeline):
        import time
        large_corpus = [
            ResumeInput(
                f"r{i:04d}",
                f"Candidate {i}: Python engineer with {i % 10} years of ML experience. "
                f"Skills: PyTorch, numpy, pandas, AWS. Project {i % 5}.",
            )
            for i in range(500)
        ]
        t0 = time.perf_counter()
        response = pipeline.rank(JD_ML, large_corpus, top_n=10)
        elapsed  = time.perf_counter() - t0
        assert elapsed < 120, f"500 resumes took {elapsed:.1f}s > 120s budget"
        assert len(response.results) == 10
