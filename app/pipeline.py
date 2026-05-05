"""
Pipeline orchestrator — wires Stage 1 (retriever) and Stage 2 (reranker).

Responsibilities
----------------
  - Accept raw resume texts + job description
  - Preprocess all inputs
  - Run retrieval → reranking
  - Return structured results with timing telemetry
  - Expose an async-friendly interface for FastAPI
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from app.preprocessor import ResumePreprocessor
from app.retriever import Retriever, IndexEntry, RetrievalResult
from app.reranker import get_reranker, RerankResult
from app.metrics import evaluate
from utils.config import get_config, Config

logger = logging.getLogger(__name__)


@dataclass
class ResumeInput:
    resume_id: str
    content: str | bytes          # raw text, PDF bytes, or DOCX bytes
    filename: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RankResponse:
    job_description: str
    results: list[RerankResult]
    total_candidates: int
    timing: dict = field(default_factory=dict)
    metrics: Optional[dict] = None    # only if ground truth provided


class Pipeline:
    """
    Two-stage resume screening pipeline.

    Usage
    -----
    pipeline = Pipeline()

    # One-time index build (or load from disk)
    pipeline.build_index(resume_inputs)

    # Per-request scoring
    response = pipeline.rank(job_description, resume_inputs, top_n=10)
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg        = config or get_config()
        self.preprocessor = ResumePreprocessor(
            max_chars=self.cfg.app.max_resume_chars
        )
        self.retriever  = Retriever(self.cfg.retriever)
        self.reranker   = get_reranker(self.cfg.reranker)
        self._executor  = ThreadPoolExecutor(max_workers=4)

    # ── public API ─────────────────────────────────────────────────────────

    def build_index(self, resumes: list[ResumeInput]) -> None:
        """Preprocess + embed + index a batch of resumes."""
        entries = [
            IndexEntry(
                resume_id=r.resume_id,
                text=self.preprocessor.process(r.content, r.filename),
                metadata=r.metadata,
            )
            for r in resumes
        ]
        self.retriever.build_index(entries)

    def rank(
        self,
        job_description: str,
        resumes: list[ResumeInput],
        top_n: Optional[int] = None,
        relevant_ids: Optional[set[str]] = None,   # for evaluation
    ) -> RankResponse:
        """
        Full two-stage ranking for a given JD + resume batch.

        1. Preprocess resumes → clean text
        2. Rebuild or reuse index
        3. Retrieve top-K candidates (Stage 1)
        4. Rerank with cross-encoder or LLM (Stage 2)
        5. Return ranked results
        """
        timing: dict[str, float] = {}
        n = top_n or self.cfg.reranker.top_n

        # ── 1. preprocess ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        clean_jd = self.preprocessor.process_job_description(job_description)
        clean_resumes = [
            IndexEntry(
                resume_id=r.resume_id,
                text=self.preprocessor.process(r.content, r.filename),
                metadata=r.metadata,
            )
            for r in resumes
        ]
        timing["preprocess_s"] = round(time.perf_counter() - t0, 3)
        logger.info(
            "Preprocessed %d resumes in %.2fs", len(resumes), timing["preprocess_s"]
        )

        # ── 2. build / refresh index ───────────────────────────────────────
        t0 = time.perf_counter()
        self.retriever.build_index(clean_resumes)
        timing["index_build_s"] = round(time.perf_counter() - t0, 3)

        # ── 3. retrieve ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        top_k        = min(self.cfg.retriever.top_k, len(resumes))
        candidates   = self.retriever.retrieve(clean_jd, top_k=top_k)
        timing["retrieval_s"] = round(time.perf_counter() - t0, 3)
        logger.info(
            "Retrieved %d candidates in %.2fs", len(candidates), timing["retrieval_s"]
        )

        # ── 4. rerank ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = self.reranker.rerank(clean_jd, candidates, top_n=n)
        timing["rerank_s"] = round(time.perf_counter() - t0, 3)
        logger.info(
            "Reranked → %d results in %.2fs", len(results), timing["rerank_s"]
        )

        timing["total_s"] = round(
            sum(v for v in timing.values()), 3
        )

        # ── 5. optional eval ───────────────────────────────────────────────
        metrics = None
        if relevant_ids:
            ranked_ids = [r.resume_id for r in results]
            metrics = evaluate(relevant_ids, ranked_ids)

        return RankResponse(
            job_description=job_description,
            results=results,
            total_candidates=len(candidates),
            timing=timing,
            metrics=metrics,
        )

    async def rank_async(
        self,
        job_description: str,
        resumes: list[ResumeInput],
        top_n: Optional[int] = None,
    ) -> RankResponse:
        """Non-blocking wrapper for FastAPI async endpoints."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rank(job_description, resumes, top_n),
        )

    def load_or_build(self, resumes: list[ResumeInput]) -> None:
        """Try loading a persisted index; build fresh if missing."""
        if not self.retriever.load_index():
            logger.info("No persisted index. Building from %d resumes.", len(resumes))
            self.build_index(resumes)
            self.retriever.save_index()

    def save_index(self) -> None:
        self.retriever.save_index()
