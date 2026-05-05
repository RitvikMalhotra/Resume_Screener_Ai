"""
FastAPI application — resume screening service.

Endpoints
---------
  POST /rank              rank a batch of resumes against a job description
  POST /index/build       build / rebuild the FAISS index
  GET  /index/stats       index metadata
  POST /evaluate          compute metrics against a labelled test set
  GET  /health            liveness check
  GET  /metrics/cache     cache hit rate

Request limits
--------------
  - 500 resumes per request (configurable)
  - 2-minute timeout enforced server-side
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.pipeline import Pipeline, ResumeInput
from utils.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── global pipeline singleton ──────────────────────────────────────────────
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm the pipeline (loads models into memory)."""
    logger.info("Warming up pipeline …")
    get_pipeline()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Screening API",
    version="1.0.0",
    description="Two-stage LLM-powered resume ranking: bi-encoder retrieval + reranking.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────

class ResumeItem(BaseModel):
    resume_id: str = Field(..., description="Unique identifier for the resume")
    text: str      = Field(..., description="Resume text content")
    metadata: dict = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resume text cannot be empty")
        return v


class RankRequest(BaseModel):
    job_description: str = Field(
        ..., min_length=50, description="Full job description text"
    )
    resumes: list[ResumeItem] = Field(
        ..., min_length=1, max_length=500,
        description="Up to 500 resumes to rank"
    )
    top_n: int = Field(default=10, ge=1, le=100, description="Number of top results")
    relevant_ids: Optional[list[str]] = Field(
        default=None, description="Optional ground-truth relevant IDs for eval"
    )


class RankedResumeItem(BaseModel):
    resume_id: str
    rank: int
    embedding_score: float
    rerank_score: float
    final_score: float
    text_snippet: str   # first 300 chars


class RankResponse(BaseModel):
    results: list[RankedResumeItem]
    total_candidates: int
    timing: dict[str, float]
    metrics: Optional[dict[str, float]] = None


class IndexBuildRequest(BaseModel):
    resumes: list[ResumeItem] = Field(..., min_length=1, max_length=10000)
    persist: bool = Field(default=True, description="Save index to disk after build")


class EvalRequest(BaseModel):
    job_description: str
    resumes: list[ResumeItem]
    relevant_ids: list[str]
    k_values: list[int] = Field(default=[1, 3, 5, 10])


# ── middleware ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "index_size": pipeline.retriever.index_size,
    }


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(request: RankRequest):
    """
    Rank resumes against a job description using two-stage pipeline.

    Stage 1: bi-encoder embedding + FAISS retrieval (top-K candidates)
    Stage 2: cross-encoder or LLM reranking of candidates
    """
    pipeline = get_pipeline()

    resume_inputs = [
        ResumeInput(
            resume_id=r.resume_id,
            content=r.text,
            metadata=r.metadata,
        )
        for r in request.resumes
    ]

    relevant_set = set(request.relevant_ids) if request.relevant_ids else None

    try:
        response = await pipeline.rank_async(
            job_description=request.job_description,
            resumes=resume_inputs,
            top_n=request.top_n,
        )
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # attach eval metrics if ground truth provided
    metrics = None
    if relevant_set:
        from app.metrics import evaluate
        ranked_ids = [r.resume_id for r in response.results]
        metrics = evaluate(relevant_set, ranked_ids)

    return RankResponse(
        results=[
            RankedResumeItem(
                resume_id=r.resume_id,
                rank=r.rank,
                embedding_score=round(r.embedding_score, 4),
                rerank_score=round(r.rerank_score, 4),
                final_score=round(r.final_score, 4),
                text_snippet=r.text[:300],
            )
            for r in response.results
        ],
        total_candidates=response.total_candidates,
        timing=response.timing,
        metrics=metrics,
    )


@app.post("/index/build")
async def build_index(
    request: IndexBuildRequest,
    background_tasks: BackgroundTasks,
):
    """Build or rebuild the FAISS index from a corpus of resumes."""
    pipeline = get_pipeline()
    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in request.resumes
    ]

    t0 = time.perf_counter()
    try:
        pipeline.build_index(resume_inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0

    if request.persist:
        background_tasks.add_task(pipeline.save_index)

    return {
        "status": "ok",
        "indexed": len(resume_inputs),
        "elapsed_s": round(elapsed, 3),
    }


@app.get("/index/stats")
async def index_stats():
    pipeline = get_pipeline()
    return {
        "index_size": pipeline.retriever.index_size,
        "cache_stats": pipeline.retriever.cache_stats(),
    }


@app.post("/evaluate")
async def evaluate_endpoint(request: EvalRequest):
    """Compute ranking metrics against a labelled test set."""
    from app.metrics import evaluate

    pipeline = get_pipeline()
    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in request.resumes
    ]

    response = await pipeline.rank_async(
        job_description=request.job_description,
        resumes=resume_inputs,
        top_n=len(request.resumes),
    )

    ranked_ids   = [r.resume_id for r in response.results]
    relevant_set = set(request.relevant_ids)
    metrics      = evaluate(relevant_set, ranked_ids, k_values=request.k_values)

    return {
        "metrics": metrics,
        "timing": response.timing,
        "n_relevant": len(relevant_set),
        "n_ranked": len(ranked_ids),
    }


@app.get("/metrics/cache")
async def cache_metrics():
    pipeline = get_pipeline()
    return pipeline.retriever.cache_stats()
