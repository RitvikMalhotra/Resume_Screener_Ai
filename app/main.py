"""
FastAPI application — Phase 1 + 2 + 3 + 4

Phase 1: Batch inference + embedding cache
Phase 2: Precision@K / NDCG evaluation with graded reports
Phase 3: Confidence scoring + keyword fallback + input validation
Phase 4: Async batch jobs + status polling + rate limiting

Confidence penalty sorting:
  Results are re-sorted after confidence scoring.
  Uncertain results are penalized so high-confidence
  candidates always surface to the top.
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.pipeline import Pipeline, ResumeInput
from app.job_queue import JobRunner
from app.rate_limiter import RateLimiter, RateLimitConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────
_pipeline: Optional[Pipeline] = None
_runner   = JobRunner(max_workers=2)
_limiter  = RateLimiter(RateLimitConfig(
    requests_per_minute       = 30,
    requests_per_minute_light = 120,
))

# Confidence penalty — subtracted from final_score before re-sorting
_CONFIDENCE_PENALTY = {
    "high":      0.00,
    "medium":    0.05,
    "low":       0.15,
    "uncertain": 0.25,
}


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up pipeline…")
    get_pipeline()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Resume Screening API",
    version="4.0.0",
    description=(
        "Two-stage LLM-powered resume ranking.\n\n"
        "**Phase 1**: Batch inference + embedding cache\n\n"
        "**Phase 2**: Precision@K / NDCG evaluation\n\n"
        "**Phase 3**: Confidence scoring + keyword fallback + validation\n\n"
        "**Phase 4**: Async batch jobs + status polling + rate limiting"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class ResumeItem(BaseModel):
    resume_id: str
    text: str
    metadata: dict = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resume text cannot be empty")
        return v


class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    resumes: list[ResumeItem] = Field(..., min_length=1, max_length=500)
    top_n: int = Field(default=10, ge=1, le=100)
    relevant_ids: Optional[list[str]] = None
    eval_k_values: list[int] = Field(default=[1, 3, 5, 10])
    skip_validation: bool = False


class BatchRankQuery(BaseModel):
    job_description: str = Field(..., min_length=20)
    resumes: list[ResumeItem] = Field(..., min_length=1, max_length=500)
    top_n: int = Field(default=10, ge=1, le=100)
    relevant_ids: Optional[list[str]] = None


class BatchRankRequest(BaseModel):
    queries: list[BatchRankQuery] = Field(..., min_length=1, max_length=20)
    eval_k_values: list[int] = Field(default=[1, 3, 5, 10])


class RankedResumeItem(BaseModel):
    resume_id: str
    rank: int
    embedding_score: float
    rerank_score: float
    final_score: float
    text_snippet: str
    confidence: Optional[dict] = None
    fallback: Optional[dict] = None
    fallback_used: bool = False


class RankResponse(BaseModel):
    results: list[RankedResumeItem]
    total_candidates: int
    timing: dict[str, float]
    faiss_info: Optional[dict] = None
    metrics_raw: Optional[dict] = None
    metrics_report: Optional[dict] = None
    validation: Optional[dict] = None


class IndexBuildRequest(BaseModel):
    resumes: list[ResumeItem] = Field(..., min_length=1, max_length=10000)
    persist: bool = True


class WarmCacheRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=5000)


class EvalRequest(BaseModel):
    job_description: str
    resumes: list[ResumeItem]
    relevant_ids: list[str]
    k_values: list[int] = Field(default=[1, 3, 5, 10])
    primary_k: int = Field(default=5)


class BatchEvalQuery(BaseModel):
    job_description: str
    resumes: list[ResumeItem]
    relevant_ids: list[str]


class BatchEvalRequest(BaseModel):
    queries: list[BatchEvalQuery] = Field(..., min_length=1, max_length=50)
    k_values: list[int] = Field(default=[1, 3, 5, 10])


# ── Middleware ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── Confidence penalty re-sort ─────────────────────────────────────────────

def apply_confidence_penalty_and_sort(results, confidences, result_dicts):
    """
    Sort by confidence tier first, then by final_score within each tier.
    Tier order: high (0) → medium (1) → low (2) → uncertain (3)
    """
    _TIER = {"high": 0, "medium": 1, "low": 2, "uncertain": 3}

    combined = list(zip(results, confidences, result_dicts))
    combined.sort(key=lambda x: (
        _TIER.get(x[1].level, 3),   # primary: confidence tier
        -x[2]["final_score"],        # secondary: score descending
    ))

    results_s      = [c[0] for c in combined]
    confidences_s  = [c[1] for c in combined]
    result_dicts_s = [c[2] for c in combined]

    for i, r in enumerate(results_s):
        r.rank = i

    return results_s, confidences_s, result_dicts_s


# ── Core ranking logic (reused by /rank and /rank/batch) ──────────────────

def _run_rank(
    pipeline: Pipeline,
    job_description: str,
    resumes: list[ResumeItem],
    top_n: int,
    relevant_ids: Optional[list[str]],
    eval_k_values: list[int],
    skip_validation: bool = False,
) -> dict:
    from app.validator  import InputValidator
    from app.confidence import score_confidence
    from app.fallback   import KeywordFallback, apply_fallback

    # Phase 3: validate
    validation_result = None
    resumes_to_use    = resumes

    if not skip_validation:
        validator = InputValidator()
        val       = validator.validate(
            jd_text = job_description,
            resumes = [{"resume_id": r.resume_id, "text": r.text} for r in resumes],
        )
        validation_result = val.to_dict()
        if not val.is_valid:
            raise ValueError(f"Validation failed: {validation_result}")
        valid_ids      = {r["resume_id"] for r in val.cleaned_resumes}
        resumes_to_use = [r for r in resumes if r.resume_id in valid_ids]

    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in resumes_to_use
    ]

    import asyncio
    loop     = asyncio.new_event_loop()
    response = loop.run_until_complete(
        pipeline.rank_async(
            job_description = job_description,
            resumes         = resume_inputs,
            top_n           = top_n,
        )
    )
    loop.close()

    # Phase 3: confidence scoring
    final_scores = [r.final_score for r in response.results]
    confidences  = score_confidence(final_scores)

    # Phase 3: keyword fallback
    result_dicts = [
        {"resume_id": r.resume_id, "final_score": r.final_score, "resume_text": r.text}
        for r in response.results
    ]
    result_dicts = apply_fallback(job_description, result_dicts, KeywordFallback())

    # Re-score confidences after fallback may have changed final scores
    updated_scores = [rd["final_score"] for rd in result_dicts]
    confidences    = score_confidence(updated_scores)

    # Confidence penalty re-sort — high confidence bubbles to top
    results_sorted, confidences_sorted, result_dicts_sorted = \
        apply_confidence_penalty_and_sort(
            list(response.results), confidences, result_dicts
        )

    # Phase 2: metrics
    metrics_raw = metrics_report = None
    if relevant_ids:
        from app.metrics import evaluate_with_report
        eval_result    = evaluate_with_report(
            relevant_ids = set(relevant_ids),
            ranked_ids   = [r.resume_id for r in results_sorted],
            k_values     = eval_k_values,
        )
        metrics_raw    = eval_result["raw"]
        metrics_report = eval_result["report"]

    info = pipeline.retriever.index_info()

    return {
        "results": [
            {
                "resume_id":       r.resume_id,
                "rank":            r.rank,
                "embedding_score": round(r.embedding_score, 4),
                "rerank_score":    round(r.rerank_score, 4),
                "final_score":     round(result_dicts_sorted[i]["final_score"], 4),
                "text_snippet":    r.text[:300],
                "confidence":      confidences_sorted[i].to_dict(),
                "fallback":        result_dicts_sorted[i].get("fallback"),
                "fallback_used":   result_dicts_sorted[i].get("fallback_used", False),
            }
            for i, r in enumerate(results_sorted)
        ],
        "total_candidates": response.total_candidates,
        "timing":           response.timing,
        "faiss_info":       info.to_dict() if info else None,
        "metrics_raw":      metrics_raw,
        "metrics_report":   metrics_report,
        "validation":       validation_result,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    _limiter.check(request, heavy=False)
    pipeline = get_pipeline()
    info     = pipeline.retriever.index_info()
    return {
        "status":    "ok",
        "version":   "4.0.0",
        "index":     info.to_dict() if info else None,
        "cache":     pipeline.retriever.cache_stats(),
        "jobs":      _runner.stats(),
        "ratelimit": _limiter.stats(),
    }


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(request: Request, body: RankRequest):
    """
    Rank resumes against a job description.

    Results are sorted by confidence-penalized score:
      High confidence → top
      Uncertain       → bottom
    """
    from app.validator  import InputValidator
    from app.confidence import score_confidence
    from app.fallback   import KeywordFallback, apply_fallback

    headers  = _limiter.check(request, heavy=True)
    pipeline = get_pipeline()

    # Phase 3: validate
    validation_result = None
    resumes_to_use    = body.resumes

    if not body.skip_validation:
        validator = InputValidator()
        val       = validator.validate(
            jd_text = body.job_description,
            resumes = [{"resume_id": r.resume_id, "text": r.text} for r in body.resumes],
        )
        validation_result = val.to_dict()
        if not val.is_valid:
            raise HTTPException(
                status_code=422,
                detail={"message": "Validation failed.", "validation": validation_result},
            )
        valid_ids      = {r["resume_id"] for r in val.cleaned_resumes}
        resumes_to_use = [r for r in body.resumes if r.resume_id in valid_ids]

    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in resumes_to_use
    ]

    try:
        response = await pipeline.rank_async(
            job_description = body.job_description,
            resumes         = resume_inputs,
            top_n           = body.top_n,
        )
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Phase 3: confidence + fallback
    final_scores = [r.final_score for r in response.results]
    confidences  = score_confidence(final_scores)
    result_dicts = [
        {"resume_id": r.resume_id, "final_score": r.final_score, "resume_text": r.text}
        for r in response.results
    ]
    result_dicts = apply_fallback(body.job_description, result_dicts, KeywordFallback())

    # Re-score after fallback
    updated_scores = [rd["final_score"] for rd in result_dicts]
    confidences    = score_confidence(updated_scores)

    # Confidence penalty re-sort
    results_sorted, confidences_sorted, result_dicts_sorted = \
        apply_confidence_penalty_and_sort(
            list(response.results), confidences, result_dicts
        )

    # Phase 2: metrics
    metrics_raw = metrics_report = None
    if body.relevant_ids:
        from app.metrics import evaluate_with_report
        eval_result    = evaluate_with_report(
            relevant_ids = set(body.relevant_ids),
            ranked_ids   = [r.resume_id for r in results_sorted],
            k_values     = body.eval_k_values,
        )
        metrics_raw    = eval_result["raw"]
        metrics_report = eval_result["report"]

    info = pipeline.retriever.index_info()
    result = {
        "results": [
            {
                "resume_id":       r.resume_id,
                "rank":            r.rank,
                "embedding_score": round(r.embedding_score, 4),
                "rerank_score":    round(r.rerank_score, 4),
                "final_score":     round(result_dicts_sorted[i]["final_score"], 4),
                "text_snippet":    r.text[:300],
                "confidence":      confidences_sorted[i].to_dict(),
                "fallback":        result_dicts_sorted[i].get("fallback"),
                "fallback_used":   result_dicts_sorted[i].get("fallback_used", False),
            }
            for i, r in enumerate(results_sorted)
        ],
        "total_candidates": response.total_candidates,
        "timing":           response.timing,
        "faiss_info":       info.to_dict() if info else None,
        "metrics_raw":      metrics_raw,
        "metrics_report":   metrics_report,
        "validation":       validation_result,
    }

    api_response = JSONResponse(content=result)
    for k, v in headers.items():
        api_response.headers[k] = v
    return api_response


@app.post("/rank/batch")
async def rank_batch(request: Request, body: BatchRankRequest):
    """
    Submit multiple ranking jobs at once. Returns job_id immediately.
    Poll GET /jobs/{job_id} for status and results.
    """
    _limiter.check(request, heavy=True)
    pipeline = get_pipeline()

    total_resumes = sum(len(q.resumes) for q in body.queries)

    def _batch_fn():
        results = []
        for q in body.queries:
            r = _run_rank(
                pipeline        = pipeline,
                job_description = q.job_description,
                resumes         = q.resumes,
                top_n           = q.top_n,
                relevant_ids    = q.relevant_ids,
                eval_k_values   = body.eval_k_values,
            )
            results.append(r)
        return results

    job = _runner.submit(
        _batch_fn,
        n_resumes = total_resumes,
        n_jobs    = len(body.queries),
    )

    return {
        "job_id":    job.job_id,
        "status":    job.status,
        "n_queries": len(body.queries),
        "n_resumes": total_resumes,
        "poll_url":  f"/jobs/{job.job_id}",
        "message":   "Job submitted. Poll /jobs/{job_id} for status.",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    """Poll job status. Returns result when status = 'done'."""
    _limiter.check(request, heavy=False)
    job = _runner.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status == "done":
        return job.to_dict_with_result()
    return job.to_dict()


@app.get("/jobs")
async def list_jobs(request: Request):
    """List all jobs and queue stats."""
    _limiter.check(request, heavy=False)
    return {
        "queue_stats": _runner.stats(),
        "rate_limit":  _limiter.stats(),
    }


@app.post("/index/build")
async def build_index(
    request: Request,
    body: IndexBuildRequest,
    background_tasks: BackgroundTasks,
):
    _limiter.check(request, heavy=True)
    pipeline = get_pipeline()
    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in body.resumes
    ]
    t0 = time.perf_counter()
    try:
        pipeline.build_index(resume_inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed = time.perf_counter() - t0
    if body.persist:
        background_tasks.add_task(pipeline.save_index)
    info = pipeline.retriever.index_info()
    return {
        "status":     "ok",
        "indexed":    len(resume_inputs),
        "elapsed_s":  round(elapsed, 3),
        "faiss_info": info.to_dict() if info else None,
    }


@app.get("/index/stats")
async def index_stats(request: Request):
    _limiter.check(request, heavy=False)
    pipeline = get_pipeline()
    info     = pipeline.retriever.index_info()
    return {
        "faiss": info.to_dict() if info else None,
        "cache": pipeline.retriever.cache_stats(),
    }


@app.post("/cache/warm")
async def warm_cache(request: Request, body: WarmCacheRequest):
    _limiter.check(request, heavy=False)
    pipeline = get_pipeline()
    result   = pipeline.retriever.warm_cache(body.texts)
    return {"status": "ok", **result}


@app.post("/evaluate")
async def evaluate_endpoint(request: Request, body: EvalRequest):
    """Evaluate ranking quality against labelled ground truth."""
    _limiter.check(request, heavy=True)
    from app.metrics import evaluate_with_report
    pipeline = get_pipeline()
    resume_inputs = [
        ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
        for r in body.resumes
    ]
    response = await pipeline.rank_async(
        job_description = body.job_description,
        resumes         = resume_inputs,
        top_n           = len(body.resumes),
    )
    ranked_ids  = [r.resume_id for r in response.results]
    eval_result = evaluate_with_report(
        relevant_ids = set(body.relevant_ids),
        ranked_ids   = ranked_ids,
        k_values     = body.k_values,
        primary_k    = body.primary_k,
    )
    return {
        "raw":        eval_result["raw"],
        "report":     eval_result["report"],
        "timing":     response.timing,
        "faiss_info": pipeline.retriever.index_info().to_dict()
                      if pipeline.retriever.index_info() else None,
    }


@app.post("/evaluate/batch")
async def batch_evaluate_endpoint(request: Request, body: BatchEvalRequest):
    """Averaged metrics across multiple queries."""
    _limiter.check(request, heavy=True)
    from app.metrics import batch_evaluate
    pipeline    = get_pipeline()
    all_queries = []
    for q in body.queries:
        resume_inputs = [
            ResumeInput(resume_id=r.resume_id, content=r.text, metadata=r.metadata)
            for r in q.resumes
        ]
        response = await pipeline.rank_async(
            job_description = q.job_description,
            resumes         = resume_inputs,
            top_n           = len(q.resumes),
        )
        all_queries.append({
            "relevant_ids": set(q.relevant_ids),
            "ranked_ids":   [r.resume_id for r in response.results],
        })
    averaged = batch_evaluate(all_queries, k_values=body.k_values)
    return {
        "averaged_metrics": averaged,
        "n_queries":        len(body.queries),
    }


@app.post("/explain")
async def explain_candidate(request: Request, body: dict):
    """
    Proxy NVIDIA NIM API call server-side to avoid CORS.
    Generates plain English explanation for a ranked candidate.
    """
    import httpx
    _limiter.check(request, heavy=False)

    NVIDIA_KEY   = "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"
    NVIDIA_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
    NVIDIA_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

    jd           = body.get("jd", "")[:800]
    resume_id    = body.get("resume_id", "")
    snippet      = body.get("snippet", "")[:600]
    rank         = body.get("rank", 0)
    final_score  = body.get("final_score", 0)
    embed_score  = body.get("embedding_score", 0)
    rerank_score = body.get("rerank_score", 0)
    confidence   = body.get("confidence", "unknown")

    prompt = f"""You are an expert technical recruiter. Analyze why this candidate ranked #{rank + 1}.

JOB DESCRIPTION:
{jd}

CANDIDATE ({resume_id}):
{snippet}

SCORES: Final={round(final_score*100)}% | Semantic={round(embed_score*100)}% | Rerank={round(rerank_score*100)}% | Confidence={confidence}

Write ONE concise paragraph (3-4 sentences) explaining:
1. Specific skills/experience that matched the JD
2. Why they ranked #{rank + 1}
3. The main gap or risk if any

Be specific. Mention actual skills from their resume. No bullet points. Do not start with "This candidate"."""

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.4,
        "top_p": 0.9,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                NVIDIA_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            res.raise_for_status()
            data = res.json()
            text = data["choices"][0]["message"]["content"].strip()
            return {"explanation": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.get("/metrics/cache")
async def cache_metrics(request: Request):
    _limiter.check(request, heavy=False)
    pipeline = get_pipeline()
    return pipeline.retriever.cache_stats()