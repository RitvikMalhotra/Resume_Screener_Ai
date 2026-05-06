"""
Phase 4 – Async Job Queue

Handles background processing of batch ranking jobs.

Architecture
------------
  - In-process job store (dict) — fine for single worker
  - Jobs run in a ThreadPoolExecutor so FastAPI stays responsive
  - Each job has: queued → running → done | error states
  - Results stored in memory with TTL cleanup

Interview talking point:
  "For batch workloads we use an async job queue so the client
   doesn't have to hold the connection open for 2 minutes.
   They POST the job, get a job_id, and poll /jobs/{id} for status.
   In production this would be Celery + Redis or AWS SQS."
"""
from __future__ import annotations
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Job state ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    job_id: str
    status: str = "queued"          # queued | running | done | error
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    n_resumes: int = 0
    n_jobs: int = 1

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.finished_at:
            return round(self.finished_at - self.created_at, 3)
        if self.started_at:
            return round(time.time() - self.started_at, 3)
        return None

    @property
    def wait_s(self) -> Optional[float]:
        if self.started_at:
            return round(self.started_at - self.created_at, 3)
        return round(time.time() - self.created_at, 3)

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "status":      self.status,
            "created_at":  self.created_at,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s":   self.elapsed_s,
            "wait_s":      self.wait_s,
            "n_resumes":   self.n_resumes,
            "n_jobs":      self.n_jobs,
            "error":       self.error,
        }

    def to_dict_with_result(self) -> dict:
        d = self.to_dict()
        d["result"] = self.result
        return d


# ── Job store ──────────────────────────────────────────────────────────────

class JobStore:
    """
    In-memory job store with TTL-based cleanup.
    Jobs older than ttl_seconds are removed automatically.
    """

    def __init__(self, ttl_seconds: int = 3600, max_jobs: int = 500):
        self._jobs: dict[str, Job] = {}
        self._ttl  = ttl_seconds
        self._max  = max_jobs

    def create(self, n_resumes: int = 0, n_jobs: int = 1) -> Job:
        self._cleanup()
        if len(self._jobs) >= self._max:
            raise RuntimeError(
                f"Job queue full ({self._max} jobs). Try again later."
            )
        job = Job(
            job_id   = str(uuid.uuid4()),
            n_resumes = n_resumes,
            n_jobs   = n_jobs,
        )
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def _cleanup(self) -> None:
        now     = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if (now - job.created_at) > self._ttl
        ]
        for jid in expired:
            del self._jobs[jid]
        if expired:
            logger.debug("Cleaned up %d expired jobs", len(expired))

    def stats(self) -> dict:
        statuses = {}
        for job in self._jobs.values():
            statuses[job.status] = statuses.get(job.status, 0) + 1
        return {
            "total_jobs": len(self._jobs),
            "by_status":  statuses,
        }


# ── Job runner ─────────────────────────────────────────────────────────────

class JobRunner:
    """
    Executes jobs in a background thread pool.
    FastAPI endpoint submits a job and returns immediately.
    Client polls /jobs/{job_id} for completion.
    """

    def __init__(self, max_workers: int = 2):
        self._pool  = ThreadPoolExecutor(max_workers=max_workers)
        self._store = JobStore()

    @property
    def store(self) -> JobStore:
        return self._store

    def submit(self, fn, *args, n_resumes: int = 0, n_jobs: int = 1, **kwargs) -> Job:
        """
        Submit a function to run in the background.
        Returns a Job immediately.
        """
        job = self._store.create(n_resumes=n_resumes, n_jobs=n_jobs)

        def _run():
            job.status     = "running"
            job.started_at = time.time()
            logger.info("Job %s started", job.job_id)
            try:
                result        = fn(*args, **kwargs)
                job.result    = result
                job.status    = "done"
                logger.info("Job %s done in %.2fs", job.job_id, job.elapsed_s)
            except Exception as exc:
                job.error  = str(exc)
                job.status = "error"
                logger.error("Job %s failed: %s", job.job_id, exc)
            finally:
                job.finished_at = time.time()

        self._pool.submit(_run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._store.get(job_id)

    def stats(self) -> dict:
        return self._store.stats()