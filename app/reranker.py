"""
Stage 2 – LLM reranker.

Two implementations, selected at runtime via config:

A) LLMReranker (primary)
   - LLaMA 3 8B Instruct loaded with bitsandbytes 4-bit quantization
   - Optional LoRA weights loaded via peft
   - Scores candidate pairs (JD, resume) using log-prob of "yes" token
   - Batch inference with padding / left-truncation

B) CrossEncoderReranker (fallback, no GPU required)
   - sentence-transformers cross-encoder
   - Significantly faster; good enough for most use cases
   - Default: cross-encoder/ms-marco-MiniLM-L-12-v2

Both expose the same interface: .rerank(jd, candidates) → list[RerankResult]
"""
from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER = True
except ImportError:
    _CROSS_ENCODER = False

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None
    _TORCH = False

from app.retriever import RetrievalResult
from utils.config import RerankerConfig, get_config
from utils.batching import chunks


@dataclass
class RerankResult:
    resume_id: str
    text: str
    embedding_score: float     # from Stage 1
    rerank_score: float        # from Stage 2
    final_score: float         # weighted combination
    rank: int


class CrossEncoderReranker:
    """
    Fast CPU/GPU cross-encoder reranker.
    Uses sentence-transformers CrossEncoder (BERT-based).
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.cfg = config or get_config().reranker
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> "CrossEncoder":
        if self._model is None:
            if not _CROSS_ENCODER:
                raise RuntimeError("sentence-transformers required")
            logger.info("Loading cross-encoder: %s", self.cfg.cross_encoder_model)
            self._model = CrossEncoder(
                self.cfg.cross_encoder_model,
                max_length=self.cfg.max_rerank_len,
            )
        return self._model

    def rerank(
        self,
        job_description: str,
        candidates: list[RetrievalResult],
        top_n: Optional[int] = None,
        alpha: float = 0.3,      # weight for embedding score in final score
    ) -> list[RerankResult]:
        """
        Rerank candidates. Returns top_n results sorted by final_score.

        final_score = alpha * embed_score + (1 - alpha) * rerank_score
        """
        if not candidates:
            return []

        n = top_n or self.cfg.top_n
        if not _CROSS_ENCODER or np is None:
            ranked = sorted(candidates, key=lambda candidate: candidate.embedding_score, reverse=True)
            return [
                RerankResult(
                    resume_id=c.resume_id,
                    text=c.text,
                    embedding_score=c.embedding_score,
                    rerank_score=c.embedding_score,
                    final_score=c.embedding_score,
                    rank=rank,
                )
                for rank, c in enumerate(ranked[:n])
            ]

        model = self._get_model()
        pairs  = [(job_description, c.text) for c in candidates]

        # batch scoring
        scores: list[float] = []
        t0 = time.perf_counter()
        for batch in chunks(pairs, self.cfg.rerank_batch_size):
            batch_scores = model.predict(batch, convert_to_numpy=True)
            # Normalize to [0, 1] via sigmoid
            normalized = 1 / (1 + np.exp(-batch_scores))
            scores.extend(normalized.tolist())
        elapsed = time.perf_counter() - t0
        logger.debug(
            "Cross-encoder scored %d pairs in %.2fs", len(pairs), elapsed
        )

        # combine scores
        embed_scores = np.array([c.embedding_score for c in candidates])
        # normalize embedding scores to [0, 1] (they're already cosine sims)
        embed_norm = (embed_scores - embed_scores.min()) / (
            (embed_scores.max() - embed_scores.min())+ 1e-9
        )
        rerank_arr  = np.array(scores)
        final_arr   = alpha * embed_norm + (1 - alpha) * rerank_arr

        # sort descending
        sorted_idx = np.argsort(-final_arr)[:n]
        results: list[RerankResult] = []
        for rank, idx in enumerate(sorted_idx):
            c = candidates[idx]
            results.append(
                RerankResult(
                    resume_id=c.resume_id,
                    text=c.text,
                    embedding_score=c.embedding_score,
                    rerank_score=float(rerank_arr[idx]),
                    final_score=float(final_arr[idx]),
                    rank=rank,
                )
            )
        return results


class LLMReranker:
    """
    LLaMA 3 8B Instruct reranker with optional LoRA weights.

    Scoring method: log-probability of token "Yes" (relevant) vs "No"
    given the prompt: "Is this resume relevant for the job? Yes/No"

    Requirements
    ------------
    pip install transformers peft bitsandbytes accelerate

    GPU memory (4-bit quantized LLaMA 3 8B): ~5–6 GB VRAM
    """

    _PROMPT_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert technical recruiter. Evaluate resume-job fit.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "Job Description:\n{jd}\n\n"
        "Resume:\n{resume}\n\n"
        "Is this resume a strong match for the job? Answer with only 'Yes' or 'No'.\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.cfg = config or get_config().reranker
        self._model  = None
        self._tokenizer = None
        self._yes_id = None
        self._no_id  = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if not _TORCH:
            raise RuntimeError("PyTorch is required for LLM reranker")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info("Loading LLM reranker: %s", self.cfg.llm_model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.llm_model_name, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"   # better for causal LM batch inference

        quant_cfg = None
        if self.cfg.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.llm_model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # load LoRA weights if provided
        if self.cfg.lora_weights_path:
            from peft import PeftModel
            logger.info("Loading LoRA weights from %s", self.cfg.lora_weights_path)
            model = PeftModel.from_pretrained(model, self.cfg.lora_weights_path)
            model = model.merge_and_unload()   # fuse for faster inference

        model.eval()
        self._model     = model
        self._tokenizer = tokenizer

        # token IDs for "Yes" and "No"
        self._yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]
        logger.info(
            "LLM reranker ready (yes_id=%d, no_id=%d)",
            self._yes_id, self._no_id,
        )

    def _score_batch(self, prompts: list[str]) -> np.ndarray:
        """Return P(Yes) for each prompt as float array."""
        import torch
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_rerank_len,
        ).to(self._model.device)

        outputs = self._model(**inputs)
        # logits at the last token position: (batch, vocab)
        last_logits = outputs.logits[:, -1, :]
        log_probs   = torch.nn.functional.log_softmax(last_logits, dim=-1)
        yes_lp = log_probs[:, self._yes_id].cpu().numpy()
        no_lp  = log_probs[:, self._no_id ].cpu().numpy()
        # normalized probability: exp(yes) / (exp(yes) + exp(no))
        p_yes  = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp) + 1e-9)
        return p_yes.astype(np.float32)

    def rerank(
        self,
        job_description: str,
        candidates: list[RetrievalResult],
        top_n: Optional[int] = None,
        alpha: float = 0.2,
    ) -> list[RerankResult]:
        self._load()

        if not candidates:
            return []

        n = top_n or self.cfg.top_n
        # truncate JD and resume to fit prompt window
        jd_short = job_description[:1500]

        prompts = [
            self._PROMPT_TEMPLATE.format(
                jd=jd_short,
                resume=c.text[:1500],
            )
            for c in candidates
        ]

        scores: list[float] = []
        t0 = time.perf_counter()
        for batch_prompts in chunks(prompts, self.cfg.rerank_batch_size):
            p_yes = self._score_batch(batch_prompts)
            scores.extend(p_yes.tolist())
        elapsed = time.perf_counter() - t0
        logger.info("LLM scored %d pairs in %.2fs", len(prompts), elapsed)

        embed_scores = np.array([c.embedding_score for c in candidates])
        embed_norm   = (embed_scores - embed_scores.min()) / (
            (embed_scores.max() - embed_scores.min()) + 1e-9
        )
        rerank_arr   = np.array(scores)
        final_arr    = alpha * embed_norm + (1 - alpha) * rerank_arr

        sorted_idx = np.argsort(-final_arr)[:n]
        results: list[RerankResult] = []
        for rank, idx in enumerate(sorted_idx):
            c = candidates[idx]
            results.append(
                RerankResult(
                    resume_id=c.resume_id,
                    text=c.text,
                    embedding_score=c.embedding_score,
                    rerank_score=float(rerank_arr[idx]),
                    final_score=float(final_arr[idx]),
                    rank=rank,
                )
            )
        return results


class HostedLLMReranker:
    """
    Stage 2 reranking via the hosted LLM API (app.llm).

    This is what runs on serverless deployments, where torch /
    sentence-transformers can't be installed inside the function size limit.
    Without it the pipeline falls back to lexical word overlap, which produces
    meaningless scores (a perfectly matching resume lands around 0.2 simply
    because it doesn't reuse the JD's exact vocabulary).

    All candidates are scored in a single request -- one call per candidate
    would blow the serverless execution window.
    """

    _MAX_CANDIDATES  = 20     # cap prompt size / latency
    _MAX_RESUME_CHARS = 1200
    _MAX_JD_CHARS     = 2000

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.cfg = config or get_config().reranker

    def _build_prompt(self, job_description: str, candidates: list[RetrievalResult]) -> str:
        blocks = []
        for i, c in enumerate(candidates):
            blocks.append(f"[{i}]\n{c.text[:self._MAX_RESUME_CHARS]}")
        candidate_text = "\n\n".join(blocks)

        return f"""You are an expert technical recruiter scoring resumes against a job description.

JOB DESCRIPTION:
{job_description[:self._MAX_JD_CHARS]}

CANDIDATES:
{candidate_text}

Score every candidate from 0 to 100 on how well they match the job description:
- 90-100: excellent match, meets essentially all requirements
- 70-89:  strong match, meets most key requirements
- 40-69:  partial match, some relevant skills but notable gaps
- 10-39:  weak match, little meaningful overlap
- 0-9:    irrelevant to this role

Judge on actual skills, seniority and domain experience -- not on whether the
resume happens to reuse the same words as the job description.

Return ONLY a JSON array with one object per candidate, no markdown, no explanation:
[{{"index": 0, "score": 85}}, {{"index": 1, "score": 20}}]"""

    def _score_candidates(self, job_description: str, candidates: list[RetrievalResult]) -> dict[int, float]:
        """Returns {candidate_index: score in [0,1]}. Raises LLMError on failure."""
        from app import llm

        # Single attempt on a short leash: /rank has a whole pipeline to get
        # through, so it's better to fall back to retrieval order quickly than
        # to retry and risk the serverless execution limit.
        raw = _run_coroutine(llm.call_llm_json(
            self._build_prompt(job_description, candidates),
            max_tokens=2000,
            temperature=0.1,
            timeout=float(os.getenv("RERANK_LLM_TIMEOUT", "20")),
            attempts=1,
        ))

        # Accept either a bare array or {"scores": [...]} / {"results": [...]}.
        if isinstance(raw, dict):
            for key in ("scores", "results", "candidates"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break

        if not isinstance(raw, list):
            raise llm.LLMError("The model did not return a list of scores.")

        scores: dict[int, float] = {}
        for position, item in enumerate(raw):
            if not isinstance(item, dict) or item.get("score") is None:
                # A truncated trailing entry has no score; skipping it lets the
                # candidate keep its retrieval score instead of being zeroed.
                continue
            try:
                index = int(item.get("index", position))
                value = float(item["score"])
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates):
                scores[index] = max(0.0, min(value, 100.0)) / 100.0

        if not scores:
            raise llm.LLMError("The model returned no usable scores.")
        return scores

    def rerank(
        self,
        job_description: str,
        candidates: list[RetrievalResult],
        top_n: Optional[int] = None,
        alpha: float = 0.0,      # embedding scores are lexical here; trust the LLM
    ) -> list[RerankResult]:
        if not candidates:
            return []

        n = top_n or self.cfg.top_n
        ordered = sorted(candidates, key=lambda c: c.embedding_score, reverse=True)
        scored_subset = ordered[: self._MAX_CANDIDATES]
        remainder     = ordered[self._MAX_CANDIDATES :]

        t0 = time.perf_counter()
        try:
            scores = self._score_candidates(job_description, scored_subset)
            logger.info(
                "Hosted LLM scored %d candidates in %.2fs",
                len(scored_subset), time.perf_counter() - t0,
            )
        except Exception as exc:
            # Never fail the whole ranking request because the LLM was
            # unavailable -- degrade to retrieval order instead.
            logger.warning("Hosted LLM rerank failed (%s); falling back to retrieval order", exc)
            return _passthrough(ordered, n)

        combined: list[RerankResult] = []
        for i, c in enumerate(scored_subset):
            llm_score   = scores.get(i, c.embedding_score)
            final_score = alpha * c.embedding_score + (1 - alpha) * llm_score
            combined.append(RerankResult(
                resume_id       = c.resume_id,
                text            = c.text,
                embedding_score = c.embedding_score,
                rerank_score    = llm_score,
                final_score     = final_score,
                rank            = 0,
            ))

        # Anything past the cap keeps retrieval order, below everything scored.
        for c in remainder:
            combined.append(RerankResult(
                resume_id       = c.resume_id,
                text            = c.text,
                embedding_score = c.embedding_score,
                rerank_score    = 0.0,
                final_score     = 0.0,
                rank            = 0,
            ))

        combined.sort(key=lambda r: r.final_score, reverse=True)
        for rank, result in enumerate(combined):
            result.rank = rank
        return combined[:n]


def _passthrough(candidates: list[RetrievalResult], n: int) -> list[RerankResult]:
    """Retrieval-order results, used when no reranking model is available."""
    return [
        RerankResult(
            resume_id       = c.resume_id,
            text            = c.text,
            embedding_score = c.embedding_score,
            rerank_score    = c.embedding_score,
            final_score     = c.embedding_score,
            rank            = rank,
        )
        for rank, c in enumerate(candidates[:n])
    ]


def _run_coroutine(coro):
    """
    Run an async call from this sync reranker.

    rerank() is invoked inside a ThreadPoolExecutor worker (via
    Pipeline.rank_async), so there's no running loop in this thread and
    asyncio.run is safe. The isolated-loop branch covers direct sync calls
    made from a thread that does own a loop.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_reranker(config: Optional[RerankerConfig] = None):
    """
    Factory, in order of preference:
      1. local LLaMA reranker, if explicitly enabled
      2. local cross-encoder, when sentence-transformers is installed
      3. hosted LLM API, when an API key is configured (serverless path)
      4. retrieval-order passthrough
    """
    cfg = config or get_config().reranker

    if cfg.use_llm_reranker:
        logger.info("Using LLMReranker (local LLaMA)")
        return LLMReranker(cfg)

    if _CROSS_ENCODER and np is not None:
        logger.info("Using CrossEncoderReranker (local cross-encoder)")
        return CrossEncoderReranker(cfg)

    from app import llm
    if llm.is_configured():
        logger.info("Using HostedLLMReranker (%s)", llm.MODEL)
        return HostedLLMReranker(cfg)

    logger.warning("No reranking model available — falling back to retrieval order")
    return CrossEncoderReranker(cfg)