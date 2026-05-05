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
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

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

    @torch.no_grad()
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


def get_reranker(config: Optional[RerankerConfig] = None) -> CrossEncoderReranker | LLMReranker:
    """Factory: returns LLMReranker if configured, else CrossEncoderReranker."""
    cfg = config or get_config().reranker
    if cfg.use_llm_reranker:
        logger.info("Using LLMReranker (LLaMA 3 8B)")
        return LLMReranker(cfg)
    logger.info("Using CrossEncoderReranker (fast fallback)")
    return CrossEncoderReranker(cfg)