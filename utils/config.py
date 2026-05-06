"""
Centralized configuration via environment variables with typed defaults.
Override any value by setting the corresponding env var.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class RetrieverConfig:
    model_name: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    device: str = os.getenv("EMBED_DEVICE", "cpu")           # "cuda" in prod
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    max_seq_len: int = int(os.getenv("EMBED_MAX_SEQ_LEN", "512"))
    # FAISS
    faiss_index_path: Path = ROOT_DIR / "data" / "faiss_index" / "resumes.index"
    faiss_meta_path: Path = ROOT_DIR / "data" / "faiss_index" / "metadata.pkl"
    faiss_nlist: int = int(os.getenv("FAISS_NLIST", "100"))   # IVF clusters
    faiss_nprobe: int = int(os.getenv("FAISS_NPROBE", "10"))  # clusters to search
    top_k: int = int(os.getenv("TOP_K", "50"))                # candidates for reranker


@dataclass
class RerankerConfig:
    # Primary: LLaMA 3 8B + LoRA (set USE_LLM_RERANKER=true to enable)
    use_llm_reranker: bool = os.getenv("USE_LLM_RERANKER", "false").lower() == "true"
    llm_model_name: str = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    lora_weights_path: str | None = os.getenv("LORA_WEIGHTS_PATH", None)
    llm_device: str = os.getenv("LLM_DEVICE", "cuda")
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    # Fallback: cross-encoder
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))
    max_rerank_len: int = int(os.getenv("MAX_RERANK_LEN", "512"))
    top_n: int = int(os.getenv("TOP_N", "10"))               # final ranked results


@dataclass
class AppConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "1"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "3600"))
    max_resume_chars: int = int(os.getenv("MAX_RESUME_CHARS", "8000"))
    # 500 resumes < 2 min → ~240ms/resume budget
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT", "120"))


@dataclass
class Config:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    app: AppConfig = field(default_factory=AppConfig)


# Singleton
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
