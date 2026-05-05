# Resume Screening System

Two-stage LLM-powered pipeline for ranking resumes against a job description.

```
Stage 1 (Retrieval)    →    Stage 2 (Reranking)
SentenceTransformers         Cross-Encoder  OR
+ FAISS IVFFlat              LLaMA 3 8B + LoRA
top-K candidates             top-N final results
```

## Project Structure

```
resume_screener/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── pipeline.py      # Two-stage orchestrator
│   ├── retriever.py     # Bi-encoder + FAISS (Stage 1)
│   ├── reranker.py      # Cross-encoder / LLM reranker (Stage 2)
│   ├── preprocessor.py  # PDF/DOCX/text parsing + cleaning
│   └── metrics.py       # precision@k, recall@k, NDCG, MRR, Spearman
├── utils/
│   ├── config.py        # Typed config via env vars
│   ├── cache.py         # LRU + optional Redis tiered cache
│   └── batching.py      # Adaptive batcher, parallel embedding
├── scripts/
│   ├── benchmark.py     # 500-resume throughput test
│   └── finetune_lora.py # LLaMA 3 8B LoRA fine-tuning
├── tests/
│   ├── test_retriever.py
│   ├── test_reranker.py
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Install

```bash
cd resume_screener
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run API

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Rank resumes

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior ML Engineer with PyTorch, LLMs, FAISS experience.",
    "resumes": [
      {"resume_id": "r1", "text": "ML engineer, PyTorch, transformers, FAISS, 6yr exp."},
      {"resume_id": "r2", "text": "Frontend developer, React, TypeScript, 4yr exp."},
      {"resume_id": "r3", "text": "NLP researcher, LLaMA fine-tuning, LoRA, RLHF."}
    ],
    "top_n": 3
  }'
```

**Response:**
```json
{
  "results": [
    {"resume_id": "r1", "rank": 0, "final_score": 0.921, ...},
    {"resume_id": "r3", "rank": 1, "final_score": 0.887, ...},
    {"resume_id": "r2", "rank": 2, "final_score": 0.234, ...}
  ],
  "total_candidates": 3,
  "timing": {"preprocess_s": 0.01, "index_build_s": 0.4, "retrieval_s": 0.02, "rerank_s": 0.3, "total_s": 0.73}
}
```

## Docker

```bash
docker compose up --build
```

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | SentenceTransformer model |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `TOP_K` | `50` | FAISS retrieval candidates |
| `USE_LLM_RERANKER` | `false` | Enable LLaMA 3 8B reranker |
| `LLM_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | HuggingFace model ID |
| `LORA_WEIGHTS_PATH` | `None` | Path to fine-tuned LoRA weights |
| `LOAD_IN_4BIT` | `true` | 4-bit quantization for LLM |
| `TOP_N` | `10` | Final ranked results |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Fallback reranker |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/rank` | Rank resumes against a JD |
| `POST` | `/index/build` | Pre-build FAISS index |
| `GET` | `/index/stats` | Index size + cache stats |
| `POST` | `/evaluate` | Compute ranking metrics |
| `GET` | `/health` | Liveness check |

## Tests

```bash
# Unit tests (no model download required for metrics/cache tests)
pytest tests/test_reranker.py::TestMetrics -v

# Full integration tests (downloads ~100MB models)
pytest tests/ -v -m slow
```

## LoRA Fine-tuning

Prepare training data as JSONL:
```jsonl
{"job_description": "...", "resume": "...", "label": 1}
{"job_description": "...", "resume": "...", "label": 0}
```

Run fine-tuning:
```bash
python scripts/finetune_lora.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/train.jsonl \
  --output_dir models/lora_reranker \
  --epochs 3
```

Enable in production:
```bash
USE_LLM_RERANKER=true LORA_WEIGHTS_PATH=models/lora_reranker uvicorn app.main:app
```

## Performance Notes

**500 resumes < 2 minutes** is achievable with:
- CPU: `all-MiniLM-L6-v2` (fast) + `MiniLM-L-6-v2` cross-encoder → ~30–60s
- GPU: `BAAI/bge-large-en-v1.5` + `ms-marco-MiniLM-L-12-v2` → ~10–20s
- GPU + LLM reranker (4-bit): `bge-large` + LLaMA 3 8B → ~60–90s on A100

Bottleneck is almost always the reranker. The bi-encoder + FAISS stage
processes 500 resumes in under 5s on CPU with cached embeddings.

Run benchmark:
```bash
python scripts/benchmark.py --n 500 --top-n 10 --runs 3
```
