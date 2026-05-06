# Resume Screener AI

A production-grade, two-stage LLM-powered resume screening system. Built with FAISS vector retrieval, cross-encoder reranking, confidence scoring, and a real-time evaluation pipeline.

**Live Demo:** [resumescreen-ai.netlify.app/ui.html](https://resumescreen-ai.netlify.app/ui.html)  
**API:** [resumescreenerai-production.up.railway.app](https://resumescreenerai-production.up.railway.app)  
**API Docs:** [resumescreenerai-production.up.railway.app/docs](https://resumescreenerai-production.up.railway.app/docs)

---

## How It Works

```
Job Description + Resumes
         ↓
  Stage 1: FAISS Retrieval
  (SentenceTransformers bi-encoder)
  → Top-K candidates retrieved
         ↓
  Stage 2: Cross-Encoder Reranking
  (ms-marco-MiniLM cross-encoder)
  → Candidates reranked by relevance
         ↓
  Phase 3: Confidence Scoring + Fallback
  → Each result labelled High/Medium/Low/Uncertain
  → Keyword fallback for weak neural scores
         ↓
  Ranked Results with Scores + Metrics
```

---

## Features

### Phase 1 — Performance
- Batched embedding inference (64 texts/call)
- Content-addressed SHA-256 embedding cache
- FAISS IVFFlat index (auto-selects Flat for N<1000)
- Throughput telemetry on every request

### Phase 2 — Ranking Science
- Precision@K, Recall@K, NDCG@K, MRR, AP@K, F1@K
- Letter grades (A/B/C/D/F) per metric
- Plain English interpretation of each metric
- Overall verdict with batch evaluation support

### Phase 3 — Robustness
- Confidence scoring: High / Medium / Low / Uncertain
- Score gap analysis between ranked candidates
- Keyword-overlap fallback when neural score < 0.15
- Input validation: duplicate IDs, empty text, short resumes

### Phase 4 — API Maturity
- `POST /rank` — synchronous ranking
- `POST /rank/batch` — async job queue, returns job_id instantly
- `GET /jobs/{job_id}` — poll for batch job status
- Sliding window rate limiter (per IP)

---

## Project Structure

```
Resume_Screener_Ai/
├── app/
│   ├── main.py            # FastAPI endpoints (all 4 phases)
│   ├── pipeline.py        # Two-stage orchestrator
│   ├── retriever.py       # FAISS bi-encoder (Stage 1)
│   ├── reranker.py        # Cross-encoder / LLM reranker (Stage 2)
│   ├── preprocessor.py    # PDF/DOCX/text parsing + cleaning
│   ├── metrics.py         # Precision@K, NDCG, MRR, Spearman
│   ├── metrics_report.py  # Graded metrics report builder
│   ├── confidence.py      # Confidence scorer (Phase 3)
│   ├── fallback.py        # Keyword fallback heuristic (Phase 3)
│   ├── validator.py       # Input validation (Phase 3)
│   ├── batch_manager.py   # Batch embedder + cache (Phase 1)
│   └── job_queue.py       # Async job runner (Phase 4)
├── utils/
│   ├── config.py          # Typed config via env vars
│   ├── cache.py           # LRU + optional Redis tiered cache
│   └── batching.py        # Adaptive batcher, parallel embedding
├── tests/
│   ├── test_retriever.py
│   ├── test_reranker.py
│   └── test_pipeline.py
├── scripts/
│   ├── benchmark.py       # 500-resume throughput test
│   └── finetune_lora.py   # LLaMA 3 8B LoRA fine-tuning
├── ui.html                # Frontend UI (all 4 phases)
├── railway.toml           # Railway deployment config
├── Procfile
└── requirements.txt
```

---

## Accessing the Live App

Open the live UI at:

```
https://resumescreen-ai.netlify.app/ui.html
```

1. Paste a job description in **01 — Job Description**
2. Add candidates in **02 — Candidates** (or click **Load Example**)
3. Optionally enable evaluation metrics and provide relevant candidate IDs
4. Click **▶ Screen Candidates** for synchronous ranking
5. Click **⚡ Batch Mode** to submit as an async job

Results show:
- Ranked candidates with final score, semantic match, and rerank score
- Confidence level (High / Medium / Low / Uncertain) per candidate
- Fallback indicator if keyword matching was used
- FAISS index info (type, vectors, dimension)
- NDCG / Precision / Recall metrics with letter grades (if eval enabled)

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/RitvikMalhotra/Resume_Screener_Ai.git
cd Resume_Screener_Ai
```

### 2. Set up environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Run the API

```bash
python run.py
```

API available at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### 4. Open the UI

Open `ui.html` in your browser. Make sure the API URL in the file points to `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + index + cache stats |
| `POST` | `/rank` | Rank resumes against a JD |
| `POST` | `/rank/batch` | Submit async batch job |
| `GET` | `/jobs/{job_id}` | Poll batch job status |
| `GET` | `/jobs` | List all jobs + queue stats |
| `POST` | `/index/build` | Build/rebuild FAISS index |
| `GET` | `/index/stats` | FAISS index metadata |
| `POST` | `/cache/warm` | Pre-embed corpus into cache |
| `POST` | `/evaluate` | Full metrics report with grades |
| `POST` | `/evaluate/batch` | Averaged metrics across queries |
| `GET` | `/metrics/cache` | Cache hit rate + throughput |

---

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | SentenceTransformer model |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `TOP_K` | `50` | FAISS retrieval candidates |
| `USE_LLM_RERANKER` | `false` | Enable LLaMA 3 8B reranker |
| `LORA_WEIGHTS_PATH` | `None` | Path to fine-tuned LoRA weights |
| `TOP_N` | `10` | Final ranked results returned |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Fallback reranker |

---

## Performance

- 500 resumes ranked in under 60s on CPU (cached embeddings)
- FAISS FlatIP for N < 1000, IVFFlat for larger corpora
- Embedding cache eliminates recomputation on repeated texts
- Batch mode allows non-blocking submission of large jobs

---

## Tech Stack

- **FastAPI** — async REST API
- **SentenceTransformers** — bi-encoder embeddings (`bge-large-en-v1.5`)
- **FAISS** — vector similarity search
- **Cross-Encoder** — `ms-marco-MiniLM-L-12-v2` reranker
- **Railway** — backend deployment
- **Netlify** — frontend hosting

---

## LoRA Fine-tuning (Optional)

To fine-tune LLaMA 3 8B as the reranker:

```bash
python scripts/finetune_lora.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/train.jsonl \
  --output_dir models/lora_reranker \
  --epochs 3
```

Training data format (JSONL):
```json
{"job_description": "...", "resume": "...", "label": 1}
{"job_description": "...", "resume": "...", "label": 0}
```

Enable after training:
```bash
USE_LLM_RERANKER=true LORA_WEIGHTS_PATH=models/lora_reranker python run.py
```