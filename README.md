# Resume Screener AI

A production-grade, two-stage LLM-powered resume screening system with PDF upload, auto name extraction, and side-by-side candidate comparison.

**Live Demo:** [resumescreen-ai.netlify.app/ui.html](https://resumescreen-ai.netlify.app/ui.html)  
**API:** [resumescreenerai-production.up.railway.app](https://resumescreenerai-production.up.railway.app)  
**API Docs:** [resumescreenerai-production.up.railway.app/docs](https://resumescreenerai-production.up.railway.app/docs)

---

## How It Works

```
Job Description + Resumes (text or PDF)
              ↓
    Stage 1: FAISS Retrieval
    SentenceTransformers bi-encoder
    → Top-K candidates retrieved
              ↓
    Stage 2: Cross-Encoder Reranking
    ms-marco-MiniLM cross-encoder
    → Candidates reranked by relevance
              ↓
    Phase 3: Confidence Scoring + Fallback
    → High/Medium/Low/Uncertain labels
    → Keyword fallback for weak neural scores
    → Uncertain results penalized in final sort
              ↓
    Ranked Results + Metrics + Comparison View
```

---

## Features

### UI
- **PDF Upload** — upload PDF resumes directly; text extracted in-browser via PDF.js
- **Auto Name Detection** — candidate ID auto-filled from name detected in PDF
- **Candidate Comparison** — select any 2 candidates for side-by-side score breakdown
- **Live System Bar** — shows FAISS index type, vector count, cache hit rate, job count
- **Evaluation Metrics** — NDCG/Precision/Recall with letter grades when ground truth provided
- **Batch Mode** — async job submission with real-time polling

### Phase 1 — Performance
- Batched embedding inference (64 texts/call)
- Content-addressed SHA-256 embedding cache
- FAISS IVFFlat index (auto-selects FlatIP for N < 1000)
- Throughput telemetry on every request

### Phase 2 — Ranking Science
- Precision@K, Recall@K, NDCG@K, MRR, AP@K, F1@K
- Letter grades (A/B/C/D/F) per metric with plain English descriptions
- Overall verdict with batch evaluation support

### Phase 3 — Robustness
- Confidence scoring: High / Medium / Low / Uncertain
- Score gap analysis between ranked candidates
- Keyword-overlap fallback when neural score < 0.15
- Confidence penalty re-sorting: uncertain results demoted automatically
- Input validation: duplicate IDs, empty text, short resumes flagged

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
│   ├── reranker.py        # Cross-encoder reranker (Stage 2)
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
├── ui.html                # Frontend UI (PDF upload + comparison)
├── railway.toml           # Railway deployment config
├── Procfile
└── requirements.txt
```

---

## Accessing the Live App

```
https://resumescreen-ai.netlify.app/ui.html
```

### Text input
1. Paste a job description in **01 — Job Description**
2. Add candidates manually in **02 — Candidates**
3. Click **▶ Screen Candidates**

### PDF upload
1. Click **📄 Upload PDF** on any candidate row
2. Select a PDF resume — text is extracted automatically in your browser
3. Candidate ID is auto-filled from the name detected in the PDF
4. Click **▶ Screen Candidates**

### Candidate comparison
1. After results appear, check the box on any 2 result cards
2. Click **Compare Selected**
3. Side-by-side view shows scores, confidence, matched keywords, and a verdict

### Evaluation metrics
1. Check **Enable evaluation metrics**
2. Enter the IDs of candidates you know are relevant (comma-separated)
3. Results include NDCG@K, Precision@K, Recall@K with letter grades

---

## Running Locally

### 1. Clone

```bash
git clone https://github.com/RitvikMalhotra/Resume_Screener_Ai.git
cd Resume_Screener_Ai
```

### 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Run

```bash
python run.py
```

API at `http://localhost:8000` — Docs at `http://localhost:8000/docs`

### 4. Open UI

Open `ui.html` and update the API constant to `http://localhost:8000`:

```javascript
const API = 'http://localhost:8000';
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status + index + cache + job stats |
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

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `TOP_K` | `50` | FAISS retrieval candidates |
| `USE_LLM_RERANKER` | `false` | Enable LLaMA 3 8B reranker |
| `LORA_WEIGHTS_PATH` | `None` | Path to fine-tuned LoRA weights |
| `TOP_N` | `10` | Final ranked results |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |

---

## Performance

- 500 resumes ranked in under 60s on CPU with cached embeddings
- FAISS FlatIP for N < 1000, IVFFlat for larger corpora
- Embedding cache eliminates recomputation on repeated texts
- Async batch mode for non-blocking large job submission
- Confidence penalty sorting ensures high-confidence results surface first

---

## Tech Stack

**Backend:** FastAPI, SentenceTransformers, FAISS, Python  
**Frontend:** Vanilla HTML/JS, PDF.js (client-side PDF parsing)  
**Deployment:** Railway (backend) + Netlify (frontend)  
**Models:** `all-MiniLM-L6-v2` (embeddings), `ms-marco-MiniLM-L-6-v2` (reranker)

---

## LoRA Fine-tuning (Optional)

```bash
python scripts/finetune_lora.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/train.jsonl \
  --output_dir models/lora_reranker \
  --epochs 3
```

Training data format:
```json
{"job_description": "...", "resume": "...", "label": 1}
{"job_description": "...", "resume": "...", "label": 0}
```

Enable after training:
```bash
USE_LLM_RERANKER=true LORA_WEIGHTS_PATH=models/lora_reranker python run.py
```