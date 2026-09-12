# Resume Screener AI

A production-grade, two-stage LLM-powered resume screening system with PDF upload, auto name extraction, and side-by-side candidate comparison.

**Deployment:** Vercel serves the frontend and FastAPI backend together.  
**API:** `/api`  
**API Docs:** `/api/docs`

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
├── vercel.json             # Vercel routing config
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
pip install -r requirements-local.txt
```

### 3. Run

```bash
python run.py
```

API at `http://localhost:8000` — Docs at `http://localhost:8000/docs`

### 4. Open UI

Open `ui.html` locally and update the API constant to `http://localhost:8000`:

```javascript
const API = 'http://localhost:8000';
```

### Deploy to Vercel

Import this repository into Vercel with the project root set to the repository root. Vercel auto-detects `app/main.py` as a FastAPI framework preset and routes every request straight into that app — no `vercel.json` or `api/` directory needed, and none of Vercel's usual static-file hosting applies once a framework preset is active. Because of that, `app/main.py` also serves the static frontend pages itself (`/`, `/ui.html`, `/dashboard.html`, `/login.html`) via explicit `FileResponse` routes. No Railway service is required. The frontend is available at `/ui.html`, the health check at `/health`, and the interactive API docs at `/docs` — note there is no `/api` prefix; `ui.html`'s `API` constant is set to `''` for the deployed site.

Set `NVIDIA_API_KEY` in the Vercel project environment variables if the AI analysis endpoints are enabled. Set `NVIDIA_MODEL` to the exact model ID shown on the NVIDIA model page if the default model is unavailable. Vercel uses a lightweight lexical ranking fallback because the full SentenceTransformer/FAISS stack exceeds Vercel's 500 MB function limit. Local installs from `requirements-local.txt` use the full embedding and cross-encoder pipeline.

Vercel functions are ephemeral and have execution, memory, and deployment-size limits. Large resume batches, model cold starts, and `/rank/batch` jobs may exceed those limits. For a reliable production deployment, keep requests small or move model inference to a dedicated inference service later while retaining this Vercel frontend and API boundary.

### AI features

Every AI feature routes through `app/llm.py`, which calls a hosted chat-completions
model (NVIDIA API catalog by default). Configure it with:

- `NVIDIA_API_KEY` — required for any AI feature.
- `NVIDIA_MODEL` — model id (default `meta/muse-glimmer-30b`).
- `NVIDIA_API_URL` — override the endpoint (mainly for testing against a mock).
- `NVIDIA_TIMEOUT` / `NVIDIA_MAX_ATTEMPTS` / `RERANK_LLM_TIMEOUT` — request budgets,
  kept tight so a slow model degrades gracefully instead of hitting the serverless
  execution limit.

The model is used for `/explain`, `/skillgap`, `/redflag`, `/jdquality`,
`/skillsummary`, `/jdenhance`, `/emailtemplate` — **and for ranking itself** on
deployments where sentence-transformers can't be installed. `get_reranker()` picks,
in order: local LLaMA (if `USE_LLM_RERANKER=true`) → local cross-encoder (if
sentence-transformers is present) → `HostedLLMReranker` via the API → retrieval-order
passthrough. Without the hosted reranker, serverless ranking falls back to lexical
word overlap, which scores a perfectly matching resume around 0.2 purely because it
doesn't reuse the JD's vocabulary.

`app/llm.py` handles the response shapes that a naive
`data["choices"][0]["message"]["content"]` breaks on: `content: null` with the answer
in `reasoning_content`, unterminated `<think>` blocks from truncated output, markdown-
fenced or prose-wrapped JSON, and JSON cut off mid-object. AI endpoints return a clean
502/503 with a readable message rather than a 500, and `/rank` degrades to retrieval
order rather than failing if the model is unavailable.

### Payments

Upgrading to Pro requires a payment Razorpay actually signed. `POST /payments/order`
creates the order server-side (the price lives in `app/payments.py`, not the client),
and `POST /auth/upgrade` only grants Pro when the HMAC-SHA256 signature over
`{order_id}|{payment_id}` verifies against the key secret. Orders are recorded in
`payment_orders` and claimed with a conditional UPDATE, so a payment can't be replayed
or applied to a different account.

- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` — required for the upgrade flow. The
  secret stays server-side; the browser only ever sees the key id.
- `PRO_PLAN_AMOUNT_PAISE` — plan price, default `99900` (₹999).

Without these set, `/payments/order` and `/auth/upgrade` return 503 and the upgrade
button reports that payments aren't configured — nobody gets a free upgrade.

### Auth

Accounts, login, and screening history are backed by a small custom auth layer in `app/main.py`/`app/auth.py`/`app/db.py` (bcrypt password hashing + JWT bearer tokens) — there is no third-party auth provider. Set two env vars for it to work:

- `DATABASE_URL` — a Postgres connection string. On Vercel, add a Neon Postgres database from the project's **Storage** tab (Marketplace) and it's injected automatically.
- `JWT_SECRET` — a random secret for signing session tokens (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`).

The `users` and `screenings` tables are created automatically on startup (`db.init_schema()`) if they don't exist. Without these two env vars set, `/health` reports `"auth": false` and the `/auth/*` and `/screenings` endpoints return `503` rather than crashing — the rest of the app (ranking, AI endpoints) works fine without them.

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
**Deployment:** Vercel (frontend + FastAPI backend)  
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