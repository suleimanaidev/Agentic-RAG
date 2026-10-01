# ⚡ NexusAI — Document Intelligence

A consolidated Streamlit RAG application for uploading, searching and chatting
with documents. **Application functionality lives in `app.py`; RAG evaluation
remains separate in `evaluate_rag.py`.**

## Features

- PDF, DOCX, CSV, TXT and Markdown ingestion, with structure-aware splitting.
- Local embedded Qdrant or a cloud cluster, using the same embedding model.
- Five search strategies: BM25 + dense hybrid, native Qdrant hybrid, semantic
  similarity, score cutoff and MMR diversity.
- Real cosine scores, source filtering and a configurable cutoff for every strategy.
- Follow-up question reformulation, streamed answers and a single final citation block.
- Evidence-first display, no-LLM preview, chat export and bounded session caches.
- Idempotent uploads, same-name file replacement, batched writes and cleanup of
  newly inserted chunks when an upload batch fails.
- Paginated recovery, cross-session invalidation within one server process, manual
  refresh and automatic refresh on interaction after 60 seconds.
- API timeouts/retries, input validation, server-side error logs and a Docker health check.

## Project layout

```text
app.py                  # All application, ingestion and search functionality
evaluate_rag.py         # Independent dense-retrieval benchmark and LLM judge
test_regressions.py      # Offline backend + Streamlit UI regression checks
requirements.txt
.env.example
.streamlit/config.toml
Dockerfile
eval_results/           # Evaluation reports
qdrant_store/           # Generated local database
```

The former standalone hybrid demo is merged into the app's search strategies.
No separate demo script is needed.

## Run locally

Use **Python 3.10+** (the container uses Python 3.11). A dedicated virtual
environment avoids conflicts with unrelated installed packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

Open <http://localhost:8501>. The first real run downloads the
`all-MiniLM-L6-v2` embedding model; subsequent runs use the HuggingFace cache.

### Configuration

Set these values in `.env` or environment variables:

| Variable | Purpose |
| --- | --- |
| `QDRANT_URL` | Cloud cluster URL; leave empty for local storage |
| `QDRANT_API_KEY` | Cloud cluster API key |
| `QDRANT_COLLECTION` | Collection name; defaults to `who_hybrid` |
| `GROQ_API_KEY` | Groq generation and independent evaluation |
| `OPENAI_API_KEY` | Optional OpenAI generation |

To use older indexed documents, set `QDRANT_COLLECTION=enterprise_knowledge_base`
if that is their existing collection. Collections must have unnamed **384-dimensional
cosine vectors**. Both application and evaluation use this collection setting.

Select files and click **Index selected documents**. Unchanged files with the same
chunking settings are skipped. Changed files replace older chunks after the new
version has been written. Scanned/image-only PDFs require OCR before upload.
Uploads are limited to 50 MB per file.

### Search behavior

- **Hybrid (Dense + BM25):** equal-weight reciprocal rank fusion of dense results
  and keyword-matching BM25 results. BM25 runs over the selected document scope.
- **Native Hybrid (Qdrant RRF):** server-side rank fusion using dense vectors and
  deterministic hashed term-frequency sparse vectors. This is not corpus-weighted
  BM25. A disposable `<collection>__native` companion index is rebuilt lazily after
  refresh/ingestion using the same Qdrant client. The primary index stays intact.
- **MMR:** selects diverse passages; the displayed scores remain cosine similarities.
- A zero cutoff disables similarity filtering. Similarity is **not** answer confidence.

## Checks and evaluation

```powershell
python -m unittest -v test_regressions
python -m compileall -q app.py evaluate_rag.py test_regressions.py
python evaluate_rag.py
```

Regression checks use deterministic embeddings and in-memory Qdrant, with no
API calls or model downloads. They cover file loading, indexing/rollback,
replacement, pagination, hybrid filtering/scoring, judge validation and Streamlit flows.

Evaluation requires `QDRANT_URL`, a Groq key and the WHO
"Substances under Surveillance" report indexed as the active collection. The five
benchmark questions and ground truths come directly from that report. It measures a
**dense-only k=8 baseline** with a separate generation prompt; it does not benchmark
every interactive strategy. LLM-judge scores are estimates. JSON and Markdown reports
go to `eval_results/`.

## Container deployment

```powershell
docker build -t nexusai .
docker run --rm -p 8501:8501 --env-file .env -v nexusai-data:/app/qdrant_store -v nexusai-models:/home/nexusai/.cache nexusai
```

The image runs as a non-root user and exposes Streamlit's `/_stcore/health` endpoint.
Use a single application process with local Qdrant; persist both the data volume
and embedding-model cache. For Cloud mode, persistent local vector storage is unnecessary.

### Deployment scope

This application provides **one shared knowledge base for a trusted team**.
Uploads and Reset Docs affect everyone using that collection. Session chats and
API-key fields are per browser session. For a public deployment, place it behind
authenticated HTTPS access. Tenant isolation, distributed write coordination and
multi-replica serving are not implemented. Cloud deployments needing these require
an authenticated backend and per-tenant storage before wider rollout.

Back up the primary Qdrant collection and supply secrets through the deployment
environment. The native companion is rebuildable. Dependency major versions are
bounded in `requirements.txt`; create a deployment lockfile in your release process
for fully reproducible images.
