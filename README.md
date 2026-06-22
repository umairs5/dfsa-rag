# DFSA Crypto-Token RAG System

A production-grade Retrieval-Augmented Generation system that answers questions about DFSA (Dubai Financial Services Authority) crypto-token regulations, with full MLOps scaffolding: evaluation harness, observability, CI/CD regression gates, and a feedback loop.

## What It Does

Ask a regulatory question, get a grounded answer with citations to specific DFSA rules:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Are Privacy Tokens permitted in the DIFC?"}'
```

```json
{
  "answer": "Under Rule 3A.2.2, a Person must not carry on a Financial Service relating to a Privacy Token or involving a Privacy Device in or from the DIFC...",
  "cited_rules": ["3A.2.2"],
  "citation_check": {"all_valid": true, "hallucinated_citations": []}
}
```

## Architecture

```
Question ──> Hybrid Retrieval ──> Grounded Generation ──> Cited Answer
              (BM25 + BGE        (Groq/Llama 3.3 70B     (with citation
               + Qdrant           + system prompt          guardrails)
               + RRF fusion)       grounding)
```

**Retrieval:** BM25 (keyword) + BGE-large (semantic) + Reciprocal Rank Fusion + optional cross-encoder reranker. Recall@5 = 1.0 with reranker, 0.914 without.

**Generation:** Groq API with Llama 3.3 70B. System prompt enforces grounding. Three-layer hallucination prevention: prompt grounding, citation validation, faithfulness heuristic.

**Eval:** Custom harness with 36 Q&A pairs across 7 categories. Retrieval metrics (recall, precision, MRR) + generation metrics (citation precision, faithfulness).

## Quick Start

```bash
# Setup
git clone https://github.com/umairs5/dfsa-rag.git
cd dfsa-rag
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: add GROQ_API_KEY (get free at console.groq.com)

# Ingest & index
python scripts/extract_pages.py
python scripts/chunk_rules.py
docker compose up qdrant -d
python scripts/index_qdrant_server.py

# Run API
uvicorn api.app:app --port 8000

# Or run everything in Docker
docker compose up --build
```

## Eval

```bash
# Retrieval eval (no API key needed)
python scripts/run_eval.py --mode hybrid --no-reranker

# Generation eval (needs GROQ_API_KEY)
python scripts/run_gen_eval.py --model llama-3.1-8b-instant --no-reranker
```

## Project Structure

```
api/           FastAPI serving layer (/ask, /feedback, /health)
retrieval/     Hybrid retrieval (BM25 + BGE + Qdrant + RRF + reranker)
generation/    Grounded generation (Groq + citation guardrails)
eval/          Evaluation harness (36 Q&A pairs, custom metrics)
scripts/       CLI tools (ingest, chunk, index, eval, feedback promotion)
docs/          Detailed system design documentation
```

## Key Results

| Component | Metric | Score |
|---|---|---|
| Retrieval (hybrid + reranker) | Recall@5 | 1.000 |
| Retrieval (hybrid, no reranker) | Recall@5 | 0.914 |
| Generation (Llama 3.3 70B) | Citation precision | 1.000 (7/7 questions) |
| Generation (Llama 3.1 8B) | Citation precision | 0.929 (5/35 hallucinated) |

## Stack

Python 3.11 | FastAPI | Qdrant | BGE-large-en-v1.5 | rank_bm25 | bge-reranker-v2-m3 | Groq (Llama 3) | Langfuse | Docker Compose | GitHub Actions

## Documentation

- [DESIGN.md](DESIGN.md) — Architecture summary and key decisions
- [docs/system-design.md](docs/system-design.md) — Detailed system design document
- [DECISIONS.md](DECISIONS.md) — Running log of every decision made

## License

MIT
