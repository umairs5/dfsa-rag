# System Design Document: DFSA Crypto-Token RAG System

## 1. Problem Statement

A compliance officer at a DFSA-regulated firm needs to quickly find accurate answers to regulatory questions about crypto-token rules. Example: "Can we provide credit to retail clients for crypto trading?" The answer must cite the specific DFSA rule, be grounded in actual regulatory text, and not hallucinate requirements that don't exist.

The naive approach — stuff all rules into an LLM prompt — doesn't scale beyond small corpora and provides no quality guarantees. This system uses Retrieval-Augmented Generation with production engineering: evaluation before optimization, hallucination guardrails, observability, CI/CD quality gates, and a user feedback loop.

### Target User

DFSA compliance officers who need specific, citable answers about crypto-token regulations in the DIFC (Dubai International Financial Centre).

### Scope

Limited to the crypto-token regulatory framework:
- **GEN 3A** — Crypto Token Requirements
- **GEN App2 A2.5** — Token Definitions
- **COB 15** — Conduct rules for crypto businesses
- **AML 9.3A/9.3B** — Travel rule for crypto transfers

This scope is deliberately narrow — topically self-contained but heavily cross-referenced, providing genuine chunking and retrieval challenges.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT (curl / browser)               │
│                      POST /ask                           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI (api/app.py)                   │
│    Lifespan: loads BGE model, BM25, Qdrant, Groq client  │
│    Endpoints: /health, /ask, /feedback                   │
│    Middleware: CORS                                      │
└──────┬───────────────┬──────────────────┬───────────────┘
       │               │                  │
       ▼               ▼                  ▼
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│ Retrieval │   │  Generation   │   │   Feedback    │
│           │   │              │   │              │
│ BM25      │   │ Groq API     │   │ feedback.jsonl│
│    +      │   │ (Llama 3.3)  │   │      ↓       │
│ Qdrant    │   │      ↓       │   │ promote.py   │
│    +      │   │ Citation     │   │      ↓       │
│ RRF Fusion│   │ Guardrails   │   │ eval_set.json│
│    +      │   │              │   │              │
│ Reranker  │   │              │   │              │
│ (optional)│   │              │   │              │
└──────────┘   └──────────────┘   └──────────────┘
       │               │
       └───────┬───────┘
               ▼
       ┌──────────────┐        ┌──────────────┐
       │   Langfuse    │        │ GitHub Actions│
       │   Tracing     │        │   CI/CD       │
       │ (per-request) │        │ (per-push)    │
       └──────────────┘        └──────────────┘
```

### Request Flow

1. Client sends `POST /ask {"question": "Are Privacy Tokens permitted?"}`
2. FastAPI validates request via Pydantic model
3. Hybrid retriever finds top-5 relevant chunks:
   - BM25 scores all 52 chunks by keyword overlap
   - BGE embeds the query, Qdrant finds nearest vectors
   - RRF merges both ranked lists by rank position
   - (Optional) Cross-encoder reranker rescores the merged candidates
4. Generation module formats chunks into a grounded prompt
5. Groq API (Llama 3.3 70B) generates an answer
6. Citation extractor pulls rule references from the answer
7. Citation validator checks each cited rule exists in the retrieved chunks
8. Langfuse records the trace (retrieval span, generation span, guardrail span)
9. Response returned with answer, citations, validity check, and source chunks

---

## 3. Component Deep Dives

### 3.1 Ingestion & Chunking

**Input:** 3 DFSA rulebook PDFs (~460 pages total, ~37 pages in scope)

**Extraction** (`scripts/extract_pages.py`):
- PyMuPDF extracts text from hardcoded page ranges per section
- Headers/footers stripped via regex (module name, page number, version string)
- Page markers (`<<<PAGE 65>>>`) inserted for downstream page tracking
- Output: raw text files in `data/intermediate/`

**Chunking** (`scripts/chunk_rules.py`):
- Rule boundary detection via regex: `r'^((?:A\d+|\d+[A-Z]?)\.\d+\.\d+[A-Z]?)\s'`
- Guidance blocks merged with preceding rule (semantically inseparable)
- False positive filtering: cross-references at line start (e.g., "...Rule\n15.3.1 to publish...") filtered by checking if preceding line ends with "Rule"
- Context header breadcrumb prepended (e.g., "GEN > 3A Crypto Token Requirements > 3A.2 Prohibitions")
- Cross-references extracted via regex for metadata
- Output: `data/processed/chunks.jsonl` (52 chunks, median 244 tokens)

**Why rule-level chunking:**
- The rule is the natural citation unit in DFSA regulations
- Guidance blocks explain the rule they follow — splitting them apart breaks semantic coherence
- Chunk sizes (50-1160 tokens) fit embedding models well without truncation
- Alternative rejected: fixed-size 500-token chunks would split rules mid-sentence and separate guidance from its rule

**Why hardcoded page ranges:**
- PDFs update ~once per year. Version strings (GEN/VER71/01-26) are tracked to detect when re-extraction is needed
- Dynamic section detection would add parser complexity for marginal benefit on 3 stable documents

### 3.2 Retrieval

**Architecture:** Two-stage hybrid retrieval (fast first stage → accurate reranker)

**Stage 1a — Sparse (BM25):**
- `rank_bm25.BM25Okapi` over tokenized chunk texts
- Tokenization preserves rule numbers as single tokens (regex: `[\w.]+`)
- Strength: exact term matching ("Algorithmic Token", "Rule 3A.2.1")
- Weakness: misses paraphrases ("privacy coins" vs "Privacy Tokens")

**Stage 1b — Dense (BGE + Qdrant):**
- BGE-large-en-v1.5 (1024-dim) embeds queries and chunks
- Query prefix: "Represent this sentence for retrieval: " (asymmetric training)
- Chunks embedded with context_header prepended for structural context
- Qdrant stores vectors (on-disk for scripts, server for API)
- Strength: semantic similarity regardless of exact wording
- Weakness: may miss exact identifiers, less interpretable

**Fusion — Reciprocal Rank Fusion (RRF):**
- Merges BM25 and dense results using only rank positions: `score = 1/(60 + rank)`
- Rank-based fusion avoids the BM25/cosine score scale mismatch
- Items found by both retrievers get boosted (scores sum)
- k=60 constant dampens difference between adjacent ranks

**Stage 2 — Cross-Encoder Reranker (optional):**
- BAAI/bge-reranker-v2-m3 (568M params)
- Scores each (query, chunk) pair with full cross-attention
- Much more accurate than bi-encoder, but O(n) forward passes
- ~30 seconds per query on CPU — defaults to off in API, on for batch eval
- Improves recall@5 from 0.914 to 1.000

**Why hybrid over dense-only:**
- Dense alone: recall@5 = 0.943. Misses queries where exact legal terms matter
- BM25 alone: recall@5 = 0.871. Misses paraphrased questions
- Hybrid + reranker: recall@5 = 1.000. Best of both

**Surprising finding:** Hybrid without reranker (0.914) performed worse than dense-only (0.943). RRF diluted strong dense results with weak BM25 candidates. The reranker fixed this by re-sorting the merged list accurately. Lesson: fusion without reranking can hurt.

### 3.3 Generation

**LLM:** Llama 3.3 70B via Groq API (free tier, ~1s inference)

**System prompt** (`generation/prompt.py`):
- Instructs the model to answer ONLY from provided context
- Requires citing specific rule numbers
- Says "I don't have enough information" when context is insufficient
- Zero-shot (no few-shot examples) — saves context window for actual chunks

**Chunk formatting in prompt:**
Each retrieved chunk is labeled with its rule_id and context_header:
```
[Source 1: Rule 3A.2.2 | GEN > 3A Crypto Token Requirements > 3A.2 Prohibitions]
3A.2.2 A Person must not in or from the DIFC...
```
This gives the LLM citation targets and structural context.

**Temperature = 0:** Deterministic output. "Must" and "should" have different legal meanings — the model should always pick the same word the regulation uses.

**Hallucination guardrails (three layers):**
1. **System prompt grounding** (preventive) — constrains the model to provided context
2. **Citation extraction + validation** (detective) — regex extracts rule references, validates against retrieved chunks
3. **Faithfulness heuristic** (detective) — checks that key legal terms in the answer appear in chunk text

**70B vs 8B comparison:**
- 70B: 0 hallucinated citations in 7 questions tested
- 8B: 5 hallucinated citations in 35 questions (14%). Worst case: proc_01 cited 5 rules, only 1 valid
- 8B invents plausible-sounding rule numbers from adjacent sections

### 3.4 Serving

**API:** FastAPI with Pydantic validation, three endpoints:
- `GET /health` — verifies Qdrant connection
- `POST /ask` — full RAG pipeline, returns answer + citations + source chunks
- `POST /feedback` — stores user judgment for eval set growth

**Startup:** Lifespan context manager loads all heavy resources once:
- BGE model (~5s)
- Cross-encoder reranker (~5s)
- Qdrant connection
- BM25 index
- Groq client

Two HybridRetriever instances pre-loaded (with/without reranker) to avoid per-request model loading.

**Containerization:** Docker Compose with two services:
- `qdrant` — official Qdrant image with named volume for persistence
- `api` — custom image built from Dockerfile, connects to Qdrant over HTTP

### 3.5 Observability

**Tool:** Langfuse (cloud, free tier)

**Trace structure per request:**
```
[trace: "ask"] ── input: question, output: answer summary
  ├── [span: "retrieval" type=retriever] ── chunk_ids, latency
  ├── [span: "llm" type=generation] ── model, tokens, cost
  └── [span: "citation_check" type=guardrail] ── valid/hallucinated
```

**Score:** `hallucination_free` (1.0 or 0.0) per trace — chartable over time.

**Graceful degradation:** If `LANGFUSE_PUBLIC_KEY` is not set, tracing is skipped silently. Observability never breaks the application.

### 3.6 Evaluation

**Custom eval harness** (not RAGAS) with 36 Q&A pairs across 7 categories:
- Definitional (8), Prohibition (8), Custody (5), Disclosure (5), AML (4), Cross-reference (3), Procedural (2), User-feedback (1+)

**Retrieval metrics:** recall@k, precision@k, MRR (Mean Reciprocal Rank)
**Generation metrics:** citation_precision, citation_recall, faithfulness_heuristic

**Pluggable interface:** `retrieve_fn(question: str, k: int) -> list[str]` — any retrieval implementation conforms. Enables ablation studies (BM25-only, dense-only, hybrid, hybrid+reranker).

### 3.7 CI/CD

**GitHub Actions** workflow runs on every push/PR to main:
1. Install dependencies (cached)
2. Extract PDFs and chunk (from scratch — tests full pipeline)
3. Build in-memory Qdrant + BM25 indices
4. Run retrieval eval (35+ questions)
5. Run generation eval via Groq (if API key available)
6. Assert thresholds: recall@5 ≥ 0.85, citation_precision ≥ 0.90
7. Exit 0 (pass) or 1 (fail)

BGE model cached via `actions/cache` (~750MB, saves ~1 min on subsequent runs).
Groq rate limits handled gracefully — generation eval scores with available questions.

### 3.8 Feedback Loop

**Flow:** User asks → gets answer → submits thumbs up/down + optional correction → feedback stored in JSONL → promotion script adds high-quality feedback to eval set → CI tests against growing eval set.

**Promotion criteria:** Feedback is promotable if it has a corrected answer (user fixed a wrong answer) or expected chunks (user specified which rules matter).

**Manual review gate:** Promotion is not automatic — `promote_feedback.py --dry-run` shows candidates, human reviews before applying. Prevents bad corrections from poisoning the eval set.

---

## 4. Key Tradeoffs

### Raw Python vs LangChain/LlamaIndex
**Chose:** Raw Python.
**Why:** This is a learning project — understanding the primitives (BM25, embeddings, RRF, cross-encoders) before adopting a framework that hides them. Every retrieval concept is explicit in the code.
**Cost:** More code (~2900 lines vs ~200 with LangChain). Slower to build.
**When to reconsider:** After building this, a LangChain/LlamaIndex rebuild would be a valuable comparison exercise — understanding what the framework abstracts and where it helps vs. hinders.

### Custom Eval vs RAGAS
**Chose:** Custom harness.
**Why:** RAGAS is a black box (calls `ragas.evaluate()`, returns numbers without showing scoring logic). Custom metrics are 5 lines each, fully transparent, and require no LLM for scoring (free, deterministic, fast).
**Cost:** No LLM-as-judge faithfulness scoring — our heuristic catches term-level hallucinations but misses semantic contradictions.
**When to reconsider:** Add RAGAS alongside custom eval when you need deeper faithfulness scoring and have budget for LLM-as-judge API calls.

### Qdrant vs Pinecone
**Chose:** Qdrant (open-source, self-hosted).
**Why:** Free, runs locally or in Docker, good Python SDK, on-disk persistence without a server.
**Cost:** More setup than Pinecone (which is fully managed).
**When to reconsider:** If deploying to production at scale, Pinecone eliminates infrastructure management.

### Reranker On vs Off by Default
**Chose:** Off by default in API.
**Why:** ~30 seconds per query on CPU is unacceptable for interactive use. Recall@5 is still 0.914 without it.
**Cost:** Recall drops from 1.000 to 0.914 (3 out of 35 questions miss a relevant chunk in top-5).
**When to reconsider:** With a GPU, reranker inference drops to <1s — enable by default.

### Groq (Llama 3.3 70B) vs OpenAI/Claude
**Chose:** Groq with open-weight Llama.
**Why:** Free tier, fast inference, model transparency. Good for a learning project and cost-sensitive use.
**Cost:** 70B model hallucinated 0 times in limited testing, but 8B hallucinated 14%. Proprietary models (GPT-4o, Claude) likely have lower hallucination rates for grounded generation.
**When to reconsider:** For a production compliance tool where hallucination is high-risk, a stronger model with better instruction following (Claude Sonnet, GPT-4o) would be worth the API cost.

---

## 5. What I'd Do Differently

1. **Use Elasticsearch instead of rank_bm25** — production-grade search with filtering, fuzzy matching, and built-in hybrid search. The `rank_bm25` library is a learning tool.

2. **Start with RAGAS alongside custom eval** — having both LLM-as-judge and heuristic metrics from the start would give a more complete quality picture.

3. **Use Cohere Rerank API instead of local cross-encoder** — eliminates the CPU bottleneck entirely. One API call vs. 30 seconds of local inference.

4. **Add a web frontend** — even a simple Streamlit/Gradio app would make the system more demonstrable than `curl` commands.

5. **Implement proper rate limiting and auth on the API** — the current API is open to anyone. Production needs API keys and request throttling.

6. **Add structured logging alongside Langfuse** — Langfuse is great for LLM-specific observability but doesn't replace traditional application logging (errors, warnings, request counts).

---

## 6. Production Readiness Checklist

| Requirement | Status | Gap |
|---|---|---|
| Core pipeline (retrieve + generate) | ✅ Done | — |
| Grounded citations | ✅ Done | — |
| Hallucination detection | ✅ Done | LLM-as-judge would be more thorough |
| Eval harness | ✅ Done | Could add RAGAS |
| API serving | ✅ Done | No auth, no rate limiting |
| Containerization | ✅ Done | Docker Compose, not Kubernetes |
| Observability | ✅ Done | No infrastructure monitoring (Prometheus) |
| CI/CD quality gate | ✅ Done | — |
| Feedback loop | ✅ Done | Manual promotion, could automate with confidence threshold |
| Auto-scaling | ❌ Not done | Would need Kubernetes |
| Authentication | ❌ Not done | API is publicly accessible |
| Load testing | ❌ Not done | Unknown concurrent user capacity |
| Data versioning | ❌ Not done | Would use DVC |
| Experiment tracking | ❌ Not done | Would use MLflow or W&B |
| Frontend | ❌ Not done | API-only, would add Streamlit/Gradio |

---

## 7. File Structure

```
dfsa-rag/
├── api/                        # FastAPI serving layer
│   ├── app.py                  # Endpoints: /health, /ask, /feedback
│   ├── models.py               # Pydantic request/response schemas
│   └── tracing.py              # Langfuse tracing wrapper
├── retrieval/                  # Hybrid retrieval pipeline
│   ├── embedder.py             # BGE + OpenAI embedding (pluggable)
│   ├── dense.py                # Qdrant vector search
│   ├── sparse.py               # BM25 keyword search
│   └── hybrid.py               # RRF fusion + cross-encoder reranker
├── generation/                 # Grounded generation
│   ├── llm.py                  # Groq LLM client (pluggable)
│   ├── prompt.py               # System prompt + chunk formatting
│   └── generate.py             # Orchestrator + citation guardrails
├── eval/                       # Evaluation infrastructure
│   ├── eval_set.json           # 36 Q&A pairs (growing via feedback)
│   ├── scoring.py              # Retrieval metrics (recall, precision, MRR)
│   ├── generation_scoring.py   # Generation metrics (citation, faithfulness)
│   └── runner.py               # Eval orchestrator (pluggable retrieve_fn)
├── scripts/                    # CLI utilities
│   ├── extract_pages.py        # PDF → raw text
│   ├── chunk_rules.py          # Raw text → rule-level chunks
│   ├── validate_chunks.py      # Chunk quality checks
│   ├── index_chunks.py         # Build local indices
│   ├── index_qdrant_server.py  # Populate Dockerized Qdrant
│   ├── run_eval.py             # Retrieval eval CLI
│   ├── run_gen_eval.py         # Generation eval CLI
│   ├── ci_eval.py              # CI regression gate
│   └── promote_feedback.py     # Feedback → eval set promotion
├── data/
│   └── raw/                    # Source PDFs (3 DFSA rulebook modules)
├── .github/workflows/eval.yml  # GitHub Actions CI
├── Dockerfile                  # API container image
├── docker-compose.yml          # API + Qdrant orchestration
├── requirements.txt            # Python dependencies
├── DESIGN.md                   # Architecture summary (this file's companion)
├── DECISIONS.md                # Running log of every decision made
└── CLAUDE.md                   # Project charter and working agreement
```
