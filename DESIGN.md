# System Design: DFSA Crypto-Token RAG

A production-grade Retrieval-Augmented Generation system for DFSA regulatory compliance, with evaluation, observability, CI/CD, and a feedback loop.

## Architecture

```
                     POST /ask
                        |
                   [FastAPI API]
                        |
          +-------------+-------------+
          |                           |
    [Hybrid Retrieval]          [Generation]
     |        |                       |
  [BM25]  [Qdrant]              [Groq API]
     |        |                  Llama 3.3 70B
     +---+----+                       |
         |                     [Citation Check]
    [RRF Fusion]                      |
         |                     [Langfuse Trace]
    [Reranker]
   (optional)

  POST /feedback ──> feedback.jsonl ──> promote_feedback.py ──> eval_set.json
                                                                     |
  GitHub Actions CI <────────────────────────────────────────────────+
```

## Corpus

3 DFSA rulebook PDFs, scoped to crypto-token regulations:
- **GEN 3A** — Crypto Token Requirements (prohibitions, suitability)
- **GEN App2 A2.5** — Definitions (Crypto Token, NFT, Utility Token, Fiat Crypto Token)
- **COB 15** — Conduct rules for crypto businesses (custody, disclosure, MTF, tech governance)
- **AML 9.3A/9.3B** — Travel rule for crypto transfers

**52 rule-level chunks**, median 244 tokens. Each chunk = one DFSA rule + its Guidance block.

## Key Decisions

| Decision | Choice | Rejected Alternative | Why |
|---|---|---|---|
| Chunking | Rule-level (one rule + guidance) | Fixed-size 500 tokens | Legal text: the rule is the natural citation unit. Fixed-size splits rules mid-sentence |
| Embeddings | BGE-large-en-v1.5 (local) | OpenAI API | Free, runs locally, top-tier quality. OpenAI also implemented for comparison |
| Sparse search | BM25 (rank_bm25) | Elasticsearch | Learning project: understand BM25 primitives before using a search engine |
| Fusion | Reciprocal Rank Fusion | Linear combination | RRF uses rank positions (not scores), avoids BM25/cosine scale mismatch |
| Reranker | bge-reranker-v2-m3 | Cohere Rerank API | Open-source, free. But ~30s/query on CPU — defaults to off in API |
| LLM | Groq (Llama 3.3 70B) | OpenAI/Claude | Free tier, fast inference. Open-weight for transparency |
| Eval | Custom harness | RAGAS | Understand the metrics before using a framework. RAGAS is a black box |
| Observability | Langfuse | Arize Phoenix, custom logs | Purpose-built for LLM/RAG, free tier, trace + generation + guardrail span types |
| Orchestration | Raw Python | LangChain/LlamaIndex | Understand primitives first. Framework adoption deferred to deliberate comparison |

## Eval Results

**Retrieval** (35 questions, hybrid without reranker):
| Mode | Recall@5 | MRR |
|---|---|---|
| BM25 only | 0.871 | 0.720 |
| Dense (BGE) only | 0.943 | 0.864 |
| Hybrid (no reranker) | 0.914 | 0.833 |
| Hybrid + reranker | **1.000** | 0.882 |

**Generation** (Llama 3.1 8B, 35 questions):
- Citation precision: 0.929 (5 hallucinated citations out of 35 questions)
- Faithfulness: 0.978
- 70B model: 0 hallucinations in 7 questions tested (rate limited)

## Hallucination Prevention

Three layers:
1. **System prompt grounding** — "Answer ONLY from provided context, cite rule numbers"
2. **Citation extraction + validation** — Regex extracts cited rules, checks they exist in retrieved chunks
3. **Faithfulness heuristic** — Verifies key legal terms in the answer appear in the source chunks

## Production Gaps (What I'd Add Next)

1. **Elasticsearch** instead of rank_bm25 for production-grade keyword search
2. **GPU inference** for the reranker (30s → <1s per query)
3. **LangChain/LlamaIndex** comparison to understand framework vs. primitive tradeoffs
4. **Kubernetes** deployment instead of Docker Compose
5. **RAGAS** alongside custom eval for LLM-as-judge faithfulness scoring
6. **Prompt versioning** via Langfuse instead of hardcoded prompt.py
7. **A/B testing** different models/prompts in production
8. **Rate limiting and auth** on the API

## Stack

Python 3.11 | FastAPI | Qdrant | sentence-transformers (BGE) | rank_bm25 | Groq API | Langfuse | Docker Compose | GitHub Actions

## How to Run

```bash
# Setup
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env  # add your keys

# Ingest + index
python scripts/extract_pages.py
python scripts/chunk_rules.py
docker compose up qdrant -d
python scripts/index_qdrant_server.py

# Serve
uvicorn api.app:app --port 8000

# Eval
python scripts/run_eval.py --mode hybrid --no-reranker
python scripts/run_gen_eval.py --model llama-3.1-8b-instant --no-reranker
```

See [docs/system-design.md](docs/system-design.md) for detailed architecture documentation.
