# Decisions Log

Format: `YYYY-MM-DD — Decision — short rationale`

2026-06-21 — Project domain: regulatory/compliance RAG, not CV/generative
media — deliberate portfolio diversification; Gulf/US/EU markets all pay a
premium for regulated-domain RAG with real eval/governance right now.

2026-06-21 — Corpus: DFSA (Dubai) only, not cross-jurisdiction (SAMA/ADGM
also considered) — keeps v1 scope provable; can expand later without
changing pipeline architecture.

2026-06-21 — Corpus scope narrowed further to the crypto-token regulatory
framework (GEN 3A, COB 15, AML 9.3A/9.3B) rather than the full DFSA
rulebook — topically self-contained, heavily cross-referenced (real
chunking difficulty), thematically current.

2026-06-21 — Orchestration: raw Python first, not LangChain/LlamaIndex —
priority is understanding retrieval/generation primitives directly;
framework adoption deferred to a later, deliberate comparison.

2026-06-21 — Working agreement with Claude Code: explain-before-implement,
no autonomous full-phase builds, decisions surfaced not auto-resolved, small
reviewable commits. See CLAUDE.md for full detail.

2026-06-21 — PDF parsing: PyMuPDF (fitz) — fast, C-based, good text
extraction. pdfplumber rejected (better tables, but our sections have no
tables). unstructured.io rejected (heavy deps, hides primitives, fights
the "understand before abstracting" goal).

2026-06-21 — Included GEN App2 A2.5 (definitions) in corpus — GEN 3A
constantly references it ("Crypto Token is defined in section A2.5").
Without it, can't answer definition questions.

2026-06-21 — Chunking strategy: rule-level (one chunk = one DFSA rule
like "3A.2.1" plus its Guidance block). Alternatives rejected: fixed-size
(breaks rules mid-sentence, separates guidance from rule — worst for legal
text), hierarchical tree (marginal benefit for ~52 chunks, extra parser
complexity). Context header breadcrumb prepended to aid retrieval.

2026-06-21 — Guidance blocks merged with parent rule, not separate chunks
— they are semantically inseparable (guidance explains the rule it follows).

2026-06-21 — Output format: JSONL (one chunk per line). Alternatives
rejected: CSV (nested fields don't fit), single JSON array (not
append-friendly), SQLite (overkill for 52 chunks), per-chunk markdown
files (metadata needs frontmatter parsing).

2026-06-21 — Page ranges hardcoded per PDF version (GEN/VER71, COB/VER50,
AML/VER30) rather than dynamic section detection — simpler, more
auditable for 3 rarely-updated PDFs.

2026-06-21 — Eval framework: custom harness in raw Python, not RAGAS —
matches "understand primitives" philosophy. RAGAS is a black box
(ragas.evaluate() returns numbers without showing scoring logic) and
requires an LLM for scoring (cost per eval run). Custom harness is ~200
lines, fully transparent, and pluggable via a simple callable interface.

2026-06-21 — Eval set: 35 hand-written Q&A pairs covering 7 categories
(definitional, prohibition, custody, disclosure, AML, cross-reference,
procedural). Each entry has expected_chunks, expected_answer, category,
difficulty. Will grow via Phase 8 feedback loop.

2026-06-21 — Eval metrics (retrieval): recall@k, precision@k, MRR.
Recall@k is most important ("did we find the right chunks?"). Generation
metrics (faithfulness, relevance) deferred to Phase 4.

2026-06-21 — Eval harness architecture: runner takes a pluggable
retrieve_fn(question, k) -> list[chunk_id]. Any retrieval implementation
(BM25, vector, hybrid) just conforms to this signature. No class
hierarchy needed.

2026-06-21 — Embeddings: both BGE-large-en-v1.5 (local, 1024d) and
OpenAI text-embedding-3-small (API, 1536d) implemented for comparison.
BGE is the primary — free, runs locally, top-tier for its size.

2026-06-21 — LLM for generation: Groq/Together with open-weight model
(deferred to Phase 4).

2026-06-21 — Qdrant on-disk mode (path=data/qdrant_bge) rather than
Docker — persistence without server overhead for 52 chunks. Switch to
Docker in Phase 5 (serving).

2026-06-21 — Hybrid fusion: Reciprocal Rank Fusion (RRF) with k=60.
Alternatives rejected: linear combination (requires normalizing BM25
and cosine scores to same scale — fragile). RRF uses rank positions
only, is parameter-free, and is the industry standard.

2026-06-21 — Cross-encoder reranker: BAAI/bge-reranker-v2-m3. Applied
only to the merged shortlist (~30 candidates), not the full corpus.

2026-06-21 — Context header prepended to chunk text before embedding
(e.g., "GEN > 3A Crypto Token Requirements > 3A.2 Prohibitions | ...")
to give vectors structural context.

2026-06-21 — Retrieval eval results (BGE embeddings):
  BM25 only:              recall@5=0.871, MRR=0.720
  Dense (BGE) only:       recall@5=0.943, MRR=0.864
  Hybrid (no reranker):   recall@5=0.914, MRR=0.833
  Hybrid + reranker:      recall@5=1.000, MRR=0.882  ← perfect recall

Key findings:
  - BM25 weak on definitional questions (recall@5=0.50) — "Crypto Token"
    appears everywhere, BM25 can't discriminate.
  - Dense excels at semantics but missed 2 questions.
  - Hybrid without reranker *worse* than dense alone — RRF dilutes strong
    dense results with weak BM25 candidates.
  - Reranker fixes everything — recall@5=1.0 across all 7 categories.
    This validates the two-stage pipeline: fast bi-encoder narrows down,
    accurate cross-encoder re-sorts.
  - Reranker is slow on CPU (~20 min for 35 questions). GPU would be ~30s.

2026-06-22 — LLM: Groq API with Llama 3.3 70B (primary) and Llama 3.1
8B (comparison). Groq chosen for free tier access and fast inference.
Both models implemented for comparison via pluggable GroqClient.

2026-06-22 — Generation: zero-shot with strong system prompt, not
few-shot. Reasons: (a) chunks are long, few-shot examples would consume
context window; (b) Llama 3.3 70B follows system instructions well;
(c) expected answers in eval set vary in style, few-shot would impose
artificial constraints. Can add few-shot later as isolated change.

2026-06-22 — Hallucination guardrails: (1) system prompt grounding
("answer ONLY from provided context"), (2) citation extraction regex
post-generation, (3) citation validity check against retrieved chunks.
LLM-as-judge for deep faithfulness available as opt-in (--llm-judge)
but default is heuristic (free, fast, deterministic).

2026-06-22 — Chunk format in prompt includes rule_id and context_header
(not just text). This gives the LLM: citation targets (what to cite),
structural context (where the rule sits), and scope awareness.

2026-06-22 — Generation eval metrics: citation_precision (are citations
valid?), citation_recall (did it cite the right rules?),
faithfulness_heuristic (do key terms in the answer appear in chunks?).
Separate file from retrieval scoring (different abstraction level).

2026-06-22 — Generation eval results (hybrid no-reranker retrieval):

  Llama 3.3 70B (partial — 7/35 questions, hit daily token limit):
    citation_precision:  1.000 (0 hallucinated citations)
    faithfulness:        ~0.84

  Llama 3.1 8B (full 35 questions):
    citation_precision:  0.929
    citation_recall:     0.829
    faithfulness:        0.978
    hallucinated:        5/35 questions

  8B hallucinated rules: A2.5.2, 2.13.1, 5.2.1, 3.4.2, 14.2.3,
  15.2.1, 14.3.3, 15.4.2. Worst case: proc_01 cited 5 rules, only
  1 was valid (precision=0.20).

Key findings:
  - 70B had 0 hallucinations in 7 questions; 8B had 5 in 35 (14%).
  - 8B invents plausible-sounding rule numbers from adjacent sections.
  - Citation guardrail catches every hallucination mechanically.
  - System prompt grounding works well for both models (faithfulness
    0.978 for 8B) — the models stay on-topic, but 8B fabricates
    citations more often.
  - Recommendation: use 70B for production, 8B only for fast iteration.

2026-06-22 — Serving: FastAPI + Docker Compose. Two services: API
(FastAPI/uvicorn) and Qdrant (official image). API loads BGE model,
reranker, BM25 index, and Groq client at startup via lifespan context
manager.

2026-06-22 — Qdrant switched from on-disk to server mode. DenseRetriever
now accepts qdrant_url parameter. On-disk mode still works for scripts.
Server mode enables proper service separation and survives API restarts.

2026-06-22 — Reranker defaults to off in API (use_reranker=false).
~30s per request on CPU is too slow for interactive use. Users can
opt in via request body. Both retriever instances (with/without
reranker) pre-loaded at startup to avoid per-request model loading.

2026-06-22 — API tested end-to-end: health check + /ask endpoint
with Llama 3.3 70B via Groq. Correct answer with valid citations
for "Are Privacy Tokens permitted in the DIFC?" — cites Rule 3A.2.2,
all citations valid, answer grounded in retrieved context.

2026-06-22 — Observability: Langfuse (cloud, free tier). Alternatives
rejected: Arize Phoenix (heavier setup, local server), custom logging
(no dashboard, no cost tracking). Langfuse is purpose-built for LLM/RAG
observability with built-in trace/span/generation types.

2026-06-22 — Tracing architecture: wrapper layer in api/tracing.py,
not inline in retrieval/generation modules. Only change to existing code:
llm.py now exposes token usage via complete_with_usage(). Tracing is
opt-in — if LANGFUSE_PUBLIC_KEY is not set, pipeline works normally.

2026-06-22 — Langfuse SDK v4 uses OpenTelemetry-based context managers
(start_as_current_observation) instead of v2's trace/span methods.
Spans use typed observation types: "retriever", "generation", "guardrail".

2026-06-22 — CI/CD: GitHub Actions workflow (.github/workflows/eval.yml)
runs on every push/PR to main. Full pipeline: ingest → index (in-memory
Qdrant) → retrieval eval → generation eval via Groq. Thresholds:
recall@5 >= 0.85, citation_precision >= 0.90. Fails the build if any
threshold is missed.

2026-06-22 — CI uses in-memory Qdrant (:memory:) instead of Docker
Qdrant — simpler for GitHub runners, no docker-compose needed in CI.
BGE model cached via actions/cache for faster subsequent runs.

2026-06-22 — GROQ_API_KEY stored as GitHub secret, not in repo.
Generation eval is skipped gracefully if key is missing (retrieval
eval still runs).

2026-06-22 — GitHub repo: https://github.com/umairs5/dfsa-rag

<!-- Next entries: feedback loop. -->
