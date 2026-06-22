# Project: DFSA Crypto-Token Regulatory RAG System

## What this is
A production-grade Retrieval-Augmented Generation system that answers questions
about DFSA (Dubai Financial Services Authority) crypto-token regulations, with
full MLOps scaffolding: evaluation harness, observability, CI/CD regression
gates, and a feedback loop. This is a portfolio/learning project targeting AI
Engineering roles in Gulf, US, and EU markets — regulated-domain RAG with real
eval/governance is one of the highest-leverage skill signals in all three
markets right now.

## How Umair wants to work — READ THIS FIRST, APPLIES TO EVERY SESSION
This project exists so Umair can **learn** MLOps and ML system design deeply —
not to hand him a finished artifact. That changes how you should behave:

- **Do not implement a full phase autonomously just because it was mentioned.**
  Before writing code for any non-trivial step, explain the approach, the
  alternatives considered, and why you're recommending this one. Wait for
  Umair to confirm or push back before writing files.
- **Surface decisions, don't make them silently.** If there's a real design
  choice (chunking strategy, model choice, library choice, schema design),
  stop and ask rather than picking a sensible default and moving on.
- **Small, reviewable diffs.** Prefer several small commits over one big one.
  Umair wants to read every diff and understand what changed and why.
- **Explain unfamiliar concepts briefly, in plain language** as they come up.
  Umair is an MS AI student strong in CV/deep learning/generative media
  (previously AI Product Specialist at Vyro.ai) — treat ML concepts as known,
  but treat production/MLOps infra concepts (serving, observability, CI/CD
  for ML, vector DB internals) as genuinely new and worth explaining.
- **Update DECISIONS.md** whenever a real decision gets made — one line,
  plain language, dated. Don't wait until the end to reconstruct this.
- If Umair explicitly says "just do it" or "your call" for something
  low-stakes, you can move faster — but default to the collaborative mode
  above.
- Never silently swap a locked decision (see "Locked scope" and "Stack"
  below) for a different approach without flagging it first.

## Current status
- [x] Phase 0: Scope locked, repo scaffolding done
- [x] Phase 1: Ingestion & chunking (52 rule-level chunks in data/processed/chunks.jsonl)
- [x] Phase 2: Eval harness (35 Q&A pairs, custom scoring, pluggable retriever interface)
- [x] Phase 3: Retrieval (BM25 + BGE dense + RRF fusion + bge-reranker)
- [x] Phase 4: Grounded generation (Groq/Llama, citation guardrails, 8B vs 70B eval)
- [x] Phase 5: Serving layer (FastAPI + Docker Compose + Qdrant server)
- [x] Phase 6: Observability (Langfuse tracing, latency/token/cost breakdown)
- [x] Phase 7: CI/CD regression gate (GitHub Actions, recall + citation thresholds)
- [x] Phase 8: Feedback loop (POST /feedback + promote_feedback.py → growing eval set)
- [ ] Phase 9: System design doc (tradeoffs made/rejected at each phase)

## Locked scope (v1 corpus) — DO NOT EXPAND WITHOUT ASKING
Not the full DFSA rulebook. Scoped to the **crypto-token regulatory framework**:
- **GEN 3A** — Crypto Token Requirements (definitions, recognition criteria)
- **COB 15** — Conduct rules for crypto businesses (disclosures, tech audits)
- **AML 9.3A / 9.3B** — Travel Rule requirements for crypto/NFT transfers

Rationale: topically self-contained (a real compliance officer's questions
would stay within this scope), heavily cross-referenced between modules
(GEN 3A <-> COB <-> AML — genuine chunking difficulty), thematically current.
Can expand to other modules later without changing pipeline architecture.

## Target product
A system where a compliance officer asks something like "what disclosure
requirements apply to a crypto custody license under DFSA rules?" and gets a
grounded answer with citations to the specific clause, plus the full
production scaffolding around it (eval, monitoring, CI/CD, governance log).

## Why this project is "industry-level," not a RAG demo
The naive version is chunk -> embed -> stuff into a prompt -> done. The real
engineering — and what interviewers actually probe for — is everything
around that loop: structure-aware chunking for cross-referenced legal text,
hybrid retrieval (vector misses exact terms like "Rule 4.2.1"), an eval
harness built BEFORE any tuning happens (can't improve what you can't
measure), forced citation grounding, hallucination guardrails, observability
(where did latency/cost actually go), and CI/CD that treats "did this change
make answers worse" as a testable regression, not a vibe.

## Stack
**Locked:**
- Vector DB: Qdrant
- Sparse retrieval: BM25 (rank_bm25, or OpenSearch if we want it closer to
  production-grade later)
- Reranker: cross-encoder (bge-reranker)
- Orchestration: raw Python first, NOT LangChain/LlamaIndex — Umair wants to
  understand the primitives before adopting a framework that hides them.
  May introduce LangChain/LlamaIndex later as a deliberate comparison.
- Serving: FastAPI + Docker (docker-compose for the full local stack)
- CI/CD: GitHub Actions

**Open — discuss before locking:**
- Embeddings: open-source (bge-large, self-hosted) vs API embeddings —
  tradeoff (cost, latency, setup effort) not yet decided
- LLM: Claude API (needs Umair's own API key) vs open-weight via
  Groq/Together for cost/latency comparison
- Eval framework: RAGAS vs custom harness
- Tracing/observability: Langfuse vs Arize Phoenix

## Background context
Umair: MS AI student at LUMS (1st year complete, 3.88 GPA), BS EE from NUST.
Previously AI Product Specialist at Vyro.ai (imagine.art) — image/video
generation features (inpainting, upscaling, LoRA, pose-guided generation).
This project is deliberately a different domain (regulatory/legal text RAG)
to diversify his portfolio beyond CV/generative media. Targeting Gulf, US,
and EU remote AI Engineering roles. Separately running a research project
(bodySITARA — privacy-preserving full-body anonymization, IEEE PerCom 2027
target, Sept 11 2026 submission deadline) — unrelated to this project except
as shared context for his general skill level and current workload.

## See also
- `DECISIONS.md` — running, dated log of every real decision made
