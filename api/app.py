"""
FastAPI application for the DFSA crypto-token RAG system.

Endpoints:
  GET  /health  — liveness check (verifies Qdrant connection)
  POST /ask     — answer a regulatory question with citations

The heavy resources (BGE model, Qdrant connection, BM25 index, LLM
client) are loaded once at startup via the lifespan context manager,
not per-request. First request after startup may be slightly slower
due to PyTorch JIT compilation.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import AskRequest, AskResponse, ChunkInfo, FeedbackRequest, FeedbackResponse
from api.tracing import traced_ask, flush as flush_traces
from generation.llm import GroqClient
from retrieval.dense import DenseRetriever
from retrieval.embedder import BGEEmbedder
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseRetriever

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
BM25_PATH = DATA_DIR / "bm25_index.pkl"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def wait_for_qdrant(client, retries: int = 5, delay: float = 2.0):
    """Retry Qdrant connection — container may not be ready yet."""
    for attempt in range(retries):
        try:
            client.get_collections()
            return True
        except Exception:
            if attempt < retries - 1:
                print(f"Qdrant not ready, retrying in {delay}s... ({attempt + 1}/{retries})")
                time.sleep(delay)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all resources at startup, clean up at shutdown."""
    print("Loading chunks...")
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    app.state.chunk_store = {c["chunk_id"]: c for c in chunks}
    app.state.chunk_lookup = {c["chunk_id"]: c["text"] for c in chunks}

    print("Loading BGE embedder...")
    embedder = BGEEmbedder()

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    dense = DenseRetriever(embedder, qdrant_url=QDRANT_URL)
    if not wait_for_qdrant(dense.client):
        raise RuntimeError("Could not connect to Qdrant")

    print("Loading BM25 index...")
    sparse = SparseRetriever()
    sparse.load(str(BM25_PATH))

    print("Building retrievers...")
    app.state.retriever = HybridRetriever(
        dense=dense,
        sparse=sparse,
        chunk_lookup=app.state.chunk_lookup,
        use_reranker=False,
    )
    app.state.retriever_with_reranker = HybridRetriever(
        dense=dense,
        sparse=sparse,
        chunk_lookup=app.state.chunk_lookup,
        use_reranker=True,
    )

    print(f"Initializing LLM ({GROQ_MODEL})...")
    app.state.llm = GroqClient(model=GROQ_MODEL)

    print("Ready to serve requests.")
    yield
    print("Flushing traces...")
    flush_traces()
    print("Shutting down.")


app = FastAPI(
    title="DFSA Crypto-Token RAG API",
    description="Answers questions about DFSA crypto-token regulations with citations",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness check — verifies Qdrant connection."""
    try:
        collections = app.state.retriever.dense.client.get_collections()
        return {"status": "ok", "qdrant_collections": len(collections.collections)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unhealthy: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a regulatory question with grounded citations."""
    retriever = (
        app.state.retriever_with_reranker if req.use_reranker
        else app.state.retriever
    )

    try:
        result = traced_ask(
            question=req.question,
            retriever=retriever,
            chunk_store=app.state.chunk_store,
            llm=app.state.llm,
            use_reranker=req.use_reranker,
            top_k=req.top_k,
        )
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(status_code=429, detail="LLM rate limit reached. Try again shortly.")
        raise HTTPException(status_code=502, detail=f"LLM error: {error_msg}")

    return AskResponse(
        answer=result["answer"],
        model=result["model"],
        cited_rules=result["cited_rules"],
        citation_check=result["citation_check"],
        retrieved_chunks=[
            ChunkInfo(
                chunk_id=c["chunk_id"],
                rule_id=c["rule_id"],
                section_title=c.get("section_title", ""),
                context_header=c.get("context_header", ""),
                text=c["text"],
            )
            for c in result["retrieved_chunks"]
        ],
        trace_id=result.get("trace_id", ""),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Record user feedback on an answer (thumbs up/down + optional correction)."""
    import uuid
    from datetime import datetime, timezone

    feedback_dir = DATA_DIR / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = feedback_dir / "feedback.jsonl"

    feedback_id = str(uuid.uuid4())[:8]
    entry = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating,
        "corrected_answer": req.corrected_answer,
        "expected_chunks": req.expected_chunks,
        "comment": req.comment,
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(status="recorded", feedback_id=feedback_id)
