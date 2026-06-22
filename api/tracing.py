"""
Langfuse tracing for the /ask pipeline (Langfuse SDK v4).

Wraps retrieval and generation with Langfuse spans so you can see
in the dashboard: where time went, token usage, cost, and whether
citations were hallucinated.

Gracefully degrades: if LANGFUSE_PUBLIC_KEY is not set, tracing is
silently skipped and the pipeline runs normally.

Langfuse v4 uses context managers (start_as_current_observation)
instead of the old trace/span methods from v2.
"""

import os
import time
import uuid

from dotenv import load_dotenv

from generation.generate import generate_answer

load_dotenv()

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

_langfuse = None


def get_langfuse():
    """Get or create the Langfuse client singleton."""
    global _langfuse
    if not LANGFUSE_ENABLED:
        return None
    if _langfuse is None:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse()
        except Exception:
            return None
    return _langfuse


def flush():
    """Flush pending traces to Langfuse (call on shutdown)."""
    lf = get_langfuse()
    if lf:
        lf.flush()


def traced_ask(
    question: str,
    retriever,
    chunk_store: dict,
    llm,
    use_reranker: bool,
    top_k: int,
) -> dict:
    """Run the full ask pipeline with Langfuse tracing."""
    lf = get_langfuse()

    if not lf:
        return _untraceable_ask(question, retriever, chunk_store, llm, top_k)

    # Root trace span
    with lf.start_as_current_observation(
        name="ask",
        as_type="span",
        input={"question": question},
        metadata={"use_reranker": use_reranker, "top_k": top_k},
    ) as trace:

        # --- Retrieval span ---
        t0 = time.time()
        chunk_ids = retriever.retrieve(question, k=top_k)
        retrieval_ms = (time.time() - t0) * 1000

        retrieved_chunks = [
            chunk_store[cid] for cid in chunk_ids if cid in chunk_store
        ]

        retrieval_obs = lf.start_observation(
            name="retrieval",
            as_type="retriever",
            input={"question": question},
            output={"chunk_ids": chunk_ids},
            metadata={
                "use_reranker": use_reranker,
                "top_k": top_k,
                "num_chunks": len(retrieved_chunks),
                "latency_ms": round(retrieval_ms, 1),
            },
        )
        retrieval_obs.end()

        # --- Generation span ---
        t0 = time.time()
        gen_result = generate_answer(question, retrieved_chunks, llm)
        generation_ms = (time.time() - t0) * 1000

        usage = gen_result.get("usage", {})
        gen_obs = lf.start_observation(
            name="llm",
            as_type="generation",
            input=question[:500],
            output=gen_result["answer"],
            model=gen_result["model"],
            usage_details={
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            },
            metadata={"latency_ms": round(generation_ms, 1)},
        )
        gen_obs.end()

        # --- Citation check span ---
        citation_check = gen_result["citation_check"]
        cit_obs = lf.start_observation(
            name="citation_check",
            as_type="guardrail",
            input={"cited_rules": gen_result["cited_rules"]},
            output=citation_check,
        )
        cit_obs.end()

        # --- Score the trace ---
        all_valid = citation_check.get("all_valid", False)
        lf.score_current_trace(
            name="hallucination_free",
            value=1.0 if all_valid else 0.0,
        )

        # Update trace output
        trace.update(
            output={
                "answer": gen_result["answer"][:200],
                "cited_rules": gen_result["cited_rules"],
                "hallucinated": not all_valid,
            },
        )

    trace_id = lf.get_current_trace_id() or str(uuid.uuid4())

    return {
        "answer": gen_result["answer"],
        "model": gen_result["model"],
        "cited_rules": gen_result["cited_rules"],
        "citation_check": citation_check,
        "retrieved_chunks": retrieved_chunks,
        "usage": usage,
        "trace_id": trace_id,
    }


def _untraceable_ask(question, retriever, chunk_store, llm, top_k):
    """Fallback when Langfuse is not configured."""
    chunk_ids = retriever.retrieve(question, k=top_k)
    retrieved_chunks = [
        chunk_store[cid] for cid in chunk_ids if cid in chunk_store
    ]
    gen_result = generate_answer(question, retrieved_chunks, llm)
    return {
        "answer": gen_result["answer"],
        "model": gen_result["model"],
        "cited_rules": gen_result["cited_rules"],
        "citation_check": gen_result["citation_check"],
        "retrieved_chunks": retrieved_chunks,
        "usage": gen_result.get("usage", {}),
        "trace_id": str(uuid.uuid4()),
    }
