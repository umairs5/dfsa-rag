"""
CI regression gate: run retrieval + generation eval and assert thresholds.

This script is called by GitHub Actions on every push. It:
  1. Extracts text from PDFs and chunks them
  2. Builds BM25 + Qdrant (in-memory) indices
  3. Runs retrieval eval (35 questions, no reranker)
  4. Runs generation eval via Groq API
  5. Asserts quality thresholds — exits 1 (fail) if any threshold is missed

Thresholds (update these as quality improves):
  - recall@5 >= 0.85
  - citation_precision >= 0.90

Exit codes:
  0 = all thresholds met (CI passes)
  1 = at least one threshold missed (CI fails)

Usage: python scripts/ci_eval.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Thresholds ---
MIN_RECALL_AT_5 = 0.85
MIN_CITATION_PRECISION = 0.90

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"


def step_ingest():
    """Run extraction and chunking if chunks don't exist."""
    if CHUNKS_PATH.exists():
        print("Chunks already exist, skipping ingestion.")
        return

    print("=== Step 1: Extract pages ===")
    from scripts.extract_pages import main as extract_main
    extract_main()

    print("\n=== Step 2: Chunk rules ===")
    from scripts.chunk_rules import main as chunk_main
    chunk_main()


def step_retrieval_eval():
    """Build in-memory indices and run retrieval eval."""
    print("\n=== Step 3: Build indices (in-memory) ===")
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]

    from retrieval.embedder import BGEEmbedder
    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever
    from retrieval.hybrid import HybridRetriever

    embedder = BGEEmbedder()
    dense = DenseRetriever(embedder, qdrant_path=":memory:")
    dense.index(chunks)

    sparse = SparseRetriever()
    sparse.index(chunks)

    chunk_lookup = {c["chunk_id"]: c["text"] for c in chunks}
    retriever = HybridRetriever(
        dense=dense, sparse=sparse,
        chunk_lookup=chunk_lookup, use_reranker=False,
    )

    print("\n=== Step 4: Retrieval eval ===")
    from eval.runner import run_eval, print_scorecard

    retrieve_fn = lambda q, k: retriever.retrieve(q, k)
    results = run_eval(retrieve_fn, k=5)
    print_scorecard(results)

    recall = results["aggregate"]["recall@5"]
    return recall, retriever, chunks


def step_generation_eval(retriever, chunks):
    """Run generation eval via Groq."""
    print("\n=== Step 5: Generation eval ===")
    import os
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set, skipping generation eval.")
        return None

    from eval.runner import load_eval_set
    from eval.scoring import score_single
    from eval.generation_scoring import citation_precision
    from generation.llm import GroqClient
    from generation.generate import generate_answer

    chunk_store = {c["chunk_id"]: c for c in chunks}
    llm = GroqClient(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    entries = load_eval_set()

    cit_precisions = []
    for i, entry in enumerate(entries):
        chunk_ids = retriever.retrieve(entry["question"], k=5)
        retrieved = [chunk_store[cid] for cid in chunk_ids if cid in chunk_store]

        gen_result = generate_answer(entry["question"], retrieved, llm)
        cp = citation_precision(gen_result["cited_rules"], retrieved)
        cit_precisions.append(cp)

        status = "OK" if gen_result["citation_check"]["all_valid"] else "HALLUCINATED"
        print(f"  [{i+1}/{len(entries)}] {entry['id']:<12} {status:<14} cit_p={cp:.2f}")

        if i < len(entries) - 1:
            time.sleep(2)

    mean_cp = statistics.mean(cit_precisions)
    print(f"\n  Mean citation_precision: {mean_cp:.3f}")
    return mean_cp


def main():
    step_ingest()

    recall, retriever, chunks = step_retrieval_eval()

    cit_prec = step_generation_eval(retriever, chunks)

    # --- Assert thresholds ---
    print("\n" + "=" * 60)
    print("CI REGRESSION GATE")
    print("=" * 60)

    passed = True

    print(f"  recall@5:            {recall:.3f}  (threshold: >= {MIN_RECALL_AT_5})")
    if recall < MIN_RECALL_AT_5:
        print(f"    FAIL")
        passed = False
    else:
        print(f"    PASS")

    if cit_prec is not None:
        print(f"  citation_precision:  {cit_prec:.3f}  (threshold: >= {MIN_CITATION_PRECISION})")
        if cit_prec < MIN_CITATION_PRECISION:
            print(f"    FAIL")
            passed = False
        else:
            print(f"    PASS")
    else:
        print(f"  citation_precision:  SKIPPED (no GROQ_API_KEY)")

    if passed:
        print(f"\n  RESULT: ALL THRESHOLDS MET")
        sys.exit(0)
    else:
        print(f"\n  RESULT: THRESHOLD(S) MISSED — REGRESSION DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
