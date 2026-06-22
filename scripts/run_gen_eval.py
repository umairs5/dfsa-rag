"""
End-to-end generation eval: retrieve → generate → score.

Runs the full RAG pipeline on each eval question and produces a
scorecard with both retrieval and generation metrics.

Usage:
  python scripts/run_gen_eval.py --model llama-3.3-70b-versatile
  python scripts/run_gen_eval.py --model llama-3.1-8b-instant
  python scripts/run_gen_eval.py --model llama-3.3-70b-versatile --delay 3

Options:
  --model    Groq model ID
  --delay    Seconds between API calls (default: 2, for rate limits)
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import load_eval_set
from eval.scoring import score_single
from eval.generation_scoring import (
    citation_precision,
    citation_recall,
    faithfulness_heuristic,
)
from generation.llm import GroqClient
from generation.generate import generate_answer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
DATA_DIR = ROOT / "data"

ALL_CHUNKS = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
CHUNK_STORE = {c["chunk_id"]: c for c in ALL_CHUNKS}


def build_retriever(use_reranker: bool = True):
    """Build the hybrid retriever."""
    from retrieval.embedder import BGEEmbedder
    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever
    from retrieval.hybrid import HybridRetriever

    embedder = BGEEmbedder()
    dense = DenseRetriever(embedder, qdrant_path=str(DATA_DIR / "qdrant_bge"))
    sparse = SparseRetriever()
    sparse.load(str(DATA_DIR / "bm25_index.pkl"))
    chunk_lookup = {c["chunk_id"]: c["text"] for c in ALL_CHUNKS}

    return HybridRetriever(
        dense=dense,
        sparse=sparse,
        chunk_lookup=chunk_lookup,
        use_reranker=use_reranker,
    )


def main():
    parser = argparse.ArgumentParser(description="End-to-end generation eval")
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model ID",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between API calls (Groq rate limit)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip cross-encoder reranking (much faster on CPU)",
    )
    args = parser.parse_args()

    print(f"Loading retriever (reranker={'off' if args.no_reranker else 'on'})...")
    retriever = build_retriever(use_reranker=not args.no_reranker)

    print(f"Initializing LLM: {args.model}")
    llm = GroqClient(model=args.model)

    entries = load_eval_set()
    results = []

    print(f"\nRunning {len(entries)} questions...\n")

    for i, entry in enumerate(entries):
        qid = entry["id"]
        question = entry["question"]

        # Retrieve
        chunk_ids = retriever.retrieve(question, k=5)
        retrieved_chunks = [CHUNK_STORE[cid] for cid in chunk_ids if cid in CHUNK_STORE]

        # Retrieval metrics
        ret_scores = score_single(chunk_ids, entry["expected_chunks"])

        # Generate
        gen_result = generate_answer(question, retrieved_chunks, llm)

        # Generation metrics
        cit_prec = citation_precision(
            gen_result["cited_rules"], retrieved_chunks
        )
        cit_rec = citation_recall(
            gen_result["cited_rules"], entry["expected_chunks"]
        )
        faith = faithfulness_heuristic(gen_result["answer"], retrieved_chunks)

        result = {
            "id": qid,
            "question": question,
            "category": entry["category"],
            "answer_preview": gen_result["answer"][:120],
            "cited_rules": gen_result["cited_rules"],
            "citation_check": gen_result["citation_check"],
            **ret_scores,
            "citation_precision": cit_prec,
            "citation_recall": cit_rec,
            "faithfulness": faith,
        }
        results.append(result)

        status = "OK" if gen_result["citation_check"]["all_valid"] else "HALLUCINATED"
        print(f"  [{i+1}/{len(entries)}] {qid:<12} {status:<14} cit_p={cit_prec:.2f} faith={faith:.2f}")

        if i < len(entries) - 1:
            time.sleep(args.delay)

    # --- Aggregate ---
    print("\n" + "=" * 60)
    print(f"GENERATION EVAL — {args.model}")
    print("=" * 60)

    metric_keys = ["recall@5", "reciprocal_rank", "citation_precision",
                   "citation_recall", "faithfulness"]
    for key in metric_keys:
        vals = [r[key] for r in results]
        print(f"  {key:<22} {statistics.mean(vals):.3f}")

    # Hallucination count
    hallucinated = [r for r in results if not r["citation_check"]["all_valid"]]
    print(f"\n  Hallucinated citations: {len(hallucinated)}/{len(results)} questions")
    for r in hallucinated:
        print(f"    [{r['id']}] hallucinated: {r['citation_check']['hallucinated_citations']}")

    # Per-category
    print(f"\n  By category:")
    print(f"  {'Category':<16} {'Count':>5}  {'CitPrec':>8}  {'CitRec':>8}  {'Faith':>8}")
    print(f"  {'-'*16} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    cats = sorted(set(r["category"] for r in results))
    for cat in cats:
        cat_rs = [r for r in results if r["category"] == cat]
        cp = statistics.mean([r["citation_precision"] for r in cat_rs])
        cr = statistics.mean([r["citation_recall"] for r in cat_rs])
        f = statistics.mean([r["faithfulness"] for r in cat_rs])
        print(f"  {cat:<16} {len(cat_rs):>5}  {cp:>8.3f}  {cr:>8.3f}  {f:>8.3f}")

    # Sample answers
    print(f"\n  Sample answers:")
    for r in results[:3]:
        print(f"    [{r['id']}] {r['question'][:50]}")
        print(f"      >{r['answer_preview']}...")
        print()


if __name__ == "__main__":
    main()
