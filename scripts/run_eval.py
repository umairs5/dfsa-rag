"""
Run the eval harness with different retrieval strategies.

Modes:
  --mode random     Random baseline (should score ~0)
  --mode perfect    Oracle baseline (should score 1.0)
  --mode bm25       BM25 sparse retrieval only
  --mode dense      Dense vector retrieval only
  --mode hybrid     Dense + sparse + reranker (full pipeline)

Options:
  --embedder bge|openai   Which embedding model (for dense/hybrid)
  --no-reranker           Skip cross-encoder reranking (ablation)

Run: python scripts/run_eval.py --mode hybrid --embedder bge
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import load_eval_set, run_eval, print_scorecard

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
DATA_DIR = ROOT / "data"

ALL_CHUNKS = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
ALL_CHUNK_IDS = [c["chunk_id"] for c in ALL_CHUNKS]
CHUNK_LOOKUP = {c["chunk_id"]: c["text"] for c in ALL_CHUNKS}
EVAL_LOOKUP = {e["question"]: e for e in load_eval_set()}


# --- Dummy retrievers (baselines) ---

def random_retriever(question: str, k: int) -> list[str]:
    return random.sample(ALL_CHUNK_IDS, min(k, len(ALL_CHUNK_IDS)))


def perfect_retriever(question: str, k: int) -> list[str]:
    entry = EVAL_LOOKUP.get(question)
    if entry is None:
        return ALL_CHUNK_IDS[:k]
    expected = entry["expected_chunks"]
    padding = [c for c in ALL_CHUNK_IDS if c not in expected]
    return expected + padding[: k - len(expected)]


# --- Real retrievers ---

def build_sparse():
    from retrieval.sparse import SparseRetriever
    sparse = SparseRetriever()
    bm25_path = str(DATA_DIR / "bm25_index.pkl")
    sparse.load(bm25_path)
    return sparse


def build_dense(embedder_type: str):
    if embedder_type == "bge":
        from retrieval.embedder import BGEEmbedder
        embedder = BGEEmbedder()
        qdrant_path = str(DATA_DIR / "qdrant_bge")
    else:
        from retrieval.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        qdrant_path = str(DATA_DIR / "qdrant_openai")

    from retrieval.dense import DenseRetriever
    return DenseRetriever(embedder, qdrant_path=qdrant_path)


def build_hybrid(embedder_type: str, use_reranker: bool):
    dense = build_dense(embedder_type)
    sparse = build_sparse()

    from retrieval.hybrid import HybridRetriever
    return HybridRetriever(
        dense=dense,
        sparse=sparse,
        chunk_lookup=CHUNK_LOOKUP,
        use_reranker=use_reranker,
    )


def main():
    parser = argparse.ArgumentParser(description="Run eval harness")
    parser.add_argument(
        "--mode",
        choices=["random", "perfect", "bm25", "dense", "hybrid"],
        default="random",
    )
    parser.add_argument(
        "--embedder",
        choices=["bge", "openai"],
        default="bge",
        help="Embedding model (for dense/hybrid modes)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip cross-encoder reranking (for ablation)",
    )
    args = parser.parse_args()

    if args.mode == "random":
        retrieve_fn = random_retriever
    elif args.mode == "perfect":
        retrieve_fn = perfect_retriever
    elif args.mode == "bm25":
        sparse = build_sparse()
        retrieve_fn = lambda q, k: [cid for cid, _ in sparse.search(q, k)]
    elif args.mode == "dense":
        dense = build_dense(args.embedder)
        retrieve_fn = lambda q, k: [cid for cid, _ in dense.search(q, k)]
    elif args.mode == "hybrid":
        hybrid = build_hybrid(args.embedder, use_reranker=not args.no_reranker)
        retrieve_fn = hybrid.retrieve

    print(f"Running eval: mode={args.mode}", end="")
    if args.mode in ("dense", "hybrid"):
        print(f", embedder={args.embedder}", end="")
    if args.mode == "hybrid":
        print(f", reranker={'off' if args.no_reranker else 'on'}", end="")
    print("\n")

    results = run_eval(retrieve_fn, k=5)
    print_scorecard(results)


if __name__ == "__main__":
    main()
