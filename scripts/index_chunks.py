"""
Build retrieval indices from chunks.jsonl.

Embeds all 52 chunks, stores in Qdrant (on-disk) and builds a BM25 index.
Run once per embedder, then use the retrieval pipeline to search.

Usage:
    python scripts/index_chunks.py --embedder bge
    python scripts/index_chunks.py --embedder openai
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.jsonl"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices")
    parser.add_argument(
        "--embedder",
        choices=["bge", "openai"],
        default="bge",
        help="Which embedding model to use",
    )
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    # --- Dense index (Qdrant) ---
    print(f"Building dense index with {args.embedder} embedder...")
    if args.embedder == "bge":
        from retrieval.embedder import BGEEmbedder
        embedder = BGEEmbedder()
        qdrant_path = str(DATA_DIR / "qdrant_bge")
    else:
        from retrieval.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        qdrant_path = str(DATA_DIR / "qdrant_openai")

    from retrieval.dense import DenseRetriever
    dense = DenseRetriever(embedder, qdrant_path=qdrant_path)

    t0 = time.time()
    dense.index(chunks)
    dt = time.time() - t0
    print(f"  Embedded and indexed {len(chunks)} chunks in {dt:.1f}s")
    print(f"  Stored at: {qdrant_path}/")

    # --- Sparse index (BM25) ---
    print("\nBuilding BM25 index...")
    from retrieval.sparse import SparseRetriever
    sparse = SparseRetriever()

    t0 = time.time()
    sparse.index(chunks)
    bm25_path = str(DATA_DIR / "bm25_index.pkl")
    sparse.save(bm25_path)
    dt = time.time() - t0
    print(f"  Built BM25 index in {dt:.2f}s")
    print(f"  Stored at: {bm25_path}")

    print("\nDone. Run `python scripts/run_eval.py --mode <mode>` to evaluate.")


if __name__ == "__main__":
    main()
