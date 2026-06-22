"""
Populate a Qdrant server with chunk embeddings.

Run this once after starting the Qdrant container but before
the API handles requests. Runs from the host machine (not inside
Docker) — it just needs network access to Qdrant.

Usage:
    docker compose up qdrant -d
    python scripts/index_qdrant_server.py
    docker compose up api --build
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Index chunks into Qdrant server")
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant server URL",
    )
    args = parser.parse_args()

    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks")

    print("Loading BGE embedder...")
    from retrieval.embedder import BGEEmbedder
    embedder = BGEEmbedder()

    print(f"Connecting to Qdrant at {args.url}...")
    from retrieval.dense import DenseRetriever
    dense = DenseRetriever(embedder, qdrant_url=args.url)

    print("Embedding and indexing...")
    t0 = time.time()
    dense.index(chunks)
    dt = time.time() - t0
    print(f"Indexed {len(chunks)} chunks in {dt:.1f}s")

    # Also build BM25 index (for the Docker image)
    print("Building BM25 index...")
    from retrieval.sparse import SparseRetriever
    sparse = SparseRetriever()
    sparse.index(chunks)
    bm25_path = str(Path(__file__).resolve().parent.parent / "data" / "bm25_index.pkl")
    sparse.save(bm25_path)
    print(f"BM25 index saved to {bm25_path}")

    print("\nDone. You can now start the API: docker compose up api --build")


if __name__ == "__main__":
    main()
