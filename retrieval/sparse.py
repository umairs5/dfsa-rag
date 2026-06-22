"""
Sparse retrieval via BM25.

BM25 is a term-frequency scoring function: it ranks documents by how
many query terms they contain, weighted by how rare each term is across
the corpus (inverse document frequency). It catches things vector search
misses: exact rule numbers ("3A.2.1"), legal terms of art ("Privacy
Device"), and abbreviations ("MTF", "DIFC").

Key concept — tokenization matters:
  We preserve rule numbers as single tokens (e.g., "3a.2.1" not split
  into "3a", "2", "1"). A query for "Rule 3A.2.1" should match a chunk
  containing that exact identifier.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace/punctuation, preserve rule numbers."""
    return re.findall(r"[\w.]+", text.lower())


class SparseRetriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index over chunk texts."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs by BM25 score."""
        assert self.bm25 is not None, "Call index() first"
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str) -> None:
        """Pickle the index to disk."""
        data = {"bm25": self.bm25, "chunk_ids": self.chunk_ids}
        Path(path).write_bytes(pickle.dumps(data))

    def load(self, path: str) -> None:
        """Load a pickled index."""
        data = pickle.loads(Path(path).read_bytes())
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
