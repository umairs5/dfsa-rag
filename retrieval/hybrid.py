"""
Hybrid retrieval: dense + sparse fusion with optional cross-encoder reranking.

Pipeline:
  1. Get top-N candidates from dense retriever (vector search)
  2. Get top-N candidates from sparse retriever (BM25)
  3. Merge via Reciprocal Rank Fusion (RRF)
  4. Optionally rerank with a cross-encoder
  5. Return top-k chunk_ids

Key concepts:

  RRF (Reciprocal Rank Fusion):
    Merges ranked lists using only rank positions, not raw scores.
    For each item at rank r: rrf_score += 1/(k+r) where k=60.
    This avoids the BM25-vs-cosine scale mismatch problem.
    Industry standard (used by Elasticsearch, Pinecone, Weaviate).

  Cross-encoder reranking:
    Unlike bi-encoders (embed query and doc separately), a cross-encoder
    sees both together. More accurate but needs one forward pass per
    candidate — only practical on a short candidate list (~30 items),
    not the full corpus (~52 here, but would be thousands in production).
"""

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF.

    Each input: list of (chunk_id, score), ordered by score descending.
    Returns: list of (chunk_id, rrf_score), ordered by rrf_score descending.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        chunk_lookup: dict[str, str],
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        use_reranker: bool = True,
    ):
        """
        Args:
            dense: vector retriever (Qdrant)
            sparse: BM25 retriever
            chunk_lookup: chunk_id -> chunk text (needed for reranker)
            reranker_model: cross-encoder model name
            use_reranker: set False to skip reranking (for ablation)
        """
        self.dense = dense
        self.sparse = sparse
        self.chunk_lookup = chunk_lookup
        self.reranker = None

        if use_reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model)

    def retrieve(
        self,
        question: str,
        k: int = 5,
        candidates_per_retriever: int = 20,
    ) -> list[str]:
        """Hybrid retrieval conforming to eval harness signature.

        Returns a list of chunk_ids, ranked by relevance.
        """
        dense_results = self.dense.search(question, k=candidates_per_retriever)
        sparse_results = self.sparse.search(question, k=candidates_per_retriever)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        candidate_ids = [cid for cid, _ in fused]

        if self.reranker is not None:
            candidate_ids = self._rerank(question, candidate_ids)

        return candidate_ids[:k]

    def _rerank(self, question: str, candidate_ids: list[str]) -> list[str]:
        """Rerank candidates using the cross-encoder."""
        pairs = [
            (question, self.chunk_lookup.get(cid, ""))
            for cid in candidate_ids
        ]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [cid for cid, _ in ranked]
