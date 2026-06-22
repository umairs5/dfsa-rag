"""
Retrieval scoring metrics for the DFSA RAG eval harness.

All functions are pure: they take retrieved and expected chunk lists,
return a float. No I/O, no side effects.

Metrics explained (for someone new to information retrieval):

- Recall@k: "Did we find the right stuff?" Of the chunks that *should*
  have been retrieved, what fraction actually appeared in the top-k?
  Recall@5 = 1.0 means every relevant chunk was in the top 5.

- Precision@k: "How much noise did we pull in?" Of the k chunks we
  retrieved, what fraction were actually relevant? Low precision means
  the LLM gets a lot of irrelevant context.

- Reciprocal Rank: "How high up is the first relevant result?" If the
  right chunk is at position 1, RR=1.0. Position 3, RR=0.33. The mean
  across all questions (MRR) tells you how well the system ranks.
"""


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected chunks that appear in the top-k retrieved."""
    if not expected:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for chunk_id in expected if chunk_id in top_k)
    return hits / len(expected)


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are actually relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in set(expected))
    return hits / k


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / (rank of first relevant chunk). Returns 0.0 if none found."""
    expected_set = set(expected)
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def score_single(
    retrieved: list[str],
    expected: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all retrieval metrics for one question."""
    if k_values is None:
        k_values = [1, 3, 5]
    result = {}
    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(retrieved, expected, k)
        result[f"precision@{k}"] = precision_at_k(retrieved, expected, k)
    result["reciprocal_rank"] = reciprocal_rank(retrieved, expected)
    return result
