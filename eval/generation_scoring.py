"""
Generation quality metrics for the DFSA RAG eval harness.

Complements eval/scoring.py (retrieval metrics) with generation-specific
metrics. Kept in a separate file because generation metrics need text
and possibly LLM access, unlike retrieval metrics which are pure
chunk_id list operations.

Metrics:
  - citation_precision: what fraction of cited rules were in the chunks?
  - citation_recall: what fraction of expected chunks were cited?
  - answer_similarity: semantic similarity to expected answer (via BGE)
  - faithfulness_heuristic: do key terms in the answer appear in chunks?
"""

import re


def citation_precision(cited_rules: list[str], retrieved_chunks: list[dict]) -> float:
    """What fraction of cited rules were actually in the retrieved chunks?

    1.0 = every citation is valid (no hallucinated citations).
    0.0 = every citation is made up.
    """
    if not cited_rules:
        return 1.0  # no citations = nothing to hallucinate
    retrieved_ids = {c["rule_id"] for c in retrieved_chunks}
    valid = sum(1 for r in cited_rules if r in retrieved_ids)
    return valid / len(cited_rules)


def citation_recall(cited_rules: list[str], expected_chunks: list[str]) -> float:
    """What fraction of expected chunks were cited in the answer?

    Checks if the rule_id portion of expected_chunk_ids appears in citations.
    E.g., expected "GEN_3A.2.1" → checks for "3A.2.1" in cited_rules.
    """
    if not expected_chunks:
        return 1.0
    cited_set = set(cited_rules)
    hits = 0
    for chunk_id in expected_chunks:
        # Extract rule_id from chunk_id (e.g., "GEN_3A.2.1" → "3A.2.1")
        parts = chunk_id.split("_", 1)
        rule_id = parts[1] if len(parts) > 1 else parts[0]
        if rule_id in cited_set:
            hits += 1
    return hits / len(expected_chunks)


def answer_similarity_bge(
    generated: str, expected: str, embedder
) -> float:
    """Cosine similarity between generated and expected answer using BGE.

    Reuses the embedder already loaded for retrieval — no extra dependency.
    Range: -1.0 to 1.0 (normalized embeddings, so typically 0.0 to 1.0).
    """
    gen_vec = embedder.embed_query(generated)
    exp_vec = embedder.embed_query(expected)
    dot = sum(a * b for a, b in zip(gen_vec, exp_vec))
    return dot  # already normalized by BGE


def faithfulness_heuristic(answer: str, chunks: list[dict]) -> float:
    """Simple check: do key noun phrases in the answer appear in the chunks?

    Extracts capitalized multi-word terms from the answer (likely legal
    terms of art like "Privacy Token", "Authorised Firm") and checks
    if they appear in the chunk text. Returns fraction found.

    Not perfect — a model could rephrase chunk content using different
    terms and score low. But it's free, fast, and catches obvious
    hallucinations (made-up terms, invented requirements).
    """
    chunk_text = " ".join(c["text"] for c in chunks).lower()

    # Extract capitalized multi-word phrases (likely legal terms)
    terms = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", answer)
    if not terms:
        return 1.0  # no key terms to check

    found = sum(1 for t in terms if t.lower() in chunk_text)
    return found / len(terms)
