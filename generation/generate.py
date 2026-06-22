"""
Generation orchestrator: retrieve → format prompt → call LLM → check citations.

Bridges retrieval (Phase 3) and the LLM. Post-processes the answer to
extract cited rules and verify they exist in the retrieved chunks.

Key concept — citation guardrail:
  After the LLM generates an answer, we extract every rule number it
  cited (e.g., "Rule 3A.2.1(2)") and check if that rule was actually
  in the retrieved chunks. If the LLM cites a rule it wasn't given,
  that's a hallucinated citation — the model made it up or pulled it
  from training data rather than the provided context.
"""

import re

from generation.llm import LLMClient
from generation.prompt import SYSTEM_PROMPT, build_user_prompt

RULE_CITE_RE = re.compile(
    r"(?:Rule\s+)"
    r"((?:A\d+|\d+[A-Z]?)\.\d+\.\d+[A-Z]?"
    r"(?:\(\d+\)(?:\([a-z]\))?)?)",
    re.IGNORECASE,
)


def extract_cited_rules(answer: str) -> list[str]:
    """Extract rule references from generated text.

    Matches patterns like: Rule 3A.2.1, Rule 15.4.5(1), Rule A2.5.1,
    Rule 3A.2.1(2)(a). Returns deduplicated list.
    """
    refs = []
    seen = set()
    for m in RULE_CITE_RE.finditer(answer):
        base = re.match(r"((?:A\d+|\d+[A-Z]?)\.\d+\.\d+[A-Z]?)", m.group(1))
        if base and base.group(1) not in seen:
            seen.add(base.group(1))
            refs.append(base.group(1))
    return refs


def check_citations(
    cited_rules: list[str], retrieved_chunks: list[dict]
) -> dict:
    """Verify cited rules exist in the retrieved chunks."""
    retrieved_rule_ids = {c["rule_id"] for c in retrieved_chunks}
    valid = [r for r in cited_rules if r in retrieved_rule_ids]
    hallucinated = [r for r in cited_rules if r not in retrieved_rule_ids]
    return {
        "all_valid": len(hallucinated) == 0,
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
    }


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    llm: LLMClient,
) -> dict:
    """Generate a grounded answer from retrieved chunks.

    Returns a dict with the answer, extracted citations, and
    citation validity check.
    """
    user_prompt = build_user_prompt(question, retrieved_chunks)
    result = llm.complete_with_usage(SYSTEM_PROMPT, user_prompt)
    answer = result["content"]

    cited_rules = extract_cited_rules(answer)
    citation_check = check_citations(cited_rules, retrieved_chunks)

    return {
        "answer": answer,
        "model": llm.model_name,
        "cited_rules": cited_rules,
        "citation_check": citation_check,
        "chunks_provided": [c["chunk_id"] for c in retrieved_chunks],
        "usage": result["usage"],
    }
