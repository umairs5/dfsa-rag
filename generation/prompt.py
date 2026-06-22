"""
System prompt and chunk formatting for grounded generation.

The system prompt is the primary hallucination guardrail: it instructs
the LLM to answer ONLY from the provided context and cite rule numbers.
This is more effective than post-hoc hallucination detection because it
prevents the problem rather than catching it after the fact.

Key concept — grounding:
  An LLM "knows" many things from training data, but that knowledge may
  be outdated, wrong, or about a different jurisdiction. Grounding means
  forcing the model to use only the specific text you provide. The system
  prompt enforces this; the chunk format makes it easy for the model to
  cite specific rules.
"""

SYSTEM_PROMPT = """\
You are a DFSA (Dubai Financial Services Authority) regulatory compliance \
assistant. Your role is to answer questions about DFSA crypto-token \
regulations based ONLY on the provided regulatory text.

Rules:
1. Answer ONLY using information from the provided context sections below.
2. Cite specific rule numbers when stating requirements (e.g., "Under Rule \
3A.2.1(2)", "Rule 15.4.5 requires...").
3. If the provided context does not contain enough information to answer the \
question, say: "Based on the provided regulatory text, I don't have enough \
information to answer this question."
4. Do not add information from outside the provided context, even if you \
believe it to be correct.
5. Be precise and concise. Compliance officers need specific, actionable \
answers, not general summaries."""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks.

    Includes rule_id and context_header so the LLM knows what to cite
    and where each rule sits in the regulatory structure.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        rule_id = chunk.get("rule_id", "unknown")
        header = chunk.get("context_header", "")
        text = chunk["text"]
        blocks.append(f"[Source {i}: Rule {rule_id} | {header}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Combine formatted context + question into the user message."""
    context = format_chunks_for_prompt(chunks)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"---\n\n"
        f"QUESTION: {question}"
    )
