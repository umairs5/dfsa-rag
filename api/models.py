"""
Pydantic models for the /ask API endpoint.

Pydantic validates incoming requests automatically — if someone sends
a request without a "question" field, FastAPI returns a 400 error with
a clear message, without you writing any validation code.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="The regulatory question to answer")
    use_reranker: bool = Field(
        default=False,
        description="Enable cross-encoder reranking (slower but higher quality)",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class ChunkInfo(BaseModel):
    chunk_id: str
    rule_id: str
    section_title: str
    context_header: str
    text: str


class AskResponse(BaseModel):
    answer: str
    model: str
    cited_rules: list[str]
    citation_check: dict
    retrieved_chunks: list[ChunkInfo]
    trace_id: str = Field(default="", description="Trace ID for linking feedback to this response")


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer that was given")
    rating: int = Field(..., ge=-1, le=1, description="-1 = bad, 0 = neutral, 1 = good")
    corrected_answer: str | None = Field(
        default=None,
        description="If the answer was wrong, what should it have said?",
    )
    expected_chunks: list[str] | None = Field(
        default=None,
        description="Which chunk_ids should have been retrieved? (for expert users)",
    )
    comment: str | None = Field(default=None, description="Free-text feedback")


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
