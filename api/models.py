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
