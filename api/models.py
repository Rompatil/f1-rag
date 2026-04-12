"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language F1 query",
        examples=["Who won the 2021 Abu Dhabi Grand Prix?"],
    )


class SourceSnippet(BaseModel):
    chunk_id: str
    content: str
    score: float
    category: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    confidence: float = Field(ge=0.0, le=1.0)
    cached: bool = False
    query: str = ""
    retrieval_count: int = 0
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    cache_stats: dict
    chunk_categories: dict[str, int] = {}


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
