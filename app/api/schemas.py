"""
Pydantic schemas for the API.
"""
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(..., description="The question to research", min_length=1, max_length=2000)
    max_steps: int | None = Field(None, description="Override max search steps", ge=1, le=10)


class TraceStep(BaseModel):
    step: int
    thought: str
    action: str
    action_query: str
    observation: str
    documents: list[dict]


class SearchResponse(BaseModel):
    question: str
    answer: str
    trace: list[TraceStep]
    total_steps: int
    sub_queries: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
