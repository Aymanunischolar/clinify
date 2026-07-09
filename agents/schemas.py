"""Structured output schemas shared across agent nodes."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    requires_coding: bool = False


class Citation(BaseModel):
    chunk_id: str
    title: str
    source: str


class ReasonerOutput(BaseModel):
    key_findings: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class WriterOutput(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class QAOutput(BaseModel):
    is_grounded: bool
    faithfulness_notes: str = ""
    revised_answer: str | None = None


class ICD10Suggestion(BaseModel):
    code: str
    description: str
    confidence: float


class CodingOutput(BaseModel):
    suggested_codes: list[ICD10Suggestion] = Field(default_factory=list)
    rationale: str = ""
