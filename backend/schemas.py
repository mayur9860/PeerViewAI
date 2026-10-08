"""
Pydantic v2 request/response models for all API endpoints.

Every route uses typed schemas — no raw dicts in route signatures.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """POST /users — request body."""
    email: str


class UserResponse(BaseModel):
    """User representation in responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Draft schemas
# ---------------------------------------------------------------------------

class DraftCreate(BaseModel):
    """POST /drafts — request body."""
    title: str
    field: str
    user_id: int


class DraftResponse(BaseModel):
    """Draft representation in responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    field: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Section schemas
# ---------------------------------------------------------------------------

class SectionUpsert(BaseModel):
    """POST /drafts/{id}/sections — request body."""
    section_type: str  # Must match SectionType enum
    content: str


class SectionResponse(BaseModel):
    """Section representation in responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    draft_id: int
    section_type: str
    content: str
    last_revised_at: datetime | None
    revision_count: int


# ---------------------------------------------------------------------------
# Analysis / ML schemas
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """POST /drafts/{id}/analyze — response body."""
    draft_id: int
    archetype_id: int
    archetype_label: str
    archetype_confidence: float
    features: dict[str, float]


class SectionRanking(BaseModel):
    """A single section in the recommendation ranking."""
    section_id: int
    section_type: str
    priority_score: float
    rank: int


class RecommendResponse(BaseModel):
    """GET /drafts/{id}/recommend — response body."""
    draft_id: int
    archetype_label: str
    ranked_sections: list[SectionRanking]


# ---------------------------------------------------------------------------
# Search / RAG schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """POST /search — request body."""
    query: str
    field: str | None = None
    section_type: str | None = None
    difficulty: str | None = None


class SearchResult(BaseModel):
    """A single RAG search result."""
    content: str
    source: str
    section_type: str | None = None
    field: str | None = None
    difficulty: str | None = None
    relevance_score: float


class SearchResponse(BaseModel):
    """POST /search — response body."""
    query: str
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Critique / Agent schemas
# ---------------------------------------------------------------------------

class CritiqueRequest(BaseModel):
    """POST /sections/{id}/critique — optional request body for context."""
    pass  # Section text is read from DB; no extra input needed


class TranscriptTurn(BaseModel):
    """A single turn in the critique transcript."""
    role: str  # "critic", "coach", "reviewer", "system"
    content: str


class CritiqueResponse(BaseModel):
    """POST /sections/{id}/critique — response body."""
    session_id: int
    section_id: int
    outcome: str
    transcript: list[TranscriptTurn]
    final_message: str


class CritiqueHistoryResponse(BaseModel):
    """GET /drafts/{id}/sections/{id}/history — response body."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    section_id: int
    created_at: datetime
    outcome: str
    transcript: list[TranscriptTurn]
