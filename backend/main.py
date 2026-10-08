"""
FastAPI application — all route registration, static file serving.

Routes follow spec §7. CRUD routes are fully functional from the start;
ML/RAG/Agent routes are wired to their respective engines.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from config import settings
from database import create_all_tables, get_db
from models import (
    CritiqueSession,
    Draft,
    FeatureSnapshot,
    Section,
    SectionType,
    SessionOutcome,
    User,
)
from schemas import (
    AnalyzeResponse,
    CritiqueHistoryResponse,
    CritiqueResponse,
    DraftCreate,
    DraftResponse,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
    SectionRanking,
    SectionResponse,
    SectionUpsert,
    TranscriptTurn,
    UserCreate,
    UserResponse,
)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup."""
    create_all_tables()
    yield


app = FastAPI(
    title="PeerReview AI",
    description="Academic writing coach with Socratic critique agent",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Static files — serve frontend from /static, index at /
# ---------------------------------------------------------------------------

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the frontend SPA."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "PeerReview AI API is running. Frontend not found at /static/index.html"}


# ---------------------------------------------------------------------------
# User routes
# ---------------------------------------------------------------------------

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    # Check for duplicate email
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email '{payload.email}' already exists",
        )
    user = User(email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Draft routes
# ---------------------------------------------------------------------------

@app.post("/drafts", response_model=DraftResponse, status_code=status.HTTP_201_CREATED)
def create_draft(payload: DraftCreate, db: Session = Depends(get_db)):
    """Create a new draft."""
    # Verify user exists
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {payload.user_id} not found",
        )
    draft = Draft(title=payload.title, field=payload.field, user_id=payload.user_id)
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


# ---------------------------------------------------------------------------
# Section routes
# ---------------------------------------------------------------------------

@app.post(
    "/drafts/{draft_id}/sections",
    response_model=SectionResponse,
    status_code=status.HTTP_200_OK,
)
def upsert_section(
    draft_id: int,
    payload: SectionUpsert,
    db: Session = Depends(get_db),
):
    """Upsert a section for a draft — creates or updates by section_type."""
    # Validate draft exists
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft with id {draft_id} not found",
        )

    # Validate section_type
    try:
        SectionType(payload.section_type)
    except ValueError:
        valid = [st.value for st in SectionType]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid section_type '{payload.section_type}'. Must be one of: {valid}",
        )

    # Upsert: find existing section of this type for this draft
    section = (
        db.query(Section)
        .filter(Section.draft_id == draft_id, Section.section_type == payload.section_type)
        .first()
    )

    if section:
        section.content = payload.content
        section.last_revised_at = datetime.utcnow()
        section.revision_count += 1
    else:
        section = Section(
            draft_id=draft_id,
            section_type=payload.section_type,
            content=payload.content,
            last_revised_at=datetime.utcnow(),
            revision_count=0,
        )
        db.add(section)

    db.commit()
    db.refresh(section)
    return section


# ---------------------------------------------------------------------------
# Analysis route — ML engine
# ---------------------------------------------------------------------------

@app.post("/drafts/{draft_id}/analyze", response_model=AnalyzeResponse)
def analyze_draft(draft_id: int, db: Session = Depends(get_db)):
    """Run feature engineering + GMM classification on a draft."""
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft with id {draft_id} not found",
        )

    sections = db.query(Section).filter(Section.draft_id == draft_id).all()
    if not sections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Draft has no sections. Add at least one section before analyzing.",
        )

    # Import here to avoid circular imports and allow graceful failure
    try:
        from ml_engine import build_feature_vector, classify_archetype
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML engine not available: {e}",
        )

    # Build feature vector
    features = build_feature_vector(draft, sections, db)

    # Classify archetype
    archetype_id, archetype_label, confidence = classify_archetype(features)

    # Persist snapshot
    snapshot = FeatureSnapshot(
        draft_id=draft_id,
        features_json=json.dumps(features),
        archetype_id=archetype_id,
        archetype_confidence=confidence,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    return AnalyzeResponse(
        draft_id=draft_id,
        archetype_id=archetype_id,
        archetype_label=archetype_label,
        archetype_confidence=confidence,
        features=features,
    )


# ---------------------------------------------------------------------------
# Recommendation route — XGBoost ranker
# ---------------------------------------------------------------------------

@app.get("/drafts/{draft_id}/recommend", response_model=RecommendResponse)
def recommend_sections(draft_id: int, db: Session = Depends(get_db)):
    """Run XGBoost ranker over sections to produce a revision priority ranking."""
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft with id {draft_id} not found",
        )

    sections = db.query(Section).filter(Section.draft_id == draft_id).all()
    if not sections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Draft has no sections.",
        )

    # Get latest archetype from feature snapshots
    latest_snapshot = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.draft_id == draft_id)
        .order_by(FeatureSnapshot.computed_at.desc())
        .first()
    )

    try:
        from ml_engine import rank_sections
        from persona_prompts import CLUSTER_LABELS
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML engine not available: {e}",
        )

    archetype_id = latest_snapshot.archetype_id if latest_snapshot else 0
    archetype_label = CLUSTER_LABELS.get(archetype_id, "Unknown Archetype")

    ranked = rank_sections(draft, sections, archetype_id, db)

    ranked_sections = [
        SectionRanking(
            section_id=r["section_id"],
            section_type=r["section_type"],
            priority_score=r["priority_score"],
            rank=i + 1,
        )
        for i, r in enumerate(ranked)
    ]

    return RecommendResponse(
        draft_id=draft_id,
        archetype_label=archetype_label,
        ranked_sections=ranked_sections,
    )


# ---------------------------------------------------------------------------
# Search route — RAG engine
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search_guidance(payload: SearchRequest):
    """Search craft guidance corpus using RAG pipeline."""
    try:
        from rag_engine import search
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG engine not available: {e}",
        )

    results = await search(
        query=payload.query,
        section_type=payload.section_type,
        field=payload.field,
        difficulty=payload.difficulty,
    )

    return SearchResponse(query=payload.query, results=results)


# ---------------------------------------------------------------------------
# Critique route — LangGraph agent
# ---------------------------------------------------------------------------

@app.post("/sections/{section_id}/critique", response_model=CritiqueResponse)
async def critique_section(section_id: int, db: Session = Depends(get_db)):
    """Run the LangGraph Socratic critique agent on a section."""
    section = db.query(Section).filter(Section.id == section_id).first()
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section with id {section_id} not found",
        )

    draft = db.query(Draft).filter(Draft.id == section.draft_id).first()

    # Get archetype label
    latest_snapshot = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.draft_id == draft.id)
        .order_by(FeatureSnapshot.computed_at.desc())
        .first()
    )

    try:
        from agent_graph import run_critique_agent
        from persona_prompts import CLUSTER_LABELS
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent not available: {e}",
        )

    archetype_id = latest_snapshot.archetype_id if latest_snapshot else 0
    archetype_label = CLUSTER_LABELS.get(archetype_id, "Unknown Archetype")

    # Run the agent
    result = await run_critique_agent(
        section_text=section.content,
        section_type=section.section_type,
        archetype_label=archetype_label,
    )

    # Persist critique session
    transcript_turns = [
        {"role": t["role"], "content": t["content"]}
        for t in result["transcript"]
    ]

    session = CritiqueSession(
        section_id=section_id,
        transcript_json=json.dumps(transcript_turns),
        outcome=result["outcome"],
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return CritiqueResponse(
        session_id=session.id,
        section_id=section_id,
        outcome=result["outcome"],
        transcript=[TranscriptTurn(**t) for t in transcript_turns],
        final_message=result["final_message"],
    )


# ---------------------------------------------------------------------------
# Critique history route
# ---------------------------------------------------------------------------

@app.get("/drafts/{draft_id}/sections/{section_id}/history")
def get_critique_history(
    draft_id: int,
    section_id: int,
    db: Session = Depends(get_db),
):
    """Get critique session history for a section."""
    # Verify draft & section exist and belong together
    section = db.query(Section).filter(
        Section.id == section_id,
        Section.draft_id == draft_id,
    ).first()
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section {section_id} not found in draft {draft_id}",
        )

    sessions = (
        db.query(CritiqueSession)
        .filter(CritiqueSession.section_id == section_id)
        .order_by(CritiqueSession.created_at.desc())
        .all()
    )

    result = []
    for s in sessions:
        transcript_data = json.loads(s.transcript_json)
        result.append({
            "id": s.id,
            "section_id": s.section_id,
            "created_at": s.created_at.isoformat(),
            "outcome": s.outcome,
            "transcript": [TranscriptTurn(**t) for t in transcript_data],
        })

    return result
