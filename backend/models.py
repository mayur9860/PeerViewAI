"""
ORM models for PeerReview AI.

Models: User, Draft, Section, FeatureSnapshot, CritiqueSession.
Enum types stored as String columns for SQLite compatibility.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from database import Base


# ---------------------------------------------------------------------------
# Enums (stored as strings in the DB for SQLite compatibility)
# ---------------------------------------------------------------------------

class SectionType(str, enum.Enum):
    """Allowed section types for an academic paper."""
    INTRO = "intro"
    RELATED_WORK = "related_work"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"


class SessionOutcome(str, enum.Enum):
    """Outcome of a critique session."""
    SAFE = "safe"
    FLAGGED_AND_RETRIED = "flagged_and_retried"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    drafts = relationship("Draft", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class Draft(Base):
    __tablename__ = "drafts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    field = Column(String, nullable=False)  # e.g. "cs", "biology", "economics"
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="drafts")
    sections = relationship("Section", back_populates="draft", cascade="all, delete-orphan")
    feature_snapshots = relationship(
        "FeatureSnapshot", back_populates="draft", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Draft(id={self.id}, title='{self.title}')>"


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    draft_id = Column(Integer, ForeignKey("drafts.id"), nullable=False)
    section_type = Column(String, nullable=False)  # SectionType enum value
    content = Column(Text, nullable=False, default="")
    last_revised_at = Column(DateTime, nullable=True)
    revision_count = Column(Integer, nullable=False, default=0)

    # Relationships
    draft = relationship("Draft", back_populates="sections")
    critique_sessions = relationship(
        "CritiqueSession", back_populates="section", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Section(id={self.id}, type='{self.section_type}')>"


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    draft_id = Column(Integer, ForeignKey("drafts.id"), nullable=False)
    computed_at = Column(DateTime, server_default=func.now(), nullable=False)
    features_json = Column(Text, nullable=False)  # Serialized dict of 15 features
    archetype_id = Column(Integer, nullable=True)  # GMM cluster assignment
    archetype_confidence = Column(Float, nullable=True)

    # Relationships
    draft = relationship("Draft", back_populates="feature_snapshots")

    def __repr__(self):
        return f"<FeatureSnapshot(id={self.id}, archetype={self.archetype_id})>"


class CritiqueSession(Base):
    __tablename__ = "critique_sessions"

    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    transcript_json = Column(Text, nullable=False)  # list of {role, content} turns
    outcome = Column(String, nullable=False)  # SessionOutcome enum value

    # Relationships
    section = relationship("Section", back_populates="critique_sessions")

    def __repr__(self):
        return f"<CritiqueSession(id={self.id}, outcome='{self.outcome}')>"
