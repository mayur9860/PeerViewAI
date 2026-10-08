"""
Database module — SQLAlchemy 2.x engine, session factory, and FastAPI dependency.

Uses SQLite for local dev; schema avoids SQLite-specific types so Postgres
swap is trivial (just change DATABASE_URL).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings


# For SQLite, we need check_same_thread=False to allow FastAPI's async usage
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def create_all_tables():
    """Create all tables defined by ORM models. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
