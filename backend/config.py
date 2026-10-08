"""
Configuration module — loads environment variables from .env file
and exposes a typed Settings object via Pydantic BaseSettings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "craft_guidance"

    # Embedding models
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Database
    database_url: str = "sqlite:///./peerreview.db"


# Singleton settings instance — import this everywhere
settings = Settings()
