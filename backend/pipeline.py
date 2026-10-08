"""
Corpus ingestion pipeline — reads craft guidance .md files,
parses YAML frontmatter, chunks at paragraph level, embeds
with bi-encoder, and upserts into ChromaDB.

CLI: python pipeline.py --ingest
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

from config import settings


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "craft_corpus"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (metadata_dict, body_text).
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                metadata = {}
            body = parts[2].strip()
            return metadata, body
    return {}, text.strip()


def chunk_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-level chunks.

    Per spec: no further splitting — these files are already short (200–500 words).
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


def ingest_corpus():
    """Ingest all .md files from data/craft_corpus into ChromaDB.

    Idempotent: delete-then-insert by file path as doc ID.
    """
    import chromadb

    if not DATA_DIR.exists():
        print(f"ERROR: Corpus directory not found: {DATA_DIR}")
        sys.exit(1)

    md_files = list(DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"WARNING: No .md files found in {DATA_DIR}")
        return

    print(f"Found {len(md_files)} corpus files in {DATA_DIR}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for md_file in md_files:
        print(f"  Processing: {md_file.name}")

        text = md_file.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(text)

        chunks = chunk_paragraphs(body)
        if not chunks:
            print(f"    Skipping (no valid chunks)")
            continue

        # Build doc IDs and metadata for each chunk
        file_key = md_file.stem  # Use filename (without extension) as base ID

        # Delete existing entries for this file (idempotent re-ingestion)
        existing_ids = [f"{file_key}_chunk_{i}" for i in range(100)]  # generous upper bound
        try:
            collection.delete(ids=existing_ids)
        except Exception:
            pass  # OK if IDs don't exist

        # Prepare batch
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{file_key}_chunk_{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "section_type": metadata.get("section_type", "general"),
                "field": metadata.get("field", "general"),
                "difficulty": metadata.get("difficulty", "intermediate"),
                "source_file": md_file.name,
            })

        # Upsert into ChromaDB (uses ChromaDB's default embedding function)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"    Added {len(chunks)} chunks")

    print(f"\nIngestion complete: {total_chunks} total chunks from {len(md_files)} files")
    print(f"ChromaDB collection '{settings.chroma_collection_name}' ready at {settings.chroma_persist_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeerReview AI corpus ingestion pipeline")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest craft corpus into ChromaDB",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_corpus()
    else:
        parser.print_help()
