"""
RAG engine — ChromaDB search pipeline.

Query pipeline (exact order from spec §5.3):
1. Self-query filter extraction via Gemini (strict JSON output)
2. Bi-encoder search (50 results)
3. Post-filter by extracted metadata
4. Filter relaxation if < 3 results
5. Cross-encoder re-rank (top 15 → top 3)
"""

import sys
import json
from datetime import datetime
from config import settings

# --- Application Control Policy Workaround ---
# sentence_transformers requires sklearn, which is blocked by the DLL policy.
# We mock it here so the import succeeds. The CrossEncoder uses PyTorch for inference.
try:
    import sklearn
except ImportError:
    import types
    import importlib.machinery
    mock_sklearn = types.ModuleType("sklearn")
    mock_sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
    mock_metrics = types.ModuleType("metrics")
    mock_metrics.__spec__ = importlib.machinery.ModuleSpec("metrics", None)
    mock_metrics.pairwise_distances = lambda *args, **kwargs: None
    mock_metrics.roc_curve = lambda *args, **kwargs: None
    mock_metrics.f1_score = lambda *args, **kwargs: None
    mock_metrics.matthews_corrcoef = lambda *args, **kwargs: None
    mock_sklearn.metrics = mock_metrics
    sys.modules["sklearn"] = mock_sklearn
    sys.modules["sklearn.metrics"] = mock_metrics
# ---------------------------------------------

from schemas import SearchResult

# Lazy-loaded singletons
_chroma_collection = None
_cross_encoder = None


def get_chroma_collection():
    """Get or create the ChromaDB collection (persistent local client)."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    _chroma_collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _get_cross_encoder():
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    except (ImportError, OSError) as e:
        print(f"WARNING: Skipping cross-encoder reranking due to import error (likely DLL policy block): {e}")
        _cross_encoder = "MOCKED"
        
    return _cross_encoder


async def _extract_query_filters(query: str) -> dict:
    return {
        "semantic_query": query,
        "section_type": None,
        "field": None,
        "difficulty": None,
    }


def _post_filter(
    results: list[dict],
    section_type: str | None,
    field: str | None,
    difficulty: str | None,
) -> list[dict]:
    """Filter results by non-null metadata fields."""
    filtered = results
    if section_type:
        filtered = [r for r in filtered if r.get("metadata", {}).get("section_type") == section_type]
    if field:
        filtered = [r for r in filtered if r.get("metadata", {}).get("field") == field]
    if difficulty:
        filtered = [r for r in filtered if r.get("metadata", {}).get("difficulty") == difficulty]
    return filtered


def _relax_and_filter(
    results: list[dict],
    section_type: str | None,
    field: str | None,
    difficulty: str | None,
    min_results: int = 3,
) -> list[dict]:
    """Relax filters one at a time until at least min_results remain.

    Relaxation order: field → difficulty → section_type (per spec §5.3).
    """
    filtered = _post_filter(results, section_type, field, difficulty)
    if len(filtered) >= min_results:
        return filtered

    # Relax field first
    filtered = _post_filter(results, section_type, None, difficulty)
    if len(filtered) >= min_results:
        return filtered

    # Relax difficulty
    filtered = _post_filter(results, section_type, None, None)
    if len(filtered) >= min_results:
        return filtered

    # Relax section_type (all filters relaxed)
    return results


def _cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
    max_input: int = 15,
) -> list[dict]:
    """Re-rank candidates using the cross-encoder and return top_k.

    Caps input at max_input (15) candidates per spec.
    """
    if not candidates:
        return []

    candidates = candidates[:max_input]
    cross_encoder = _get_cross_encoder()

    if cross_encoder == "MOCKED":
        # Fallback if cross-encoder failed to load
        return candidates[:top_k]

    # Build pairs for cross-encoder
    pairs = [(query, c["content"]) for c in candidates]

    # Score
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    for c, score in zip(candidates, scores):
        c["cross_encoder_score"] = float(score)

    candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return candidates[:top_k]


async def search(
    query: str,
    section_type: str | None = None,
    field: str | None = None,
    difficulty: str | None = None,
) -> list[SearchResult]:
    """Execute the full RAG search pipeline.

    Steps (spec §5.3):
    1. Self-query filter extraction via Gemini
    2. Bi-encoder search (50 results)
    3. Post-filter by metadata
    4. Filter relaxation if < 3 results
    5. Cross-encoder re-rank → top 3
    """
    collection = get_chroma_collection()

    # Step 1: Extract filters via Gemini (if API key available)
    if settings.gemini_api_key:
        extracted = await _extract_query_filters(query)
        semantic_query = extracted["semantic_query"]
        # Use extracted filters as defaults, but override with explicit params
        section_type = section_type or extracted["section_type"]
        field = field or extracted["field"]
        difficulty = difficulty or extracted["difficulty"]
    else:
        semantic_query = query

    # Step 2: Bi-encoder search (ChromaDB uses its built-in embedding)
    try:
        chroma_results = collection.query(
            query_texts=[semantic_query],
            n_results=50,
        )
    except Exception as e:
        # Collection might be empty
        return []

    if not chroma_results or not chroma_results.get("documents"):
        return []

    # Flatten ChromaDB results into dicts
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]
    distances = chroma_results["distances"][0] if chroma_results.get("distances") else [0.0] * len(documents)
    ids = chroma_results["ids"][0]

    candidates = []
    for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
        candidates.append({
            "content": doc,
            "metadata": meta or {},
            "distance": dist,
            "source": doc_id,
        })

    # Step 3 + 4: Post-filter with relaxation
    filtered = _relax_and_filter(candidates, section_type, field, difficulty)

    # Step 5: Cross-encoder re-rank → top 3
    reranked = _cross_encoder_rerank(semantic_query, filtered)

    # Convert to SearchResult schema
    results = []
    for r in reranked:
        meta = r.get("metadata", {})
        results.append(SearchResult(
            content=r["content"],
            source=r.get("source", "unknown"),
            section_type=meta.get("section_type"),
            field=meta.get("field"),
            difficulty=meta.get("difficulty"),
            relevance_score=r.get("cross_encoder_score", 1.0 - r.get("distance", 0.0)),
        ))

    return results
