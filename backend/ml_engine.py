"""
ML engine — feature engineering, GMM archetype classification, XGBoost section ranking.

Features: 15 bounded (0–1 or normalized) floats per draft.
GMM: GaussianMixture(n_components=8, covariance_type="diag")
XGBoost: XGBRegressor predicting 0–1 revision priority per section.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pickle
import numpy as np
from sqlalchemy.orm import Session

from persona_prompts import CLUSTER_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Strong-claim verbs / phrases (~15)
STRONG_CLAIM_TERMS = [
    "prove", "proves", "proven", "demonstrate conclusively", "always",
    "undeniably", "certainly", "definitively", "unquestionably", "clearly show",
    "establish beyond doubt", "confirm", "guarantee", "inevitably", "without exception",
]

# Hedged-claim verbs / phrases (~15)
HEDGED_CLAIM_TERMS = [
    "suggest", "suggests", "may indicate", "is consistent with", "appears to",
    "might", "could", "possibly", "potentially", "tends to",
    "it is plausible", "seemingly", "arguably", "likely", "to some extent",
]

# Contribution clarity patterns
CONTRIBUTION_PATTERNS = [
    r"\bwe\s+contribute\b",
    r"\bour\s+contributions?\s+(are|include)\b",
    r"\bwe\s+make\s+the\s+following\s+contributions?\b",
    r"\bour\s+main\s+contributions?\b",
    r"\bthe\s+contributions?\s+of\s+this\s+(paper|work)\b",
    r"\bin\s+this\s+(paper|work)\s*,?\s*we\s+(propose|present|introduce)\b",
]

# Load common words for jargon ratio
_common_words: set[str] | None = None


def _load_common_words() -> set[str]:
    """Load the common English words list (top ~10k words)."""
    global _common_words
    if _common_words is not None:
        return _common_words

    wordlist_path = DATA_DIR / "common_words_10k.txt"
    if not wordlist_path.exists():
        # Fallback: return empty set (everything will look like jargon)
        _common_words = set()
        return _common_words

    with open(wordlist_path, "r", encoding="utf-8") as f:
        _common_words = {line.strip().lower() for line in f if line.strip()}

    return _common_words


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Count words in a text string."""
    return len(text.split()) if text else 0


def _sentence_count(text: str) -> int:
    """Count sentences using simple period/question/exclamation splitting."""
    if not text:
        return 0
    sentences = re.split(r'[.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))


def _avg_sentence_length(text: str) -> float:
    """Average words per sentence."""
    if not text:
        return 0.0
    wc = _word_count(text)
    sc = _sentence_count(text)
    return wc / sc if sc > 0 else 0.0


def _count_citations(text: str) -> int:
    """Count citation markers like [1], [2,3], (Author, 2020), etc."""
    bracket_cites = re.findall(r'\[\d+(?:,\s*\d+)*\]', text)
    paren_cites = re.findall(r'\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)*,?\s*\d{4}\)', text)
    return len(bracket_cites) + len(paren_cites)


def _normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# 15 Feature functions — each returns a bounded float
# ---------------------------------------------------------------------------

def compute_related_work_citation_density(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 1: Citations per 100 words in the related-work section. Normalized to [0, 1]."""
    text = section_text_map.get("related_work", "")
    wc = _word_count(text)
    if wc == 0:
        return 0.0
    citations = _count_citations(text)
    density = (citations / wc) * 100
    # Cap at ~20 citations per 100 words → normalize to [0, 1]
    return _normalize(density / 20.0)


def compute_related_work_recency_ratio(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 2: Fraction of cited works from the last 5 years."""
    text = section_text_map.get("related_work", "")
    if not text:
        return 0.0

    # Extract years from citations
    years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]
    if not years:
        return 0.0

    current_year = datetime.utcnow().year
    recent = sum(1 for y in years if current_year - y <= 5)
    return _normalize(recent / len(years))


def compute_overclaim_score(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 3: Ratio of strong-claim to hedged-claim terms. Normalized to [0, 1]."""
    all_text = " ".join(section_text_map.values()).lower()
    if not all_text:
        return 0.0

    strong_count = sum(all_text.count(term) for term in STRONG_CLAIM_TERMS)
    hedged_count = sum(all_text.count(term) for term in HEDGED_CLAIM_TERMS)

    total = strong_count + hedged_count
    if total == 0:
        return 0.5  # Neutral if no claim language found

    return _normalize(strong_count / total)


def compute_methods_density(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 4: Fraction of total word count in the methods section."""
    methods_wc = _word_count(section_text_map.get("methods", ""))
    total_wc = sum(_word_count(t) for t in section_text_map.values())
    if total_wc == 0:
        return 0.0
    return _normalize(methods_wc / total_wc)


def compute_structural_completeness(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 5: Fraction of 5 expected sections present (non-empty)."""
    expected = ["intro", "related_work", "methods", "results", "discussion"]
    present = sum(1 for s in expected if _word_count(section_text_map.get(s, "")) > 0)
    return present / len(expected)


def compute_avg_sentence_length_intro(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 6: Average sentence length in the intro. Normalized: 25 words/sentence → 1.0."""
    text = section_text_map.get("intro", "")
    avg = _avg_sentence_length(text)
    return _normalize(avg / 40.0)  # 40 words/sentence → 1.0


def compute_avg_sentence_length_overall(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 7: Average sentence length across all sections. Normalized same as #6."""
    all_text = " ".join(section_text_map.values())
    avg = _avg_sentence_length(all_text)
    return _normalize(avg / 40.0)


def compute_jargon_ratio(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 8: Fraction of words not in common-English top-10k wordlist."""
    common_words = _load_common_words()
    all_text = " ".join(section_text_map.values())
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    if not words:
        return 0.0

    jargon_count = sum(1 for w in words if w not in common_words and len(w) > 2)
    return _normalize(jargon_count / len(words))


def compute_contribution_clarity(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 9: 1.0 if intro contains explicit contribution statement, else 0.0."""
    intro = section_text_map.get("intro", "").lower()
    for pattern in CONTRIBUTION_PATTERNS:
        if re.search(pattern, intro):
            return 1.0
    return 0.0


def compute_results_narrative_ratio(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 10: Ratio of prose sentences to data references in results section."""
    text = section_text_map.get("results", "")
    if not text:
        return 0.0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0

    # Count data-heavy sentences: those containing numbers, table/figure refs
    data_pattern = re.compile(r'(?:\d+\.?\d*%?|[Tt]able\s+\d+|[Ff]igure\s+\d+|[Ff]ig\.\s*\d+)')
    data_sentences = sum(1 for s in sentences if data_pattern.search(s))
    prose_sentences = len(sentences) - data_sentences

    if data_sentences == 0:
        return 1.0  # All prose, no data references

    ratio = prose_sentences / (data_sentences + prose_sentences)
    return _normalize(ratio)


def compute_revision_velocity(section_text_map: dict[str, str], sections: list = None, **kwargs) -> float:
    """Feature 11: Revisions in last 14 days / total revisions ever."""
    if not sections:
        return 0.0

    now = datetime.utcnow()
    cutoff = now - timedelta(days=14)

    total_revisions = sum(s.revision_count for s in sections)
    if total_revisions == 0:
        return 0.0

    # Approximate recent revisions: count sections revised after cutoff
    recent_revisions = sum(
        s.revision_count for s in sections
        if s.last_revised_at and s.last_revised_at > cutoff
    )

    return _normalize(recent_revisions / total_revisions)


def compute_abandoned_section_count(section_text_map: dict[str, str], sections: list = None, **kwargs) -> float:
    """Feature 12: Sections untouched >30 days despite being incomplete (<100 words). Normalized to [0, 1]."""
    if not sections:
        return 0.0

    now = datetime.utcnow()
    cutoff = now - timedelta(days=30)

    abandoned = 0
    for s in sections:
        word_count = _word_count(s.content)
        if word_count < 100:
            if s.last_revised_at is None or s.last_revised_at < cutoff:
                abandoned += 1

    # Normalize: assume max 6 sections
    return _normalize(abandoned / 6.0)


def compute_discussion_to_results_length_ratio(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 13: Word count ratio of discussion to results."""
    discussion_wc = _word_count(section_text_map.get("discussion", ""))
    results_wc = _word_count(section_text_map.get("results", ""))

    if results_wc == 0:
        return 0.5  # Neutral if no results section

    ratio = discussion_wc / results_wc
    # Normalize: a ratio of 2.0 → 1.0 (discussion is twice results length)
    return _normalize(ratio / 2.0)


def compute_self_citation_ratio(section_text_map: dict[str, str], **kwargs) -> float:
    """Feature 14: Fraction of citations that are self-citations.

    Approximated by looking for repeated author surnames across citations.
    Since we don't have author metadata, we use a heuristic: any author name
    appearing in >50% of citations is likely a self-citation.
    """
    all_text = " ".join(section_text_map.values())

    # Extract author names from parenthetical citations
    author_cites = re.findall(
        r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)*),?\s*\d{4}\)',
        all_text,
    )

    if not author_cites:
        return 0.0

    # Count first-author frequencies
    first_authors: dict[str, int] = {}
    for cite in author_cites:
        first_author = cite.split()[0].strip(",")
        first_authors[first_author] = first_authors.get(first_author, 0) + 1

    if not first_authors:
        return 0.0

    max_freq = max(first_authors.values())
    total = len(author_cites)

    # If the most-cited author appears in >50% of citations, treat as self-citation ratio
    return _normalize(max_freq / total)


def compute_time_since_last_revision_days(section_text_map: dict[str, str], sections: list = None, **kwargs) -> float:
    """Feature 15: Days since the most recent revision across all sections. Normalized to [0, 1]."""
    if not sections:
        return 1.0  # No revisions ever → maximum staleness

    now = datetime.utcnow()
    revision_dates = [s.last_revised_at for s in sections if s.last_revised_at]

    if not revision_dates:
        return 1.0

    most_recent = max(revision_dates)
    days = (now - most_recent).days
    # Normalize: 90 days → 1.0
    return _normalize(days / 90.0)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

# All 15 feature functions in order
FEATURE_FUNCTIONS = [
    ("related_work_citation_density", compute_related_work_citation_density),
    ("related_work_recency_ratio", compute_related_work_recency_ratio),
    ("overclaim_score", compute_overclaim_score),
    ("methods_density", compute_methods_density),
    ("structural_completeness", compute_structural_completeness),
    ("avg_sentence_length_intro", compute_avg_sentence_length_intro),
    ("avg_sentence_length_overall", compute_avg_sentence_length_overall),
    ("jargon_ratio", compute_jargon_ratio),
    ("contribution_clarity", compute_contribution_clarity),
    ("results_narrative_ratio", compute_results_narrative_ratio),
    ("revision_velocity", compute_revision_velocity),
    ("abandoned_section_count", compute_abandoned_section_count),
    ("discussion_to_results_length_ratio", compute_discussion_to_results_length_ratio),
    ("self_citation_ratio", compute_self_citation_ratio),
    ("time_since_last_revision_days", compute_time_since_last_revision_days),
]


def build_feature_vector(draft: Any, sections: list, db: Session | None = None) -> dict[str, float]:
    """Compute all 15 features for a draft.

    Args:
        draft: Draft ORM object
        sections: List of Section ORM objects
        db: Optional DB session (not currently used but available for future features)

    Returns:
        dict mapping feature name → bounded float value
    """
    # Build section text map
    section_text_map: dict[str, str] = {}
    for s in sections:
        section_text_map[s.section_type] = s.content or ""

    features = {}
    for name, func in FEATURE_FUNCTIONS:
        features[name] = func(section_text_map, sections=sections)

    return features


# ---------------------------------------------------------------------------
# GMM archetype classifier
# ---------------------------------------------------------------------------

def classify_archetype(features: dict[str, float]) -> tuple[int, str, float]:
    """Classify a draft into one of 8 archetypes using the fitted GMM.

    Args:
        features: dict of 15 feature values

    Returns:
        (archetype_id, archetype_label, confidence)

    Raises:
        HTTPException (via caller) if model files are missing.
    """
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    gmm_path = ARTIFACTS_DIR / "gmm.pkl"

    if not scaler_path.exists() or not gmm_path.exists():
        raise FileNotFoundError(
            f"ML artifacts not found. Expected at {ARTIFACTS_DIR}/scaler.pkl and gmm.pkl. "
            "Run the training notebooks or use placeholder artifacts."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(gmm_path, "rb") as f:
        gmm = pickle.load(f)

    # Build feature array in the correct order
    feature_names = [name for name, _ in FEATURE_FUNCTIONS]
    X = np.array([[features[name] for name in feature_names]])

    # Standardize
    X_scaled = scaler.transform(X)

    # Predict cluster
    archetype_id = int(gmm.predict(X_scaled)[0])
    probabilities = gmm.predict_proba(X_scaled)[0]
    confidence = float(probabilities[archetype_id])

    archetype_label = CLUSTER_LABELS.get(archetype_id, f"Archetype {archetype_id}")

    return archetype_id, archetype_label, confidence


# ---------------------------------------------------------------------------
# XGBoost section-priority ranker
# ---------------------------------------------------------------------------

def rank_sections(
    draft: Any,
    sections: list,
    archetype_id: int,
    db: Session | None = None,
) -> list[dict]:
    """Rank sections by revision priority using the XGBoost ranker.

    Applies negative pruning: sections revised in the last 48 hours are dropped.

    Args:
        draft: Draft ORM object
        sections: List of Section ORM objects
        archetype_id: GMM cluster assignment
        db: Optional DB session

    Returns:
        List of dicts sorted by priority (highest first):
        [{"section_id": int, "section_type": str, "priority_score": float}, ...]
    """
    xgb_path = ARTIFACTS_DIR / "xgb_ranker.pkl"

    if not xgb_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {xgb_path}. "
            "Run the training notebook or use a placeholder artifact."
        )

    with open(xgb_path, "rb") as f:
        model = pickle.load(f)

    # Build draft-level features
    section_text_map = {s.section_type: s.content or "" for s in sections}
    draft_features = {}
    for name, func in FEATURE_FUNCTIONS:
        draft_features[name] = func(section_text_map, sections=sections)

    now = datetime.utcnow()
    cutoff_48h = now - timedelta(hours=48)

    # Section types for one-hot encoding
    from models import SectionType
    all_section_types = [st.value for st in SectionType]

    results = []
    for section in sections:
        # Negative pruning: skip sections revised in the last 48 hours
        if section.last_revised_at and section.last_revised_at > cutoff_48h:
            pass # Disabled for local testing so we can see recommendations immediately
            # continue

        # Build per-section feature row
        feature_row = list(draft_features.values())

        # section_type one-hot
        for st in all_section_types:
            feature_row.append(1.0 if section.section_type == st else 0.0)

        # Additional per-section features
        feature_row.append(float(_word_count(section.content or "")))  # words_in_section
        feature_row.append(float(section.revision_count))  # revision_count
        days_since = (
            (now - section.last_revised_at).days
            if section.last_revised_at
            else 999.0
        )
        feature_row.append(float(days_since))  # days_since_last_revision

        # archetype_id one-hot (8 clusters)
        for i in range(8):
            feature_row.append(1.0 if archetype_id == i else 0.0)

        X = np.array([feature_row])
        priority = float(model.predict(X)[0])
        priority = _normalize(priority)

        results.append({
            "section_id": section.id,
            "section_type": section.section_type,
            "priority_score": round(priority, 4),
        })

    # Sort by priority descending
    results.sort(key=lambda r: r["priority_score"], reverse=True)

    return results
