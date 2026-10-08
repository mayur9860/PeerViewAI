"""
Tests for ml_engine — feature engineering functions on small fixture drafts.

Tests verify:
- Each feature function returns a bounded float (0–1)
- Feature vector builder produces all 15 features
- Negative pruning (48-hour filter)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine import (
    FEATURE_FUNCTIONS,
    build_feature_vector,
    compute_abandoned_section_count,
    compute_avg_sentence_length_intro,
    compute_avg_sentence_length_overall,
    compute_contribution_clarity,
    compute_discussion_to_results_length_ratio,
    compute_jargon_ratio,
    compute_methods_density,
    compute_overclaim_score,
    compute_related_work_citation_density,
    compute_related_work_recency_ratio,
    compute_results_narrative_ratio,
    compute_revision_velocity,
    compute_self_citation_ratio,
    compute_structural_completeness,
    compute_time_since_last_revision_days,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_section_text_map():
    """A minimal but realistic section text map for testing."""
    return {
        "intro": (
            "Machine learning has transformed natural language processing. "
            "Recent advances suggest that large models may indicate better performance. "
            "We contribute three innovations to this field. "
            "Our contributions are: (1) a new architecture, (2) a training method, (3) analysis."
        ),
        "related_work": (
            "Smith (2023) proposed a transformer variant. "
            "Jones et al. (2022) demonstrated improvements using attention. "
            "Lee (2019) introduced the baseline method [1]. "
            "Park and Kim (2024) extended this with [2, 3] citations."
        ),
        "methods": (
            "We train our model using stochastic gradient descent with a learning rate of 0.001. "
            "The batch size is 32 and we train for 100 epochs. "
            "We use dropout with p=0.3 to prevent overfitting on our relatively small training set."
        ),
        "results": (
            "Group A achieved 94.3% accuracy compared to 91.2% for the baseline. "
            "Table 1 shows the full comparison. "
            "Figure 2 illustrates the training curves. "
            "The improvement of 3.1% is statistically significant with p=0.003."
        ),
        "discussion": (
            "Our results are consistent with prior work suggesting that attention mechanisms "
            "improve performance on this task. The improvement may indicate that the architecture "
            "captures longer-range dependencies. However, the results might not generalize to other domains."
        ),
    }


@pytest.fixture
def mock_sections():
    """Mock Section ORM objects for testing."""
    sections = []
    now = datetime.utcnow()

    for section_type, content in [
        ("intro", "Some introduction text with enough words to not be abandoned."),
        ("methods", "Short"),  # < 100 words, old revision → should be "abandoned"
        ("results", "Results with data and numbers and analysis discussion."),
    ]:
        s = MagicMock()
        s.section_type = section_type
        s.content = content
        s.revision_count = 3
        s.last_revised_at = now - timedelta(days=7)
        s.id = hash(section_type) % 1000
        sections.append(s)

    # Make methods section old and short (abandoned)
    sections[1].last_revised_at = now - timedelta(days=60)
    sections[1].revision_count = 1

    return sections


@pytest.fixture
def mock_draft():
    """Mock Draft ORM object."""
    draft = MagicMock()
    draft.id = 1
    draft.field = "cs"
    return draft


# ---------------------------------------------------------------------------
# Feature function tests
# ---------------------------------------------------------------------------

class TestFeatureFunctions:
    """Test that each feature function returns a bounded float."""

    def test_related_work_citation_density(self, sample_section_text_map):
        val = compute_related_work_citation_density(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_related_work_recency_ratio(self, sample_section_text_map):
        val = compute_related_work_recency_ratio(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_overclaim_score(self, sample_section_text_map):
        val = compute_overclaim_score(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_overclaim_score_with_strong_claims(self):
        text_map = {"results": "We prove conclusively that our method always works."}
        val = compute_overclaim_score(text_map)
        assert val > 0.5, "High overclaim text should produce high score"

    def test_overclaim_score_with_hedged_claims(self):
        text_map = {"results": "Our results suggest that the method may indicate improvement."}
        val = compute_overclaim_score(text_map)
        assert val < 0.5, "Hedged text should produce low overclaim score"

    def test_methods_density(self, sample_section_text_map):
        val = compute_methods_density(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_structural_completeness(self, sample_section_text_map):
        val = compute_structural_completeness(sample_section_text_map)
        assert val == 1.0, "All 5 expected sections present"

    def test_structural_completeness_partial(self):
        val = compute_structural_completeness({"intro": "text", "methods": "text"})
        assert val == 0.4, "2 of 5 expected sections"

    def test_avg_sentence_length_intro(self, sample_section_text_map):
        val = compute_avg_sentence_length_intro(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_avg_sentence_length_overall(self, sample_section_text_map):
        val = compute_avg_sentence_length_overall(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_jargon_ratio(self, sample_section_text_map):
        val = compute_jargon_ratio(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_contribution_clarity_present(self, sample_section_text_map):
        val = compute_contribution_clarity(sample_section_text_map)
        assert val == 1.0, "Intro contains 'Our contributions are'"

    def test_contribution_clarity_absent(self):
        val = compute_contribution_clarity({"intro": "This paper discusses methods."})
        assert val == 0.0

    def test_results_narrative_ratio(self, sample_section_text_map):
        val = compute_results_narrative_ratio(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_revision_velocity(self, sample_section_text_map, mock_sections):
        val = compute_revision_velocity(sample_section_text_map, sections=mock_sections)
        assert 0.0 <= val <= 1.0

    def test_abandoned_section_count(self, sample_section_text_map, mock_sections):
        val = compute_abandoned_section_count(sample_section_text_map, sections=mock_sections)
        assert 0.0 <= val <= 1.0

    def test_discussion_to_results_length_ratio(self, sample_section_text_map):
        val = compute_discussion_to_results_length_ratio(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_self_citation_ratio(self, sample_section_text_map):
        val = compute_self_citation_ratio(sample_section_text_map)
        assert 0.0 <= val <= 1.0

    def test_time_since_last_revision_days(self, sample_section_text_map, mock_sections):
        val = compute_time_since_last_revision_days(sample_section_text_map, sections=mock_sections)
        assert 0.0 <= val <= 1.0

    def test_empty_text_map(self):
        """All features should handle empty text gracefully."""
        empty_map = {}
        for name, func in FEATURE_FUNCTIONS:
            val = func(empty_map, sections=[])
            assert isinstance(val, float), f"{name} did not return float"
            assert 0.0 <= val <= 1.0, f"{name} out of bounds: {val}"


class TestBuildFeatureVector:
    """Test the feature vector builder."""

    def test_produces_all_15_features(self, mock_draft, mock_sections):
        features = build_feature_vector(mock_draft, mock_sections)
        assert len(features) == 15
        assert all(isinstance(v, float) for v in features.values())

    def test_all_features_bounded(self, mock_draft, mock_sections):
        features = build_feature_vector(mock_draft, mock_sections)
        for name, val in features.items():
            assert 0.0 <= val <= 1.0, f"Feature '{name}' out of bounds: {val}"


class TestNegativePruning:
    """Test 48-hour negative pruning in the ranker."""

    def test_recently_revised_sections_excluded(self):
        """Sections revised in the last 48h should be excluded from ranking."""
        from ml_engine import rank_sections

        now = datetime.utcnow()

        # Create sections: one revised 1 hour ago, one revised 5 days ago
        recent_section = MagicMock()
        recent_section.id = 1
        recent_section.section_type = "intro"
        recent_section.content = "Recent content"
        recent_section.revision_count = 5
        recent_section.last_revised_at = now - timedelta(hours=1)

        old_section = MagicMock()
        old_section.id = 2
        old_section.section_type = "methods"
        old_section.content = "Old content"
        old_section.revision_count = 2
        old_section.last_revised_at = now - timedelta(days=5)

        draft = MagicMock()
        draft.id = 1

        try:
            results = rank_sections(draft, [recent_section, old_section], 0)
            # Recent section should be excluded
            section_ids = [r["section_id"] for r in results]
            assert 1 not in section_ids, "Recently revised section should be pruned"
            assert 2 in section_ids, "Old section should be included"
        except FileNotFoundError:
            pytest.skip("XGBoost model not available")
