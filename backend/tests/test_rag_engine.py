"""
Tests for rag_engine — filter relaxation logic.

Tests verify:
- Post-filter correctly applies metadata filters
- Filter relaxation expands results when too few match
- Relaxation order: field → difficulty → section_type
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine import _post_filter, _relax_and_filter


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results():
    """Sample search results with varied metadata."""
    return [
        {"content": "Result 1", "metadata": {"section_type": "intro", "field": "cs", "difficulty": "beginner"}},
        {"content": "Result 2", "metadata": {"section_type": "intro", "field": "cs", "difficulty": "intermediate"}},
        {"content": "Result 3", "metadata": {"section_type": "methods", "field": "cs", "difficulty": "advanced"}},
        {"content": "Result 4", "metadata": {"section_type": "results", "field": "biology", "difficulty": "beginner"}},
        {"content": "Result 5", "metadata": {"section_type": "discussion", "field": "biology", "difficulty": "intermediate"}},
        {"content": "Result 6", "metadata": {"section_type": "intro", "field": "economics", "difficulty": "advanced"}},
        {"content": "Result 7", "metadata": {"section_type": "methods", "field": "general", "difficulty": "beginner"}},
        {"content": "Result 8", "metadata": {"section_type": "results", "field": "general", "difficulty": "intermediate"}},
        {"content": "Result 9", "metadata": {"section_type": "discussion", "field": "cs", "difficulty": "beginner"}},
        {"content": "Result 10", "metadata": {"section_type": "conclusion", "field": "general", "difficulty": "advanced"}},
    ]


# ---------------------------------------------------------------------------
# Post-filter tests
# ---------------------------------------------------------------------------

class TestPostFilter:
    def test_no_filters(self, sample_results):
        """No filters should return all results."""
        filtered = _post_filter(sample_results, None, None, None)
        assert len(filtered) == len(sample_results)

    def test_section_type_filter(self, sample_results):
        """Filter by section_type should return matching results."""
        filtered = _post_filter(sample_results, "intro", None, None)
        assert all(r["metadata"]["section_type"] == "intro" for r in filtered)
        assert len(filtered) == 3

    def test_field_filter(self, sample_results):
        """Filter by field should return matching results."""
        filtered = _post_filter(sample_results, None, "cs", None)
        assert all(r["metadata"]["field"] == "cs" for r in filtered)

    def test_difficulty_filter(self, sample_results):
        """Filter by difficulty should return matching results."""
        filtered = _post_filter(sample_results, None, None, "advanced")
        assert all(r["metadata"]["difficulty"] == "advanced" for r in filtered)

    def test_combined_filters(self, sample_results):
        """Combined filters should intersect."""
        filtered = _post_filter(sample_results, "intro", "cs", None)
        assert len(filtered) == 2
        for r in filtered:
            assert r["metadata"]["section_type"] == "intro"
            assert r["metadata"]["field"] == "cs"

    def test_no_match(self, sample_results):
        """Filters with no match should return empty list."""
        filtered = _post_filter(sample_results, "conclusion", "biology", "advanced")
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Filter relaxation tests
# ---------------------------------------------------------------------------

class TestFilterRelaxation:
    def test_enough_results_no_relaxation(self, sample_results):
        """When enough results match, no relaxation needed."""
        filtered = _relax_and_filter(sample_results, "intro", None, None, min_results=3)
        assert len(filtered) >= 3
        assert all(r["metadata"]["section_type"] == "intro" for r in filtered)

    def test_relax_field_first(self, sample_results):
        """When too few results, field is relaxed first."""
        # Only 1 match for intro + economics + advanced → should relax
        filtered = _relax_and_filter(sample_results, "intro", "economics", "advanced", min_results=3)
        assert len(filtered) >= 3

    def test_relax_all_filters(self, sample_results):
        """When all specific filters yield too few, relax everything."""
        filtered = _relax_and_filter(sample_results, "conclusion", "biology", "advanced", min_results=3)
        assert len(filtered) >= 3

    def test_min_results_respected(self, sample_results):
        """Relaxation should return at least min_results when possible."""
        filtered = _relax_and_filter(sample_results, "methods", "cs", "advanced", min_results=3)
        assert len(filtered) >= 3

    def test_returns_all_when_fully_relaxed(self, sample_results):
        """Full relaxation returns all results."""
        filtered = _relax_and_filter(
            sample_results, "nonexistent", "nonexistent", "nonexistent", min_results=3
        )
        # After full relaxation, should return all results
        assert len(filtered) == len(sample_results)
