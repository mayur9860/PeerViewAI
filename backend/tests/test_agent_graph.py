"""
Tests for agent_graph — reviewer's unsafe-detection heuristic.

Tests verify:
- Safe responses (questions only) pass through
- Unsafe responses (containing replacement prose) are flagged
- Max-retry fallback behavior
"""

import sys
import re
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_graph import FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Reviewer heuristic tests (regex check extracted for unit testing)
# ---------------------------------------------------------------------------

def _check_quoted_text_heuristic(response: str) -> tuple[str, str]:
    """Extract the reviewer's deterministic heuristic check.

    Returns (verdict, reason) — "safe"/"unsafe".
    """
    quoted_strings = re.findall(
        r'["\u201c\u201d]((?:[^"\u201c\u201d]|\n)+?)["\u201c\u201d]',
        response,
    )

    for quoted in quoted_strings:
        word_count = len(quoted.split())
        if word_count > 8:
            # Check for attribution
            context_before = response[:response.find(quoted)][-100:]
            attribution_patterns = [
                r'you\s+wrote', r'your\s+(?:text|draft|section|paper)',
                r'the\s+(?:original|user)', r'from\s+(?:the|your)',
                r'currently\s+(?:reads|says|states)',
            ]
            is_attributed = any(
                re.search(p, context_before, re.IGNORECASE)
                for p in attribution_patterns
            )
            if not is_attributed:
                return "unsafe", f"Unattributed quoted text: '{quoted[:50]}...'"

    return "safe", ""


class TestReviewerHeuristic:
    """Test the reviewer's regex/heuristic check for replacement prose."""

    def test_safe_questions_only(self):
        """Pure Socratic questions should pass."""
        response = (
            "What evidence in your results section supports this claim? "
            "Have you considered how a skeptical reviewer might interpret Figure 3? "
            "Could you articulate why this finding contradicts the baseline results?"
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "safe"

    def test_safe_short_quotes(self):
        """Short quoted phrases (≤8 words) from user's text should pass."""
        response = (
            'You mention "improved accuracy" in paragraph 2 — what specific metric are you referring to? '
            'The phrase "state of the art" appears three times — is this justified?'
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "safe"

    def test_unsafe_replacement_prose(self):
        """Unattributed quoted text >8 words should be flagged as replacement."""
        response = (
            'Consider rewriting this as: "Our proposed method achieves significant improvements '
            'over the baseline approach by leveraging novel attention mechanisms that capture '
            'long-range dependencies more effectively."'
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "unsafe"
        assert "Unattributed" in reason

    def test_safe_attributed_quote(self):
        """Attributed quotes from the user's draft should pass."""
        response = (
            'Your text currently reads: "We demonstrate conclusively that our approach always '
            'outperforms all existing baselines across every benchmark." — Is this claim supported '
            'by your data in Table 2?'
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "safe"

    def test_safe_with_user_wrote_attribution(self):
        """'You wrote' attribution should pass."""
        response = (
            'In the introduction, you wrote "This paper presents a novel framework for understanding '
            'the relationship between attention and performance" — could you be more specific about '
            'what makes it novel?'
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "safe"

    def test_unsafe_suggested_rewrite(self):
        """Suggested rewrites without attribution should be flagged."""
        response = (
            'A stronger version might be: "The experimental results demonstrate a statistically '
            'significant improvement in classification accuracy compared to three competitive baselines."'
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "unsafe"

    def test_empty_response(self):
        """Empty response should be safe (no quoted text to flag)."""
        verdict, reason = _check_quoted_text_heuristic("")
        assert verdict == "safe"

    def test_no_quotes_at_all(self):
        """Response with no quotes should be safe."""
        response = (
            "What methodology did you use to validate this claim? "
            "How might a reviewer in a different subfield interpret these results?"
        )
        verdict, reason = _check_quoted_text_heuristic(response)
        assert verdict == "safe"


class TestFallbackResponse:
    """Test the hardcoded fallback response."""

    def test_fallback_is_socratic(self):
        """Fallback response should contain questions, not statements."""
        assert "?" in FALLBACK_RESPONSE, "Fallback should contain questions"

    def test_fallback_no_replacement_prose(self):
        """Fallback response should not contain quoted replacement text."""
        verdict, _ = _check_quoted_text_heuristic(FALLBACK_RESPONSE)
        assert verdict == "safe", "Fallback should pass the heuristic check"

    def test_fallback_is_not_empty(self):
        """Fallback response should have substance."""
        assert len(FALLBACK_RESPONSE) > 50
