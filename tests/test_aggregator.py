"""Unit tests for processing/aggregator.py"""

import pytest
from schemas.models import ChunkDetail
from processing.aggregator import ScoreAggregator
from config.settings import THRESHOLD_MODES, YOUDEN_THRESHOLD, CLOSEST_POINT_THRESHOLD, FPR_THRESHOLD


def _make_chunks(scores: list[float]) -> list[ChunkDetail]:
    """Create a list of ChunkDetail with given scores."""
    return [
        ChunkDetail(
            chunk_index=i + 1,
            text=f"chunk {i + 1}",
            token_count=100,
            score=s,
            label="",
        )
        for i, s in enumerate(scores)
    ]


agg = ScoreAggregator()
THRESHOLD = YOUDEN_THRESHOLD


# Decision Logic Tests

class TestDecisionLogic:
    """Test the majority voting decision rules."""

    def test_all_ai(self):
        """All scores below threshold → 100% AI → 'AI-generated'."""
        chunks = _make_chunks([0.50, 0.60, 0.70, 0.80])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.ai_chunk_count == 4
        assert result.ai_percentage == 100.0
        assert result.final_decision == "AI-generated"
        assert all(c.label == "AI" for c in result.chunk_details)

    def test_all_human(self):
        """All scores above threshold → 0% AI → 'Human-written'."""
        chunks = _make_chunks([0.95, 0.96, 1.00, 0.94])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.ai_chunk_count == 0
        assert result.ai_percentage == 0.0
        assert result.final_decision == "Human-written"
        assert all(c.label == "Human" for c in result.chunk_details)

    def test_majority_ai(self):
        """60% AI chunks → 'AI-generated'."""
        scores = [0.80] * 6 + [0.95] * 4
        result = agg.aggregate(_make_chunks(scores), THRESHOLD)
        assert result.ai_percentage == 60.0
        assert result.final_decision == "AI-generated"

    def test_minority_ai(self):
        """40% AI chunks → 'Human-written but contains AI-generated parts'."""
        scores = [0.80] * 4 + [0.95] * 6
        result = agg.aggregate(_make_chunks(scores), THRESHOLD)
        assert result.ai_percentage == 40.0
        assert result.final_decision == "Human-written but contains AI-generated parts"

    def test_exactly_50_percent(self):
        """50% AI chunks → hybrid (not majority)."""
        scores = [0.80] * 5 + [0.95] * 5
        result = agg.aggregate(_make_chunks(scores), THRESHOLD)
        assert result.ai_percentage == 50.0
        assert result.final_decision == "Human-written but contains AI-generated parts"


# Boundary Tests

class TestBoundaries:
    """Test threshold boundary behavior."""

    def test_threshold_boundary_exact(self):
        """Score exactly at threshold → Human (not AI, threshold is strict <)."""
        chunks = _make_chunks([THRESHOLD])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.chunk_details[0].label == "Human"

    def test_threshold_just_below(self):
        """Score just below threshold → AI."""
        chunks = _make_chunks([THRESHOLD - 0.0001])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.chunk_details[0].label == "AI"

    def test_single_chunk_ai(self):
        chunks = _make_chunks([0.50])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.ai_percentage == 100.0
        assert result.final_decision == "AI-generated"

    def test_single_chunk_human(self):
        chunks = _make_chunks([0.95])
        result = agg.aggregate(chunks, THRESHOLD)
        assert result.ai_percentage == 0.0
        assert result.final_decision == "Human-written"


# Error Handling Tests

class TestErrorHandling:
    """Test error cases."""

    def test_empty_chunks_raises(self):
        """Empty chunks list should raise ValueError."""
        with pytest.raises(ValueError):
            agg.aggregate([], THRESHOLD)

    def test_applied_threshold_in_result(self):
        """Verify that applied_threshold is correctly stored."""
        chunks = _make_chunks([0.70, 0.90])
        result = agg.aggregate(chunks, 0.9015)
        assert result.applied_threshold == 0.9015


# Different Threshold Tests

class TestDifferentThresholds:
    """Test with different VietBinoculars thresholds."""

    def test_youden_threshold(self):
        chunks = _make_chunks([0.91, 0.92])  # Both below YOUDEN_THRESHOLD (~0.9280)
        result = agg.aggregate(chunks, YOUDEN_THRESHOLD)
        assert result.ai_chunk_count == 2

    def test_fpr_threshold(self):
        """FPR threshold (~0.8993) is lower — more permissive."""
        chunks = _make_chunks([0.91, 0.92])  # Both above FPR_THRESHOLD (~0.8993)
        result = agg.aggregate(chunks, FPR_THRESHOLD)
        assert result.ai_chunk_count == 0
        assert result.final_decision == "Human-written"

    def test_closest_point_threshold(self):
        chunks = _make_chunks([0.91, 0.92, 0.93])
        # With CLOSEST_POINT_THRESHOLD (~0.9251): 0.91 < 0.9251 → AI, 0.92 < 0.9251 → AI, 0.93 ≥ 0.9251 → Human
        result = agg.aggregate(chunks, CLOSEST_POINT_THRESHOLD)
        assert result.ai_chunk_count == 2

class TestDynamicThresholdModes:
    """Explicitly verify that the aggregator correctly applies all 3 user-selectable threshold modes."""

    def test_all_three_modes_from_settings(self):
        chunks = _make_chunks([0.90, 0.92, 0.93])

        # 1. Low FPR Mode (~0.8993)
        # All scores (0.90, 0.92, 0.93) are >= 0.8993 -> 0 AI chunks
        mode_fpr = "Low FPR (Fewer False Alarms)"
        res_fpr = agg.aggregate(chunks, threshold=THRESHOLD_MODES[mode_fpr])
        assert res_fpr.ai_chunk_count == 0
        assert res_fpr.applied_threshold == THRESHOLD_MODES[mode_fpr]

        # 2. Youden Mode (~0.9280)
        # 0.90, 0.92 < 0.9280 (AI). 0.93 >= 0.9280 (Human) -> 2 AI chunks
        mode_youden = "Youden (Balanced F1)"
        res_youden = agg.aggregate(chunks, threshold=THRESHOLD_MODES[mode_youden])
        assert res_youden.ai_chunk_count == 2
        assert res_youden.applied_threshold == THRESHOLD_MODES[mode_youden]

        # 3. Closest Point Mode (~0.9251)
        # 0.90, 0.92 < 0.9251 (AI). 0.93 >= 0.9251 (Human) -> 2 AI chunks
        mode_closest = "Closest Point (Near-Perfect)"
        res_closest = agg.aggregate(chunks, threshold=THRESHOLD_MODES[mode_closest])
        assert res_closest.ai_chunk_count == 2
        assert res_closest.applied_threshold == THRESHOLD_MODES[mode_closest]
