"""Tests for trait feedback module."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.analysis.trait_feedback import (
    FeedbackStore,
    LearnedPattern,
    TraitFeedback,
    load_feedback_store,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_feedback_path(tmp_path) -> Path:
    """Create a temporary path for feedback storage."""
    return tmp_path / "feedback" / "test_feedback.json"


@pytest.fixture
def feedback_store(temp_feedback_path) -> FeedbackStore:
    """Create a FeedbackStore with temporary storage."""
    return FeedbackStore(storage_path=temp_feedback_path)


@pytest.fixture
def sample_feedback() -> TraitFeedback:
    """Create a sample TraitFeedback entry."""
    return TraitFeedback(
        feedback_id=str(uuid.uuid4()),
        trait_name="payment_method",
        trait_path="custom.payment_method",
        timestamp=datetime.now(timezone.utc),
        feedback_type="obvious",
        reason="Payment method is a result, not a driver",
        use_case="segmentation",
    )


# =============================================================================
# TESTS: TraitFeedback
# =============================================================================


class TestTraitFeedback:
    """Tests for TraitFeedback dataclass."""

    def test_create_feedback(self):
        """Test creating a feedback entry."""
        feedback = TraitFeedback(
            feedback_id="test-123",
            trait_name="test_trait",
            trait_path="extra.test_trait",
            timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            feedback_type="useful",
        )

        assert feedback.feedback_id == "test-123"
        assert feedback.trait_name == "test_trait"
        assert feedback.feedback_type == "useful"
        assert feedback.reason is None

    def test_to_dict(self, sample_feedback):
        """Test conversion to dictionary."""
        result = sample_feedback.to_dict()

        assert result["feedback_id"] == sample_feedback.feedback_id
        assert result["trait_name"] == "payment_method"
        assert result["feedback_type"] == "obvious"
        assert result["reason"] == "Payment method is a result, not a driver"
        assert "timestamp" in result

    def test_from_dict(self, sample_feedback):
        """Test creating from dictionary."""
        data = sample_feedback.to_dict()
        restored = TraitFeedback.from_dict(data)

        assert restored.feedback_id == sample_feedback.feedback_id
        assert restored.trait_name == sample_feedback.trait_name
        assert restored.feedback_type == sample_feedback.feedback_type
        assert restored.reason == sample_feedback.reason

    def test_all_feedback_types(self):
        """Test all feedback types are valid."""
        valid_types = ["useful", "not_useful", "obvious", "incorrect", "needs_context"]

        for feedback_type in valid_types:
            feedback = TraitFeedback(
                feedback_id="test",
                trait_name="test",
                trait_path="test",
                timestamp=datetime.now(timezone.utc),
                feedback_type=feedback_type,
            )
            assert feedback.feedback_type == feedback_type


# =============================================================================
# TESTS: LearnedPattern
# =============================================================================


class TestLearnedPattern:
    """Tests for LearnedPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a learned pattern."""
        pattern = LearnedPattern(
            pattern="payment",
            adjustment=0.5,
            sample_size=5,
        )

        assert pattern.pattern == "payment"
        assert pattern.adjustment == 0.5
        assert pattern.sample_size == 5
        assert pattern.last_updated is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern = LearnedPattern(
            pattern="shipping",
            adjustment=0.6,
            sample_size=3,
        )

        result = pattern.to_dict()

        assert result["pattern"] == "shipping"
        assert result["adjustment"] == 0.6
        assert result["sample_size"] == 3
        assert "last_updated" in result

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "pattern": "payment",
            "adjustment": 0.4,
            "sample_size": 10,
            "last_updated": "2025-01-15T10:30:00+00:00",
        }

        pattern = LearnedPattern.from_dict(data)

        assert pattern.pattern == "payment"
        assert pattern.adjustment == 0.4
        assert pattern.sample_size == 10


# =============================================================================
# TESTS: FeedbackStore - Basic Operations
# =============================================================================


class TestFeedbackStoreBasics:
    """Tests for basic FeedbackStore operations."""

    def test_create_store(self, temp_feedback_path):
        """Test creating a new feedback store."""
        store = FeedbackStore(storage_path=temp_feedback_path)

        assert store is not None
        assert len(store.get_all_feedback()) == 0

    def test_add_feedback(self, feedback_store, sample_feedback):
        """Test adding feedback."""
        feedback_store.add_feedback(sample_feedback)

        all_feedback = feedback_store.get_all_feedback()
        assert len(all_feedback) == 1
        assert all_feedback[0].trait_name == "payment_method"

    def test_add_feedback_simple(self, feedback_store):
        """Test add_feedback_simple convenience method."""
        feedback = feedback_store.add_feedback_simple(
            trait_name="test_trait",
            trait_path="extra.test_trait",
            feedback_type="useful",
            reason="Very helpful",
        )

        assert feedback.trait_name == "test_trait"
        assert feedback.feedback_type == "useful"
        assert feedback.reason == "Very helpful"
        assert len(feedback_store.get_all_feedback()) == 1

    def test_get_feedback_for_trait(self, feedback_store, sample_feedback):
        """Test getting feedback for specific trait."""
        feedback_store.add_feedback(sample_feedback)

        # Add another feedback for different trait
        other_feedback = TraitFeedback(
            feedback_id=str(uuid.uuid4()),
            trait_name="category",
            trait_path="extra.category",
            timestamp=datetime.now(timezone.utc),
            feedback_type="useful",
        )
        feedback_store.add_feedback(other_feedback)

        # Get feedback for payment_method only
        payment_feedback = feedback_store.get_feedback_for_trait("payment_method")

        assert len(payment_feedback) == 1
        assert payment_feedback[0].trait_name == "payment_method"

    def test_persistence(self, temp_feedback_path, sample_feedback):
        """Test that feedback persists across store instances."""
        # Create store and add feedback
        store1 = FeedbackStore(storage_path=temp_feedback_path)
        store1.add_feedback(sample_feedback)

        # Create new store instance
        store2 = FeedbackStore(storage_path=temp_feedback_path)

        assert len(store2.get_all_feedback()) == 1
        assert store2.get_all_feedback()[0].trait_name == "payment_method"

    def test_clear_feedback(self, feedback_store, sample_feedback):
        """Test clearing all feedback."""
        feedback_store.add_feedback(sample_feedback)
        assert len(feedback_store.get_all_feedback()) == 1

        feedback_store.clear()

        assert len(feedback_store.get_all_feedback()) == 0


# =============================================================================
# TESTS: FeedbackStore - Adjustment Calculation
# =============================================================================


class TestFeedbackStoreAdjustments:
    """Tests for FeedbackStore adjustment calculations."""

    def test_no_feedback_neutral_adjustment(self, feedback_store):
        """Test that no feedback gives neutral adjustment."""
        adjustment = feedback_store.get_adjustment_factor("unknown_trait")

        assert adjustment == 1.0

    def test_single_negative_feedback(self, feedback_store):
        """Test adjustment for single negative feedback."""
        feedback_store.add_feedback_simple(
            trait_name="payment_method",
            trait_path="custom.payment_method",
            feedback_type="obvious",
        )

        adjustment = feedback_store.get_adjustment_factor("payment_method")

        # Single negative: 1.0 - 0.10 = 0.90
        assert adjustment == 0.90

    def test_multiple_negative_feedback(self, feedback_store):
        """Test adjustment accumulates with multiple negatives."""
        # Add 5 negative feedback entries
        for _ in range(5):
            feedback_store.add_feedback_simple(
                trait_name="payment_method",
                trait_path="custom.payment_method",
                feedback_type="obvious",
            )

        adjustment = feedback_store.get_adjustment_factor("payment_method")

        # 5 negatives: 1.0 - (5 * 0.10) = 0.50
        assert adjustment == 0.50

    def test_positive_feedback_boost(self, feedback_store):
        """Test that positive feedback increases adjustment."""
        for _ in range(4):
            feedback_store.add_feedback_simple(
                trait_name="category",
                trait_path="extra.category",
                feedback_type="useful",
            )

        adjustment = feedback_store.get_adjustment_factor("category")

        # 4 useful: 1.0 + (4 * 0.05) = 1.20
        assert adjustment == 1.20

    def test_mixed_feedback(self, feedback_store):
        """Test adjustment with mixed positive and negative feedback."""
        # 2 negative
        for _ in range(2):
            feedback_store.add_feedback_simple(
                trait_name="test_trait",
                trait_path="extra.test_trait",
                feedback_type="not_useful",
            )
        # 3 positive
        for _ in range(3):
            feedback_store.add_feedback_simple(
                trait_name="test_trait",
                trait_path="extra.test_trait",
                feedback_type="useful",
            )

        adjustment = feedback_store.get_adjustment_factor("test_trait")

        # 2 negative, 3 positive: 1.0 - (2 * 0.10) + (3 * 0.05) = 0.95
        assert adjustment == pytest.approx(0.95)

    def test_adjustment_minimum_clamp(self, feedback_store):
        """Test that adjustment doesn't go below minimum."""
        # Add 20 negative feedback entries (would be 1.0 - 2.0 = -1.0 unclamped)
        for _ in range(20):
            feedback_store.add_feedback_simple(
                trait_name="bad_trait",
                trait_path="extra.bad_trait",
                feedback_type="not_useful",
            )

        adjustment = feedback_store.get_adjustment_factor("bad_trait")

        # Should be clamped at 0.2 (80% max penalty)
        assert adjustment == 0.2

    def test_adjustment_maximum_clamp(self, feedback_store):
        """Test that adjustment doesn't go above maximum."""
        # Add 20 positive feedback entries (would be 1.0 + 1.0 = 2.0 unclamped)
        for _ in range(20):
            feedback_store.add_feedback_simple(
                trait_name="great_trait",
                trait_path="extra.great_trait",
                feedback_type="useful",
            )

        adjustment = feedback_store.get_adjustment_factor("great_trait")

        # Should be clamped at 1.5 (50% max boost)
        assert adjustment == 1.5


# =============================================================================
# TESTS: FeedbackStore - Pattern Learning
# =============================================================================


class TestFeedbackStorePatterns:
    """Tests for FeedbackStore pattern learning."""

    def test_learn_payment_pattern(self, feedback_store):
        """Test that payment pattern is learned from feedback."""
        # Add 5 negative feedbacks for payment-related traits
        for trait in ["payment_method", "payment_type", "payment_status"]:
            for _ in range(2):
                feedback_store.add_feedback_simple(
                    trait_name=trait,
                    trait_path=f"custom.{trait}",
                    feedback_type="obvious",
                )

        patterns = feedback_store.get_patterns()

        assert "payment" in patterns
        assert patterns["payment"].sample_size >= 3

    def test_pattern_applies_to_new_traits(self, feedback_store):
        """Test learned patterns apply to new traits."""
        # Learn payment pattern
        for _ in range(5):
            feedback_store.add_feedback_simple(
                trait_name="payment_method",
                trait_path="custom.payment_method",
                feedback_type="obvious",
            )

        # New payment-related trait should get adjusted
        adjustment = feedback_store.get_adjustment_factor("payment_gateway")

        # Should be less than 1.0 due to learned pattern
        assert adjustment < 1.0

    def test_pattern_not_learned_insufficient_samples(self, feedback_store):
        """Test pattern is not learned with insufficient samples."""
        # Add only 2 feedbacks (below MIN_PATTERN_SAMPLES of 3)
        for _ in range(2):
            feedback_store.add_feedback_simple(
                trait_name="rare_pattern_trait",
                trait_path="extra.rare_pattern_trait",
                feedback_type="obvious",
            )

        patterns = feedback_store.get_patterns()

        # Should not have learned a pattern yet
        assert len(patterns) == 0


# =============================================================================
# TESTS: FeedbackStore - Summary
# =============================================================================


class TestFeedbackStoreSummary:
    """Tests for FeedbackStore summary functionality."""

    def test_empty_summary(self, feedback_store):
        """Test summary for empty store."""
        summary = feedback_store.get_summary()

        assert summary["total_feedback"] == 0
        assert summary["by_type"] == {}
        assert summary["by_trait"] == {}

    def test_summary_with_feedback(self, feedback_store):
        """Test summary with feedback entries."""
        # Add various feedback
        feedback_store.add_feedback_simple("trait_a", "path_a", "useful")
        feedback_store.add_feedback_simple("trait_a", "path_a", "useful")
        feedback_store.add_feedback_simple("trait_b", "path_b", "obvious")
        feedback_store.add_feedback_simple("trait_c", "path_c", "not_useful")

        summary = feedback_store.get_summary()

        assert summary["total_feedback"] == 4
        assert summary["by_type"]["useful"] == 2
        assert summary["by_type"]["obvious"] == 1
        assert summary["by_type"]["not_useful"] == 1
        assert summary["by_trait"]["trait_a"] == 2


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_feedback_store(self, temp_feedback_path):
        """Test load_feedback_store function."""
        store = load_feedback_store(temp_feedback_path)

        assert isinstance(store, FeedbackStore)
        assert store.storage_path == temp_feedback_path


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_corrupted_json_handling(self, temp_feedback_path):
        """Test handling of corrupted JSON file."""
        # Create a corrupted file
        temp_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_feedback_path, "w") as f:
            f.write("{ invalid json }")

        # Should not crash, should start with empty feedback
        store = FeedbackStore(storage_path=temp_feedback_path)
        assert len(store.get_all_feedback()) == 0

    def test_missing_fields_in_json(self, temp_feedback_path):
        """Test handling of feedback entries with missing fields."""
        # Create a file with incomplete entry
        temp_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "feedback": [
                {"feedback_id": "test", "trait_name": "test"}
                # Missing required fields
            ],
            "learned_patterns": {},
        }
        with open(temp_feedback_path, "w") as f:
            json.dump(data, f)

        # Should skip invalid entries
        store = FeedbackStore(storage_path=temp_feedback_path)
        assert len(store.get_all_feedback()) == 0

    def test_empty_trait_name(self, feedback_store):
        """Test handling of empty trait name."""
        feedback_store.add_feedback_simple(
            trait_name="",
            trait_path="",
            feedback_type="useful",
        )

        feedback = feedback_store.get_feedback_for_trait("")
        assert len(feedback) == 1
