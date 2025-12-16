"""
Module: trait_feedback

Purpose: Persist and manage user feedback on trait recommendations.

This module implements a learning loop where users can rate trait recommendations
and the system adjusts future scores based on accumulated feedback.

Key components:
- TraitFeedback: Dataclass for individual feedback entries
- FeedbackStore: Persistence layer with adjustment calculation
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


FeedbackType = Literal["useful", "not_useful", "obvious", "incorrect", "needs_context"]


@dataclass
class TraitFeedback:
    """User feedback on a trait's usefulness."""

    feedback_id: str
    trait_name: str
    trait_path: str
    timestamp: datetime

    # Feedback classification
    feedback_type: FeedbackType

    # Optional details
    reason: str | None = None
    suggested_alternative: str | None = None

    # Context
    use_case: str | None = None  # "segmentation", "personalization", "retention"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feedback_id": self.feedback_id,
            "trait_name": self.trait_name,
            "trait_path": self.trait_path,
            "timestamp": self.timestamp.isoformat(),
            "feedback_type": self.feedback_type,
            "reason": self.reason,
            "suggested_alternative": self.suggested_alternative,
            "use_case": self.use_case,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraitFeedback":
        """Create from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            trait_name=data["trait_name"],
            trait_path=data["trait_path"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            feedback_type=data["feedback_type"],
            reason=data.get("reason"),
            suggested_alternative=data.get("suggested_alternative"),
            use_case=data.get("use_case"),
        )


@dataclass
class LearnedPattern:
    """A learned pattern from accumulated feedback."""

    pattern: str  # Regex pattern to match trait names
    adjustment: float  # Multiplier to apply (0.2 - 1.5)
    sample_size: int  # Number of feedback entries supporting this
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern": self.pattern,
            "adjustment": self.adjustment,
            "sample_size": self.sample_size,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedPattern":
        """Create from dictionary."""
        return cls(
            pattern=data["pattern"],
            adjustment=data["adjustment"],
            sample_size=data["sample_size"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )


# =============================================================================
# FEEDBACK STORE
# =============================================================================


class FeedbackStore:
    """
    Persists and retrieves trait feedback.

    Stores feedback in a JSON file and calculates score adjustments
    based on accumulated user ratings.

    Example:
        >>> store = FeedbackStore()
        >>> store.add_feedback(TraitFeedback(..., feedback_type="obvious"))
        >>> adjustment = store.get_adjustment_factor("payment_method")
        >>> adjusted_score = original_score * adjustment
    """

    # Default storage path
    DEFAULT_PATH = Path("data/feedback/trait_feedback.json")

    # Adjustment formula parameters (Moderate: 5 negatives = 50% penalty)
    NEGATIVE_PENALTY = 0.10  # Each negative reduces by 10%
    POSITIVE_BOOST = 0.05  # Each positive increases by 5%
    MIN_ADJUSTMENT = 0.2  # Maximum penalty (80% reduction)
    MAX_ADJUSTMENT = 1.5  # Maximum boost (50% increase)

    # Pattern learning parameters
    MIN_PATTERN_SAMPLES = 3  # Minimum feedback to learn a pattern

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize FeedbackStore.

        Args:
            storage_path: Path to JSON storage file. Creates parent directories if needed.
        """
        self.storage_path = Path(storage_path) if storage_path else self.DEFAULT_PATH
        self._feedback: list[TraitFeedback] = []
        self._learned_patterns: dict[str, LearnedPattern] = {}
        self._load()

    def _load(self) -> None:
        """Load feedback from storage file."""
        if not self.storage_path.exists():
            logger.debug(f"No existing feedback file at {self.storage_path}")
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Load feedback entries
            for entry in data.get("feedback", []):
                try:
                    self._feedback.append(TraitFeedback.from_dict(entry))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid feedback entry: {e}")

            # Load learned patterns
            for pattern_name, pattern_data in data.get("learned_patterns", {}).items():
                try:
                    self._learned_patterns[pattern_name] = LearnedPattern.from_dict(
                        pattern_data
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid pattern {pattern_name}: {e}")

            logger.info(
                f"Loaded {len(self._feedback)} feedback entries, "
                f"{len(self._learned_patterns)} learned patterns"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse feedback file: {e}")
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")

    def _save(self) -> None:
        """Save feedback to storage file."""
        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "feedback": [f.to_dict() for f in self._feedback],
            "learned_patterns": {
                name: pattern.to_dict()
                for name, pattern in self._learned_patterns.items()
            },
        }

        try:
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved feedback to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    def add_feedback(self, feedback: TraitFeedback) -> None:
        """
        Add new feedback and persist.

        Args:
            feedback: TraitFeedback entry to add
        """
        self._feedback.append(feedback)
        self._update_patterns()
        self._save()
        logger.info(
            f"Added feedback for '{feedback.trait_name}': {feedback.feedback_type}"
        )

    def add_feedback_simple(
        self,
        trait_name: str,
        trait_path: str,
        feedback_type: FeedbackType,
        *,
        reason: str | None = None,
        use_case: str | None = None,
    ) -> TraitFeedback:
        """
        Convenience method to add feedback with auto-generated ID and timestamp.

        Args:
            trait_name: Name of the trait
            trait_path: Full path of the trait
            feedback_type: Type of feedback
            reason: Optional reason for feedback
            use_case: Optional use case context

        Returns:
            The created TraitFeedback object
        """
        import uuid

        feedback = TraitFeedback(
            feedback_id=str(uuid.uuid4()),
            trait_name=trait_name,
            trait_path=trait_path,
            timestamp=datetime.now(timezone.utc),
            feedback_type=feedback_type,
            reason=reason,
            use_case=use_case,
        )
        self.add_feedback(feedback)
        return feedback

    def get_feedback_for_trait(self, trait_name: str) -> list[TraitFeedback]:
        """
        Get all feedback for a specific trait.

        Args:
            trait_name: Name of the trait

        Returns:
            List of TraitFeedback entries for this trait
        """
        return [f for f in self._feedback if f.trait_name == trait_name]

    def get_all_feedback(self) -> list[TraitFeedback]:
        """Get all feedback entries."""
        return list(self._feedback)

    def get_adjustment_factor(self, trait_name: str) -> float:
        """
        Calculate score adjustment based on accumulated feedback.

        Uses the Moderate formula: 5 negatives = 50% penalty
        - Each negative reduces by 10%
        - Each positive increases by 5%
        - Clamped between 0.2 (80% penalty) and 1.5 (50% boost)

        Args:
            trait_name: Name of the trait

        Returns:
            Multiplier (0.2 to 1.5) to apply to trait scores
        """
        # First check for direct feedback on this trait
        feedback = self.get_feedback_for_trait(trait_name)

        if feedback:
            return self._calculate_adjustment(feedback)

        # Check learned patterns
        for pattern_name, pattern in self._learned_patterns.items():
            try:
                if re.search(pattern.pattern, trait_name, re.IGNORECASE):
                    logger.debug(
                        f"Applying learned pattern '{pattern_name}' to '{trait_name}': {pattern.adjustment}"
                    )
                    return pattern.adjustment
            except re.error:
                continue

        # No feedback or pattern match - neutral
        return 1.0

    def _calculate_adjustment(self, feedback: list[TraitFeedback]) -> float:
        """Calculate adjustment factor from feedback list."""
        useful_count = sum(1 for f in feedback if f.feedback_type == "useful")
        negative_count = sum(
            1 for f in feedback if f.feedback_type in ("not_useful", "obvious", "incorrect")
        )

        # Moderate penalty: 5 negatives = 50% penalty (0.5 multiplier)
        adjustment = (
            1.0
            - (negative_count * self.NEGATIVE_PENALTY)
            + (useful_count * self.POSITIVE_BOOST)
        )

        # Clamp between min and max
        return max(self.MIN_ADJUSTMENT, min(self.MAX_ADJUSTMENT, adjustment))

    def _update_patterns(self) -> None:
        """Learn patterns from accumulated feedback."""
        # Group feedback by trait name prefix
        prefix_feedback: dict[str, list[TraitFeedback]] = {}

        for feedback in self._feedback:
            # Extract potential pattern prefixes
            trait_lower = feedback.trait_name.lower()

            # Try common prefixes
            for prefix in ["payment", "shipping", "order", "category", "brand", "device"]:
                if prefix in trait_lower:
                    if prefix not in prefix_feedback:
                        prefix_feedback[prefix] = []
                    prefix_feedback[prefix].append(feedback)
                    break

        # Create or update patterns
        for prefix, feedbacks in prefix_feedback.items():
            if len(feedbacks) >= self.MIN_PATTERN_SAMPLES:
                adjustment = self._calculate_adjustment(feedbacks)

                # Only create pattern if significantly different from neutral
                if abs(adjustment - 1.0) >= 0.15:
                    self._learned_patterns[prefix] = LearnedPattern(
                        pattern=prefix,
                        adjustment=adjustment,
                        sample_size=len(feedbacks),
                        last_updated=datetime.now(timezone.utc),
                    )
                    logger.info(
                        f"Learned pattern '{prefix}': adjustment={adjustment:.2f} "
                        f"(from {len(feedbacks)} samples)"
                    )

    def get_patterns(self) -> dict[str, LearnedPattern]:
        """
        Get all learned patterns.

        Returns:
            Dict mapping pattern names to LearnedPattern objects
        """
        return dict(self._learned_patterns)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics about stored feedback.

        Returns:
            Dict with feedback statistics
        """
        total = len(self._feedback)
        if total == 0:
            return {
                "total_feedback": 0,
                "by_type": {},
                "by_trait": {},
                "learned_patterns": {},
            }

        by_type: dict[str, int] = {}
        by_trait: dict[str, int] = {}

        for feedback in self._feedback:
            by_type[feedback.feedback_type] = by_type.get(feedback.feedback_type, 0) + 1
            by_trait[feedback.trait_name] = by_trait.get(feedback.trait_name, 0) + 1

        return {
            "total_feedback": total,
            "by_type": by_type,
            "by_trait": by_trait,
            "learned_patterns": {
                name: {"adjustment": p.adjustment, "sample_size": p.sample_size}
                for name, p in self._learned_patterns.items()
            },
        }

    def clear(self) -> None:
        """Clear all feedback (use with caution)."""
        self._feedback = []
        self._learned_patterns = {}
        self._save()
        logger.info("Cleared all feedback")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_feedback_store(path: Path | str | None = None) -> FeedbackStore:
    """
    Load or create a feedback store.

    Args:
        path: Optional path to storage file

    Returns:
        FeedbackStore instance
    """
    return FeedbackStore(storage_path=path)


def get_adjustment_for_trait(trait_name: str, store: FeedbackStore | None = None) -> float:
    """
    Get adjustment factor for a trait.

    Args:
        trait_name: Name of the trait
        store: Optional FeedbackStore (uses default if None)

    Returns:
        Adjustment multiplier
    """
    if store is None:
        store = FeedbackStore()
    return store.get_adjustment_factor(trait_name)
