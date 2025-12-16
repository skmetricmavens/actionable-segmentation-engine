"""
Module: trait_discovery

Purpose: Client-agnostic discovery and scoring of product traits for segmentation.

This module automatically discovers all product-related fields from event data,
classifies their types, and scores them by business impact:
- Revenue impact: Which traits correlate with higher CLV?
- Retention impact: Which traits correlate with lower churn?
- Personalization value: Which traits enable meaningful personalization?

Key Design Principle:
NO hardcoded field names - everything is discovered dynamically from data patterns.
Works with any client's data schema.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from src.data.schemas import CustomerProfile, EventRecord, EventType

if TYPE_CHECKING:
    from src.analysis.trait_explainer import TraitExplanation
    from src.analysis.trait_feedback import FeedbackStore

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TraitMetadata:
    """Metadata about a discovered trait field."""

    field_name: str
    field_path: str  # Full path for nested fields: "extra.color"

    # Type classification
    field_type: str = "unknown"  # categorical, numeric, hierarchical, id, text

    # Source information
    source_events: set[EventType] = field(default_factory=set)
    sample_values: list[Any] = field(default_factory=list)

    # Statistics
    total_occurrences: int = 0
    distinct_values: int = 0
    null_count: int = 0
    customers_with_trait: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of occurrences that are non-null."""
        if self.total_occurrences == 0:
            return 0.0
        return (self.total_occurrences - self.null_count) / self.total_occurrences

    @property
    def uniqueness_ratio(self) -> float:
        """Ratio of distinct values to total non-null occurrences."""
        non_null = self.total_occurrences - self.null_count
        if non_null == 0:
            return 0.0
        return self.distinct_values / non_null


@dataclass
class TraitValueScore:
    """Score of a trait's business value."""

    trait_name: str
    trait_path: str
    trait_type: str  # categorical, numeric, hierarchical

    # Impact scores (0-1, higher = more impactful)
    revenue_impact: float = 0.0
    retention_impact: float = 0.0
    personalization_value: float = 0.0

    # Statistical details
    revenue_f_statistic: float = 0.0
    revenue_p_value: float = 1.0
    retention_chi2_statistic: float = 0.0
    retention_p_value: float = 1.0
    entropy: float = 0.0

    # Trait characteristics
    distinct_values: int = 0
    customer_coverage: float = 0.0  # % of customers with this trait
    concentration: float = 0.0  # Max single value share (low = good diversity)

    # Top values by revenue
    top_revenue_values: list[tuple[str, float]] = field(default_factory=list)
    # Top values by retention
    top_retention_values: list[tuple[str, float]] = field(default_factory=list)

    # Explanation (populated by TraitExplainer)
    explanation: "TraitExplanation | None" = None

    # Feedback-adjusted scoring
    feedback_adjustment: float = 1.0  # Multiplier from user feedback
    adjusted_overall_score: float | None = None  # overall_score * feedback_adjustment

    @property
    def overall_score(self) -> float:
        """Weighted combination of impact scores."""
        return (
            0.4 * self.revenue_impact
            + 0.3 * self.retention_impact
            + 0.3 * self.personalization_value
        )

    @property
    def is_significant(self) -> bool:
        """Whether the trait has statistically significant impact."""
        return self.revenue_p_value < 0.05 or self.retention_p_value < 0.05

    @property
    def recommended_uses(self) -> list[str]:
        """Recommended use cases for this trait."""
        uses = []
        if self.revenue_impact >= 0.5 and self.revenue_p_value < 0.05:
            uses.append("segmentation")
        if self.retention_impact >= 0.5 and self.retention_p_value < 0.05:
            uses.append("retention_targeting")
        if self.personalization_value >= 0.6:
            uses.append("personalization")
        if self.distinct_values >= 3 and self.concentration < 0.7:
            uses.append("content_customization")
        return uses


@dataclass
class TraitDiscoveryResult:
    """Complete trait analysis results."""

    # All traits discovered and scored
    traits: list[TraitValueScore] = field(default_factory=list)

    # Metadata about discovery process
    total_events_scanned: int = 0
    total_customers: int = 0
    total_fields_found: int = 0
    fields_filtered_out: int = 0

    # Filtering reasons
    filtered_as_id: list[str] = field(default_factory=list)
    filtered_as_text: list[str] = field(default_factory=list)
    filtered_low_coverage: list[str] = field(default_factory=list)

    @property
    def top_revenue_traits(self) -> list[TraitValueScore]:
        """Top traits by revenue impact."""
        return sorted(
            [t for t in self.traits if t.revenue_p_value < 0.1],
            key=lambda x: x.revenue_impact,
            reverse=True,
        )[:5]

    @property
    def top_retention_traits(self) -> list[TraitValueScore]:
        """Top traits by retention impact."""
        return sorted(
            [t for t in self.traits if t.retention_p_value < 0.1],
            key=lambda x: x.retention_impact,
            reverse=True,
        )[:5]

    @property
    def top_personalization_traits(self) -> list[TraitValueScore]:
        """Top traits for personalization."""
        return sorted(
            self.traits,
            key=lambda x: x.personalization_value,
            reverse=True,
        )[:5]

    @property
    def recommended_segmentation_traits(self) -> list[str]:
        """Traits recommended for segmentation."""
        return [
            t.trait_name
            for t in self.traits
            if "segmentation" in t.recommended_uses
        ][:5]

    @property
    def recommended_personalization_traits(self) -> list[str]:
        """Traits recommended for personalization."""
        return [
            t.trait_name
            for t in self.traits
            if "personalization" in t.recommended_uses
        ][:5]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "events_scanned": self.total_events_scanned,
            "customers_analyzed": self.total_customers,
            "fields_discovered": self.total_fields_found,
            "traits_scored": len(self.traits),
            "significant_traits": sum(1 for t in self.traits if t.is_significant),
            "top_revenue_trait": self.top_revenue_traits[0].trait_name
            if self.top_revenue_traits
            else None,
            "top_retention_trait": self.top_retention_traits[0].trait_name
            if self.top_retention_traits
            else None,
            "top_personalization_trait": self.top_personalization_traits[0].trait_name
            if self.top_personalization_traits
            else None,
        }


# =============================================================================
# TRAIT VALUE ANALYZER
# =============================================================================


class TraitValueAnalyzer:
    """
    Client-agnostic trait discovery and scoring.

    Discovers ALL fields from event data dynamically, classifies them by type,
    and scores their business value for segmentation, retention, and personalization.

    Example:
        >>> analyzer = TraitValueAnalyzer()
        >>> result = analyzer.analyze(events, profiles)
        >>> for trait in result.top_revenue_traits:
        ...     print(f"{trait.trait_name}: {trait.revenue_impact:.2f}")
    """

    # Fields that are always skipped (system fields, not product traits)
    SYSTEM_FIELD_PATTERNS = [
        r"^__",  # Internal fields like __cplocation__
        r"_timestamp$",
        r"_id$",  # Usually IDs
        r"^session_id$",
        r"^event_id$",
        r"^customer_id$",
        r"^internal_customer_id$",
    ]

    # Event types that contain product information
    PRODUCT_EVENT_TYPES = {
        EventType.PURCHASE,
        EventType.PURCHASE_ITEM,
        EventType.ADD_TO_CART,
        EventType.VIEW_ITEM,
        EventType.CHECKOUT,
    }

    def __init__(
        self,
        *,
        min_coverage: float = 0.05,
        max_cardinality: int = 100,
        min_cardinality: int = 2,
        max_missing_rate: float = 0.95,
        numeric_threshold: float = 0.8,
        id_uniqueness_threshold: float = 0.7,
        min_customers_per_value: int = 5,
        max_sample_values: int = 1000,
    ):
        """
        Initialize TraitValueAnalyzer.

        Args:
            min_coverage: Minimum fraction of events with non-null value (default 5%)
            max_cardinality: Maximum distinct values to be considered a trait (default 100)
            min_cardinality: Minimum distinct values (default 2)
            max_missing_rate: Maximum null rate to include field (default 95%)
            numeric_threshold: Fraction of values that must be numeric (default 80%)
            id_uniqueness_threshold: Uniqueness ratio above which field is an ID (default 70%)
            min_customers_per_value: Minimum customers per trait value for scoring (default 5)
            max_sample_values: Maximum values to sample for type detection (default 1000)
        """
        self.min_coverage = min_coverage
        self.max_cardinality = max_cardinality
        self.min_cardinality = min_cardinality
        self.max_missing_rate = max_missing_rate
        self.numeric_threshold = numeric_threshold
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.min_customers_per_value = min_customers_per_value
        self.max_sample_values = max_sample_values

        # Compiled patterns for system fields
        self._system_patterns = [re.compile(p, re.IGNORECASE) for p in self.SYSTEM_FIELD_PATTERNS]

    def analyze(
        self,
        events: list[EventRecord],
        profiles: list[CustomerProfile],
        *,
        feedback_store: "FeedbackStore | None" = None,
        generate_explanations: bool = True,
    ) -> TraitDiscoveryResult:
        """
        Run complete trait discovery and scoring.

        Args:
            events: List of event records to analyze
            profiles: Customer profiles for business impact scoring
            feedback_store: Optional FeedbackStore for applying learned adjustments
            generate_explanations: Whether to generate human-readable explanations

        Returns:
            TraitDiscoveryResult with scored traits
        """
        logger.info(f"Starting trait discovery on {len(events)} events, {len(profiles)} profiles")

        result = TraitDiscoveryResult(
            total_events_scanned=len(events),
            total_customers=len(profiles),
        )

        # Phase 1: Discover all fields
        field_metadata = self._discover_fields(events)
        result.total_fields_found = len(field_metadata)
        logger.info(f"Discovered {len(field_metadata)} unique fields")

        # Phase 2: Classify field types and filter
        trait_metadata = self._classify_and_filter_fields(field_metadata, result)
        logger.info(f"After filtering: {len(trait_metadata)} usable traits")

        if not trait_metadata:
            logger.warning("No usable traits found after filtering")
            return result

        # Phase 3: Build customer trait profiles
        customer_traits = self._build_customer_trait_profiles(events, trait_metadata)
        logger.info(f"Built trait profiles for {len(customer_traits)} customers")

        # Phase 4: Score each trait
        profile_lookup = {p.internal_customer_id: p for p in profiles}
        scored_traits = []

        for trait_name, metadata in trait_metadata.items():
            trait_values = {
                cid: traits.get(trait_name)
                for cid, traits in customer_traits.items()
                if traits.get(trait_name) is not None
            }

            if len(trait_values) < self.min_customers_per_value * 2:
                continue

            score = self._score_trait(
                trait_name=trait_name,
                metadata=metadata,
                trait_values=trait_values,
                profile_lookup=profile_lookup,
                total_customers=len(profiles),
            )
            scored_traits.append(score)

        # Phase 5: Apply feedback adjustments
        if feedback_store:
            for trait in scored_traits:
                trait.feedback_adjustment = feedback_store.get_adjustment_factor(
                    trait.trait_name
                )
                trait.adjusted_overall_score = (
                    trait.overall_score * trait.feedback_adjustment
                )
            logger.info("Applied feedback adjustments to trait scores")

        # Phase 6: Generate explanations
        if generate_explanations:
            from src.analysis.trait_explainer import TraitExplainer

            explainer = TraitExplainer()
            for trait in scored_traits:
                trait.explanation = explainer.explain(trait)
            logger.info("Generated explanations for traits")

        # Sort by adjusted score if available, otherwise by overall score
        result.traits = sorted(
            scored_traits,
            key=lambda x: x.adjusted_overall_score
            if x.adjusted_overall_score is not None
            else x.overall_score,
            reverse=True,
        )
        logger.info(f"Scored {len(result.traits)} traits")

        return result

    def _discover_fields(
        self,
        events: list[EventRecord],
    ) -> dict[str, TraitMetadata]:
        """
        Phase 1: Scan all events to discover fields.

        Recursively scans properties, extra, and custom_properties.
        """
        field_metadata: dict[str, TraitMetadata] = {}
        value_samples: dict[str, list[Any]] = defaultdict(list)
        customer_values: dict[str, set[str]] = defaultdict(set)

        for event in events:
            # Only scan product-related events
            if event.event_type not in self.PRODUCT_EVENT_TYPES:
                continue

            customer_id = event.internal_customer_id
            props = event.properties

            # Scan main properties
            self._scan_properties(
                props.model_dump(exclude_none=True),
                field_metadata,
                value_samples,
                customer_values,
                customer_id,
                event.event_type,
                prefix="",
            )

            # Scan extra properties
            if props.extra:
                self._scan_properties(
                    props.extra,
                    field_metadata,
                    value_samples,
                    customer_values,
                    customer_id,
                    event.event_type,
                    prefix="extra.",
                )

            # Scan custom properties
            if props.custom_properties:
                self._scan_properties(
                    props.custom_properties,
                    field_metadata,
                    value_samples,
                    customer_values,
                    customer_id,
                    event.event_type,
                    prefix="custom.",
                )

        # Finalize metadata
        for field_path, metadata in field_metadata.items():
            metadata.sample_values = value_samples[field_path][: self.max_sample_values]
            metadata.customers_with_trait = len(customer_values[field_path])
            metadata.distinct_values = len(set(v for v in metadata.sample_values if v is not None))

        return field_metadata

    def _scan_properties(
        self,
        props: dict[str, Any],
        field_metadata: dict[str, TraitMetadata],
        value_samples: dict[str, list[Any]],
        customer_values: dict[str, set[str]],
        customer_id: str,
        event_type: EventType,
        prefix: str,
    ) -> None:
        """Recursively scan a properties dictionary."""
        for key, value in props.items():
            # Skip system fields
            if self._is_system_field(key):
                continue

            field_path = f"{prefix}{key}"

            # Initialize metadata if new field
            if field_path not in field_metadata:
                field_metadata[field_path] = TraitMetadata(
                    field_name=key,
                    field_path=field_path,
                )

            metadata = field_metadata[field_path]
            metadata.total_occurrences += 1
            metadata.source_events.add(event_type)

            if value is None:
                metadata.null_count += 1
            else:
                # Handle different value types
                if isinstance(value, dict):
                    # Recursively scan nested dicts
                    self._scan_properties(
                        value,
                        field_metadata,
                        value_samples,
                        customer_values,
                        customer_id,
                        event_type,
                        prefix=f"{field_path}.",
                    )
                elif isinstance(value, list):
                    # For lists, we don't traverse deeply (could be product arrays)
                    # Just note that it's a list type
                    if len(value_samples[field_path]) < self.max_sample_values:
                        value_samples[field_path].append(f"[list:{len(value)}]")
                else:
                    # Scalar value
                    str_value = str(value)
                    if len(value_samples[field_path]) < self.max_sample_values:
                        value_samples[field_path].append(value)
                    customer_values[field_path].add(customer_id)

    def _is_system_field(self, field_name: str) -> bool:
        """Check if field is a system/internal field to skip."""
        return any(pattern.search(field_name) for pattern in self._system_patterns)

    def _classify_and_filter_fields(
        self,
        field_metadata: dict[str, TraitMetadata],
        result: TraitDiscoveryResult,
    ) -> dict[str, TraitMetadata]:
        """
        Phase 2: Classify field types and filter non-trait fields.
        """
        usable_traits: dict[str, TraitMetadata] = {}

        for field_path, metadata in field_metadata.items():
            # Skip low coverage fields
            if metadata.coverage < self.min_coverage:
                result.filtered_low_coverage.append(field_path)
                continue

            # Skip high missing rate
            if metadata.null_count / max(metadata.total_occurrences, 1) > self.max_missing_rate:
                result.filtered_low_coverage.append(field_path)
                continue

            # Classify type
            metadata.field_type = self._classify_field_type(metadata)

            # Filter by type
            if metadata.field_type == "id":
                result.filtered_as_id.append(field_path)
                continue

            if metadata.field_type == "text":
                result.filtered_as_text.append(field_path)
                continue

            if metadata.field_type == "list":
                # Skip list fields for now
                continue

            # Filter by cardinality
            if metadata.distinct_values < self.min_cardinality:
                continue

            if metadata.distinct_values > self.max_cardinality:
                result.filtered_as_id.append(field_path)
                continue

            usable_traits[field_path] = metadata

        result.fields_filtered_out = len(field_metadata) - len(usable_traits)
        return usable_traits

    def _classify_field_type(self, metadata: TraitMetadata) -> str:
        """
        Classify field type from data patterns.

        Returns: categorical, numeric, hierarchical, id, text, list
        """
        values = [v for v in metadata.sample_values if v is not None]
        if not values:
            return "text"

        # Check for list markers
        if values and isinstance(values[0], str) and values[0].startswith("[list:"):
            return "list"

        # High uniqueness = likely an ID
        if metadata.uniqueness_ratio > self.id_uniqueness_threshold:
            return "id"

        # Check if numeric
        numeric_count = sum(1 for v in values if self._is_numeric(v))
        if numeric_count / len(values) >= self.numeric_threshold:
            return "numeric"

        # Check for hierarchy patterns
        str_values = [str(v) for v in values]
        hierarchy_count = sum(
            1 for v in str_values if any(sep in v for sep in [">", "/", "::", "|", "\\"])
        )
        if hierarchy_count / len(str_values) > 0.3:
            return "hierarchical"

        # Check for long text
        avg_length = sum(len(str(v)) for v in values) / len(values)
        if avg_length > 100:
            return "text"

        # Default: categorical
        return "categorical"

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        if isinstance(value, (int, float, Decimal)):
            return True
        if isinstance(value, str):
            try:
                float(value.replace(",", "").replace(" ", ""))
                return True
            except (ValueError, AttributeError):
                return False
        return False

    def _build_customer_trait_profiles(
        self,
        events: list[EventRecord],
        trait_metadata: dict[str, TraitMetadata],
    ) -> dict[str, dict[str, Any]]:
        """
        Phase 3: Build trait profiles per customer.

        For each customer, determine their dominant value for each trait.
        """
        # Collect all values per customer per trait
        customer_trait_values: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )

        for event in events:
            if event.event_type not in self.PRODUCT_EVENT_TYPES:
                continue

            customer_id = event.internal_customer_id
            props = event.properties

            # Extract values for each trait
            for field_path, metadata in trait_metadata.items():
                value = self._extract_field_value(props, field_path)
                if value is not None:
                    # Weight by event type (purchases count more)
                    weight = 5 if event.event_type == EventType.PURCHASE else 1
                    customer_trait_values[customer_id][field_path][str(value)] += weight

        # Determine dominant value for each trait
        customer_profiles: dict[str, dict[str, Any]] = {}
        for customer_id, trait_counters in customer_trait_values.items():
            customer_profiles[customer_id] = {}
            for field_path, counter in trait_counters.items():
                if counter:
                    # Most common value (weighted)
                    dominant_value = counter.most_common(1)[0][0]
                    customer_profiles[customer_id][field_path] = dominant_value

        return customer_profiles

    def _extract_field_value(
        self,
        props: Any,
        field_path: str,
    ) -> Any:
        """Extract a field value from properties using dot notation path."""
        parts = field_path.split(".")
        current = props

        for part in parts:
            if part == "extra" and hasattr(current, "extra"):
                current = current.extra
            elif part == "custom" and hasattr(current, "custom_properties"):
                current = current.custom_properties
            elif hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _score_trait(
        self,
        trait_name: str,
        metadata: TraitMetadata,
        trait_values: dict[str, str],
        profile_lookup: dict[str, CustomerProfile],
        total_customers: int,
    ) -> TraitValueScore:
        """
        Phase 4: Score a trait's business value.
        """
        score = TraitValueScore(
            trait_name=metadata.field_name,
            trait_path=metadata.field_path,
            trait_type=metadata.field_type,
            distinct_values=metadata.distinct_values,
            customer_coverage=len(trait_values) / max(total_customers, 1),
        )

        # Calculate value distribution
        value_counts = Counter(trait_values.values())
        total_with_trait = sum(value_counts.values())
        if total_with_trait > 0:
            score.concentration = max(value_counts.values()) / total_with_trait

        # Score for revenue impact
        self._score_revenue_impact(score, trait_values, profile_lookup, value_counts)

        # Score for retention impact
        self._score_retention_impact(score, trait_values, profile_lookup, value_counts)

        # Score for personalization value
        self._score_personalization_value(score, value_counts)

        return score

    def _score_revenue_impact(
        self,
        score: TraitValueScore,
        trait_values: dict[str, str],
        profile_lookup: dict[str, CustomerProfile],
        value_counts: Counter,
    ) -> None:
        """
        Score trait's impact on revenue using ANOVA F-test.
        """
        # Group revenue by trait value
        revenue_by_value: dict[str, list[float]] = defaultdict(list)

        for customer_id, trait_value in trait_values.items():
            profile = profile_lookup.get(customer_id)
            if profile:
                revenue = float(profile.total_revenue)
                revenue_by_value[trait_value].append(revenue)

        # Need at least 2 groups with sufficient samples
        valid_groups = [
            (val, revs)
            for val, revs in revenue_by_value.items()
            if len(revs) >= self.min_customers_per_value
        ]

        if len(valid_groups) < 2:
            return

        # Run ANOVA
        group_revenues = [revs for _, revs in valid_groups]
        try:
            f_stat, p_value = stats.f_oneway(*group_revenues)
            if not np.isnan(f_stat):
                score.revenue_f_statistic = float(f_stat)
                score.revenue_p_value = float(p_value)

                # Convert F-statistic to 0-1 impact score
                # F > 10 is strong, F > 5 is moderate
                score.revenue_impact = min(1.0, f_stat / 20.0)
        except Exception:
            pass

        # Find top revenue values
        value_avg_revenue = [
            (val, sum(revs) / len(revs))
            for val, revs in valid_groups
        ]
        value_avg_revenue.sort(key=lambda x: x[1], reverse=True)
        score.top_revenue_values = value_avg_revenue[:5]

    def _score_retention_impact(
        self,
        score: TraitValueScore,
        trait_values: dict[str, str],
        profile_lookup: dict[str, CustomerProfile],
        value_counts: Counter,
    ) -> None:
        """
        Score trait's impact on retention using Chi-square test.
        """
        # Create contingency table: trait_value x churn_bucket
        # Churn buckets: low (<0.3), medium (0.3-0.7), high (>0.7)

        contingency: dict[str, dict[str, int]] = defaultdict(lambda: {"low": 0, "medium": 0, "high": 0})

        for customer_id, trait_value in trait_values.items():
            profile = profile_lookup.get(customer_id)
            if profile:
                churn = profile.churn_risk_score
                if churn < 0.3:
                    bucket = "low"
                elif churn < 0.7:
                    bucket = "medium"
                else:
                    bucket = "high"
                contingency[trait_value][bucket] += 1

        # Filter to values with enough samples
        valid_values = [
            val for val, buckets in contingency.items()
            if sum(buckets.values()) >= self.min_customers_per_value
        ]

        if len(valid_values) < 2:
            return

        # Build contingency matrix
        observed = []
        for val in valid_values:
            observed.append([
                contingency[val]["low"],
                contingency[val]["medium"],
                contingency[val]["high"],
            ])

        observed = np.array(observed)

        # Check for valid contingency table
        if observed.sum() == 0 or observed.shape[0] < 2:
            return

        # Run Chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            if not np.isnan(chi2):
                score.retention_chi2_statistic = float(chi2)
                score.retention_p_value = float(p_value)

                # Convert chi2 to 0-1 impact score
                # Normalize by degrees of freedom
                normalized_chi2 = chi2 / max(dof, 1)
                score.retention_impact = min(1.0, normalized_chi2 / 10.0)
        except Exception:
            pass

        # Find top retention values (lowest churn)
        value_churn_rate = []
        for val in valid_values:
            total = sum(contingency[val].values())
            high_churn = contingency[val]["high"]
            if total > 0:
                value_churn_rate.append((val, 1 - high_churn / total))  # Retention rate

        value_churn_rate.sort(key=lambda x: x[1], reverse=True)
        score.top_retention_values = value_churn_rate[:5]

    def _score_personalization_value(
        self,
        score: TraitValueScore,
        value_counts: Counter,
    ) -> None:
        """
        Score trait's personalization value using entropy.

        Good personalization traits have:
        - Moderate cardinality (not too few, not too many values)
        - Even distribution (high entropy)
        - Low concentration
        """
        total = sum(value_counts.values())
        if total == 0:
            return

        # Calculate entropy
        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        score.entropy = float(entropy)

        # Normalize entropy by max possible entropy
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Cardinality penalty: too few or too many values reduce personalization value
        cardinality = len(value_counts)
        if cardinality < 3:
            cardinality_factor = 0.3
        elif cardinality > 50:
            cardinality_factor = 0.7
        else:
            cardinality_factor = 1.0

        # Concentration penalty: high concentration = less useful for personalization
        concentration_factor = 1.0 - score.concentration * 0.5

        score.personalization_value = (
            normalized_entropy * cardinality_factor * concentration_factor
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def discover_traits(
    events: list[EventRecord],
    profiles: list[CustomerProfile],
    *,
    min_coverage: float = 0.05,
    max_cardinality: int = 100,
) -> TraitDiscoveryResult:
    """
    Convenience function to run trait discovery.

    Args:
        events: Event records to analyze
        profiles: Customer profiles for scoring
        min_coverage: Minimum coverage threshold
        max_cardinality: Maximum distinct values

    Returns:
        TraitDiscoveryResult with scored traits
    """
    analyzer = TraitValueAnalyzer(
        min_coverage=min_coverage,
        max_cardinality=max_cardinality,
    )
    return analyzer.analyze(events, profiles)


def format_trait_report(result: TraitDiscoveryResult) -> str:
    """
    Generate a human-readable trait discovery report.

    Args:
        result: TraitDiscoveryResult

    Returns:
        Formatted text report
    """
    lines = []

    lines.append("=" * 70)
    lines.append("TRAIT VALUE DISCOVERY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    summary = result.get_summary()
    lines.append("DISCOVERY SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Events Scanned: {summary['events_scanned']:,}")
    lines.append(f"Customers Analyzed: {summary['customers_analyzed']:,}")
    lines.append(f"Fields Discovered: {summary['fields_discovered']}")
    lines.append(f"Usable Traits: {summary['traits_scored']}")
    lines.append(f"Statistically Significant: {summary['significant_traits']}")
    lines.append("")

    # Filtering summary
    if result.filtered_as_id or result.filtered_as_text or result.filtered_low_coverage:
        lines.append("FILTERED FIELDS")
        lines.append("-" * 40)
        if result.filtered_as_id:
            lines.append(f"  ID fields (high uniqueness): {len(result.filtered_as_id)}")
        if result.filtered_as_text:
            lines.append(f"  Text fields (too long): {len(result.filtered_as_text)}")
        if result.filtered_low_coverage:
            lines.append(f"  Low coverage: {len(result.filtered_low_coverage)}")
        lines.append("")

    # Top revenue traits
    lines.append("TOP TRAITS BY REVENUE IMPACT")
    lines.append("-" * 40)
    for i, trait in enumerate(result.top_revenue_traits, 1):
        sig = "***" if trait.revenue_p_value < 0.001 else "**" if trait.revenue_p_value < 0.01 else "*" if trait.revenue_p_value < 0.05 else ""
        # Show adjusted score if available
        score_str = f"Score: {trait.overall_score:.2f}"
        if trait.adjusted_overall_score is not None:
            score_str = f"Score: {trait.overall_score:.2f} (adj: {trait.adjusted_overall_score:.2f})"
        lines.append(
            f"{i}. {trait.trait_name} | Impact: {trait.revenue_impact:.2f} | "
            f"F={trait.revenue_f_statistic:.1f} {sig}"
        )
        if trait.top_revenue_values:
            top_val, top_rev = trait.top_revenue_values[0]
            bottom_val, bottom_rev = trait.top_revenue_values[-1]
            lines.append(f"   Best: '{top_val}' (${top_rev:.0f} avg) vs '{bottom_val}' (${bottom_rev:.0f} avg)")
        # Add explanation if available
        if trait.explanation:
            if trait.explanation.revenue_explanation:
                lines.append(f"   WHY: {trait.explanation.revenue_explanation}")
            if trait.explanation.caveats:
                lines.append(f"   CAVEAT: {trait.explanation.caveats[0]}")
            if trait.feedback_adjustment != 1.0:
                pct_change = (1 - trait.feedback_adjustment) * 100
                lines.append(f"   FEEDBACK: Score adjusted by {pct_change:+.0f}% based on user feedback")
    lines.append("")

    # Top retention traits
    lines.append("TOP TRAITS BY RETENTION IMPACT")
    lines.append("-" * 40)
    for i, trait in enumerate(result.top_retention_traits, 1):
        sig = "***" if trait.retention_p_value < 0.001 else "**" if trait.retention_p_value < 0.01 else "*" if trait.retention_p_value < 0.05 else ""
        lines.append(
            f"{i}. {trait.trait_name} | Impact: {trait.retention_impact:.2f} | "
            f"χ²={trait.retention_chi2_statistic:.1f} {sig}"
        )
        if trait.top_retention_values:
            top_val, top_ret = trait.top_retention_values[0]
            lines.append(f"   Best retention: '{top_val}' ({top_ret:.0%} retention rate)")
        # Add explanation if available
        if trait.explanation and trait.explanation.retention_explanation:
            lines.append(f"   WHY: {trait.explanation.retention_explanation}")
    lines.append("")

    # Top personalization traits
    lines.append("TOP TRAITS FOR PERSONALIZATION")
    lines.append("-" * 40)
    for i, trait in enumerate(result.top_personalization_traits, 1):
        lines.append(
            f"{i}. {trait.trait_name} | Value: {trait.personalization_value:.2f} | "
            f"Entropy: {trait.entropy:.2f} | Values: {trait.distinct_values}"
        )
        # Add explanation if available
        if trait.explanation and trait.explanation.personalization_explanation:
            lines.append(f"   WHY: {trait.explanation.personalization_explanation}")
    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    if result.recommended_segmentation_traits:
        lines.append(f"Segment on: {', '.join(result.recommended_segmentation_traits)}")
    if result.recommended_personalization_traits:
        lines.append(f"Personalize on: {', '.join(result.recommended_personalization_traits)}")
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
