"""
Module: integrated_analysis

Purpose: Unified segment analysis combining all pipeline components.

Produces actionable segments with:
- Validation status (size, robustness)
- Actionability evaluation
- Business explanation
- Whitespace opportunities per segment
- Final "usable segment" decision

Key Concept:
A segment is "usable" when it passes all quality gates AND has actionable opportunities.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.data.schemas import (
    ActionabilityEvaluation,
    CustomerProfile,
    RobustnessScore,
    Segment,
    SegmentExplanation,
    SegmentViability,
)
from src.segmentation.segment_validator import ValidationResult
from src.segmentation.whitespace import (
    CategoryWhitespace,
    WhitespaceAnalyzer,
    WhitespaceAnalysisResult,
)


# =============================================================================
# INTEGRATED DATA STRUCTURES
# =============================================================================


@dataclass
class SegmentWhitespace:
    """Whitespace opportunities specific to a segment's customers."""

    segment_id: str

    # Category opportunities for this segment's non-buyers
    category_opportunities: list[CategoryWhitespace] = field(default_factory=list)

    # Cross-sell opportunities (segment customers who buy X but not Y)
    cross_sell_opportunities: list[CategoryWhitespace] = field(default_factory=list)

    # Summary metrics
    total_opportunity_value: Decimal = Decimal("0")
    total_lookalike_count: int = 0
    top_category: str | None = None


@dataclass
class UsableSegment:
    """
    A segment that has passed all quality gates and is ready for activation.

    This is the final output: everything you need to act on a segment.
    """

    # Core segment
    segment: Segment

    # Quality assessments
    validation: ValidationResult
    robustness: RobustnessScore | None
    viability: SegmentViability | None

    # Actionability
    actionability: ActionabilityEvaluation
    explanation: SegmentExplanation | None

    # Whitespace opportunities
    whitespace: SegmentWhitespace | None

    # Final status
    is_usable: bool = False
    usability_reasons: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    # Priority score (0-100) based on opportunity value and quality
    priority_score: float = 0.0


@dataclass
class IntegratedAnalysisResult:
    """Complete integrated analysis output."""

    # All usable segments (passed all gates)
    usable_segments: list[UsableSegment] = field(default_factory=list)

    # Segments that didn't pass (with reasons)
    rejected_segments: list[UsableSegment] = field(default_factory=list)

    # Global whitespace analysis
    global_whitespace: WhitespaceAnalysisResult | None = None

    # Summary statistics
    total_segments: int = 0
    total_usable: int = 0
    total_customers: int = 0
    customers_in_usable_segments: int = 0

    # Opportunity metrics
    total_segment_clv: Decimal = Decimal("0")
    total_whitespace_opportunity: Decimal = Decimal("0")

    def get_top_usable_segments(self, n: int = 5) -> list[UsableSegment]:
        """Get top N usable segments by priority score."""
        return sorted(
            self.usable_segments,
            key=lambda s: s.priority_score,
            reverse=True
        )[:n]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_segments": self.total_segments,
            "usable_segments": self.total_usable,
            "rejected_segments": len(self.rejected_segments),
            "usability_rate": self.total_usable / self.total_segments if self.total_segments > 0 else 0,
            "total_customers": self.total_customers,
            "customers_in_usable_segments": self.customers_in_usable_segments,
            "customer_coverage": self.customers_in_usable_segments / self.total_customers if self.total_customers > 0 else 0,
            "total_segment_clv": float(self.total_segment_clv),
            "total_whitespace_opportunity": float(self.total_whitespace_opportunity),
        }


# =============================================================================
# INTEGRATED ANALYZER
# =============================================================================


class IntegratedAnalyzer:
    """
    Combines all analysis components into a unified segment assessment.

    Takes pipeline outputs and produces actionable segment recommendations
    with whitespace opportunities.
    """

    def __init__(
        self,
        *,
        require_valid: bool = True,
        require_actionable: bool = True,
        require_robustness: bool = True,
        min_robustness: float = 0.3,
        include_whitespace: bool = True,
        whitespace_similarity_threshold: float = 0.5,
    ):
        """
        Initialize IntegratedAnalyzer.

        Args:
            require_valid: Segment must pass validation
            require_actionable: Segment must be actionable
            require_robustness: Segment must meet robustness threshold
            min_robustness: Minimum robustness score (if require_robustness=True)
            include_whitespace: Analyze whitespace opportunities
            whitespace_similarity_threshold: Threshold for lookalike similarity
        """
        self.require_valid = require_valid
        self.require_actionable = require_actionable
        self.require_robustness = require_robustness
        self.min_robustness = min_robustness
        self.include_whitespace = include_whitespace
        self.whitespace_similarity_threshold = whitespace_similarity_threshold

    def analyze(
        self,
        profiles: list[CustomerProfile],
        segments: list[Segment],
        *,
        validation_results: dict[str, ValidationResult] | None = None,
        robustness_scores: dict[str, RobustnessScore] | None = None,
        viabilities: dict[str, SegmentViability] | None = None,
        actionability_evaluations: dict[str, ActionabilityEvaluation] | None = None,
        explanations: dict[str, SegmentExplanation] | None = None,
    ) -> IntegratedAnalysisResult:
        """
        Run integrated analysis on segments.

        Args:
            profiles: All customer profiles
            segments: Segments from clustering
            validation_results: Validation results per segment
            robustness_scores: Robustness scores per segment
            viabilities: Viability assessments per segment
            actionability_evaluations: Actionability evaluations per segment
            explanations: Business explanations per segment

        Returns:
            IntegratedAnalysisResult with usable segments and opportunities
        """
        validation_results = validation_results or {}
        robustness_scores = robustness_scores or {}
        viabilities = viabilities or {}
        actionability_evaluations = actionability_evaluations or {}
        explanations = explanations or {}

        # Build profile lookup by segment
        profile_lookup = {p.internal_customer_id: p for p in profiles}
        segment_profiles: dict[str, list[CustomerProfile]] = {}

        for segment in segments:
            segment_profiles[segment.segment_id] = [
                profile_lookup[m.internal_customer_id]
                for m in segment.members
                if m.internal_customer_id in profile_lookup
            ]

        # Run global whitespace analysis
        global_whitespace: WhitespaceAnalysisResult | None = None
        if self.include_whitespace and profiles:
            whitespace_analyzer = WhitespaceAnalyzer(
                similarity_threshold=self.whitespace_similarity_threshold
            )
            global_whitespace = whitespace_analyzer.analyze(profiles)

        # Analyze each segment
        usable_segments: list[UsableSegment] = []
        rejected_segments: list[UsableSegment] = []

        for segment in segments:
            usable_segment = self._analyze_segment(
                segment=segment,
                segment_profiles=segment_profiles.get(segment.segment_id, []),
                validation=validation_results.get(segment.segment_id),
                robustness=robustness_scores.get(segment.segment_id),
                viability=viabilities.get(segment.segment_id),
                actionability=actionability_evaluations.get(segment.segment_id),
                explanation=explanations.get(segment.segment_id),
                global_whitespace=global_whitespace,
                profile_lookup=profile_lookup,
            )

            if usable_segment.is_usable:
                usable_segments.append(usable_segment)
            else:
                rejected_segments.append(usable_segment)

        # Calculate summary stats
        total_segment_clv = sum(s.segment.total_clv for s in usable_segments)
        total_whitespace = sum(
            s.whitespace.total_opportunity_value
            for s in usable_segments
            if s.whitespace
        )
        customers_in_usable = sum(s.segment.size for s in usable_segments)

        return IntegratedAnalysisResult(
            usable_segments=usable_segments,
            rejected_segments=rejected_segments,
            global_whitespace=global_whitespace,
            total_segments=len(segments),
            total_usable=len(usable_segments),
            total_customers=len(profiles),
            customers_in_usable_segments=customers_in_usable,
            total_segment_clv=total_segment_clv,
            total_whitespace_opportunity=total_whitespace,
        )

    def _analyze_segment(
        self,
        segment: Segment,
        segment_profiles: list[CustomerProfile],
        validation: ValidationResult | None,
        robustness: RobustnessScore | None,
        viability: SegmentViability | None,
        actionability: ActionabilityEvaluation | None,
        explanation: SegmentExplanation | None,
        global_whitespace: WhitespaceAnalysisResult | None,
        profile_lookup: dict[str, CustomerProfile],
    ) -> UsableSegment:
        """Analyze a single segment for usability."""

        usability_reasons: list[str] = []
        recommended_actions: list[str] = []
        is_usable = True

        # Check validation (but filter out actionability-related rejections since we check that separately)
        if self.require_valid:
            if validation is None:
                is_usable = False
                usability_reasons.append("Failed validation")
            elif not validation.is_valid:
                # Filter out "No actionability dimensions" since we check actionability separately
                relevant_rejections = [
                    r for r in validation.rejection_reasons
                    if "actionability dimensions" not in r.lower()
                ]
                if relevant_rejections:
                    is_usable = False
                    usability_reasons.extend(relevant_rejections)

        # Check actionability
        if self.require_actionable:
            if actionability is None or not actionability.is_actionable:
                is_usable = False
                usability_reasons.append("Not actionable")
            else:
                if actionability.recommended_action:
                    recommended_actions.append(actionability.recommended_action)

        # Check robustness
        if self.require_robustness:
            if robustness is None:
                is_usable = False
                usability_reasons.append("Missing robustness assessment")
            elif robustness.overall_robustness < self.min_robustness:
                is_usable = False
                usability_reasons.append(
                    f"Low robustness ({robustness.overall_robustness:.2f} < {self.min_robustness})"
                )

        # Analyze whitespace for this segment's customers
        segment_whitespace: SegmentWhitespace | None = None
        if self.include_whitespace and global_whitespace and segment_profiles:
            segment_whitespace = self._extract_segment_whitespace(
                segment_id=segment.segment_id,
                segment_profiles=segment_profiles,
                global_whitespace=global_whitespace,
            )

            # Add whitespace-based recommendations
            if segment_whitespace and segment_whitespace.top_category:
                recommended_actions.append(
                    f"Cross-sell opportunity: Target {segment_whitespace.top_category} "
                    f"(${float(segment_whitespace.total_opportunity_value):,.0f} potential)"
                )

        # Calculate priority score
        priority_score = self._calculate_priority(
            segment=segment,
            robustness=robustness,
            viability=viability,
            whitespace=segment_whitespace,
        )

        return UsableSegment(
            segment=segment,
            validation=validation or ValidationResult(is_valid=False, rejection_reasons=["Not validated"]),
            robustness=robustness,
            viability=viability,
            actionability=actionability or ActionabilityEvaluation(
                segment_id=segment.segment_id,
                is_actionable=False,
                reasoning="Not evaluated",
            ),
            explanation=explanation,
            whitespace=segment_whitespace,
            is_usable=is_usable,
            usability_reasons=usability_reasons if not is_usable else ["Passed all quality gates"],
            recommended_actions=recommended_actions,
            priority_score=priority_score,
        )

    def _extract_segment_whitespace(
        self,
        segment_id: str,
        segment_profiles: list[CustomerProfile],
        global_whitespace: WhitespaceAnalysisResult,
    ) -> SegmentWhitespace:
        """Extract whitespace opportunities specific to segment customers."""

        segment_customer_ids = {p.internal_customer_id for p in segment_profiles}

        category_opportunities: list[CategoryWhitespace] = []
        total_value = Decimal("0")
        total_lookalikes = 0

        # Find opportunities where segment customers are lookalikes
        for category, cat_whitespace in global_whitespace.category_whitespaces.items():
            # Count segment customers in this category's lookalikes
            segment_lookalikes = [
                c for c in cat_whitespace.top_candidates
                if c.customer_id in segment_customer_ids
            ]

            if segment_lookalikes:
                # Calculate segment-specific opportunity
                avg_similarity = sum(c.similarity_score for c in segment_lookalikes) / len(segment_lookalikes)
                opportunity_value = cat_whitespace.buyer_avg_clv * len(segment_lookalikes)

                segment_cat_whitespace = CategoryWhitespace(
                    category=category,
                    n_buyers=cat_whitespace.n_buyers,
                    buyer_avg_clv=cat_whitespace.buyer_avg_clv,
                    n_candidates=len(segment_lookalikes),
                    n_lookalikes=len(segment_lookalikes),
                    total_opportunity_value=opportunity_value,
                    avg_similarity_score=avg_similarity,
                    top_candidates=segment_lookalikes[:10],
                )

                category_opportunities.append(segment_cat_whitespace)
                total_value += opportunity_value
                total_lookalikes += len(segment_lookalikes)

        # Sort by opportunity value
        category_opportunities.sort(
            key=lambda x: x.total_opportunity_value,
            reverse=True
        )

        top_category = category_opportunities[0].category if category_opportunities else None

        return SegmentWhitespace(
            segment_id=segment_id,
            category_opportunities=category_opportunities,
            total_opportunity_value=total_value,
            total_lookalike_count=total_lookalikes,
            top_category=top_category,
        )

    def _calculate_priority(
        self,
        segment: Segment,
        robustness: RobustnessScore | None,
        viability: SegmentViability | None,
        whitespace: SegmentWhitespace | None,
    ) -> float:
        """
        Calculate priority score (0-100) for segment activation.

        Higher = more valuable and easier to activate.
        """
        score = 0.0

        # Size contribution (0-20 points)
        # Larger segments = more impact
        if segment.size >= 500:
            score += 20
        elif segment.size >= 100:
            score += 15
        elif segment.size >= 50:
            score += 10
        else:
            score += 5

        # CLV contribution (0-25 points)
        avg_clv = float(segment.avg_clv)
        if avg_clv >= 10000:
            score += 25
        elif avg_clv >= 5000:
            score += 20
        elif avg_clv >= 1000:
            score += 15
        else:
            score += 5

        # Robustness contribution (0-20 points)
        if robustness:
            score += robustness.overall_robustness * 20

        # Viability contribution (0-15 points)
        if viability:
            roi_factor = min(viability.expected_roi / 5.0, 1.0)  # Cap at 5x ROI
            score += roi_factor * 15

        # Whitespace contribution (0-20 points)
        if whitespace and whitespace.total_opportunity_value > 0:
            opp_value = float(whitespace.total_opportunity_value)
            if opp_value >= 100000:
                score += 20
            elif opp_value >= 50000:
                score += 15
            elif opp_value >= 10000:
                score += 10
            else:
                score += 5

        return min(100.0, score)


# =============================================================================
# REPORT GENERATION
# =============================================================================


def format_integrated_report(result: IntegratedAnalysisResult) -> str:
    """
    Generate a comprehensive text report of integrated analysis.

    Args:
        result: IntegratedAnalysisResult

    Returns:
        Formatted text report
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 70)
    lines.append("INTEGRATED SEGMENT ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Executive Summary
    summary = result.get_summary()
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 50)
    lines.append(f"Total Segments Analyzed: {summary['total_segments']}")
    lines.append(f"Usable Segments: {summary['usable_segments']} ({summary['usability_rate']:.1%})")
    lines.append(f"Rejected Segments: {summary['rejected_segments']}")
    lines.append("")
    lines.append(f"Total Customers: {summary['total_customers']:,}")
    lines.append(f"Customers in Usable Segments: {summary['customers_in_usable_segments']:,} ({summary['customer_coverage']:.1%})")
    lines.append("")
    lines.append(f"Total CLV (Usable Segments): ${summary['total_segment_clv']:,.2f}")
    lines.append(f"Total Whitespace Opportunity: ${summary['total_whitespace_opportunity']:,.2f}")
    lines.append("")

    # Usable Segments Detail
    lines.append("=" * 70)
    lines.append("USABLE SEGMENTS (Ready for Activation)")
    lines.append("=" * 70)

    for i, usable in enumerate(result.get_top_usable_segments(10), 1):
        lines.append("")
        lines.append(f"#{i} [{usable.segment.segment_id}] {usable.segment.name}")
        lines.append(f"    Priority Score: {usable.priority_score:.1f}/100")
        lines.append(f"    Size: {usable.segment.size:,} customers")
        lines.append(f"    Total CLV: ${float(usable.segment.total_clv):,.2f}")
        lines.append(f"    Avg CLV: ${float(usable.segment.avg_clv):,.2f}")

        if usable.robustness:
            lines.append(f"    Robustness: {usable.robustness.robustness_tier.value} ({usable.robustness.overall_robustness:.2f})")

        # Defining traits
        if usable.segment.defining_traits:
            lines.append(f"    Traits: {', '.join(usable.segment.defining_traits[:3])}")

        # Actionability
        if usable.actionability.actionability_dimensions:
            dims = [d.value for d in usable.actionability.actionability_dimensions]
            lines.append(f"    Actionable Dimensions: {', '.join(dims)}")

        # Whitespace opportunities
        if usable.whitespace and usable.whitespace.category_opportunities:
            lines.append(f"    Whitespace Opportunities:")
            for cat_opp in usable.whitespace.category_opportunities[:3]:
                lines.append(
                    f"      - {cat_opp.category}: {cat_opp.n_lookalikes} lookalikes, "
                    f"${float(cat_opp.total_opportunity_value):,.0f} potential"
                )

        # Recommended actions
        if usable.recommended_actions:
            lines.append(f"    Recommended Actions:")
            for action in usable.recommended_actions[:3]:
                lines.append(f"      -> {action}")

    # Rejected Segments Summary
    if result.rejected_segments:
        lines.append("")
        lines.append("=" * 70)
        lines.append("REJECTED SEGMENTS (Did not pass quality gates)")
        lines.append("=" * 70)

        for rejected in result.rejected_segments[:5]:
            lines.append("")
            lines.append(f"[{rejected.segment.segment_id}] {rejected.segment.name}")
            lines.append(f"    Size: {rejected.segment.size:,} customers")
            lines.append(f"    Rejection Reasons:")
            for reason in rejected.usability_reasons:
                lines.append(f"      - {reason}")

    # Global Whitespace Summary
    if result.global_whitespace:
        lines.append("")
        lines.append("=" * 70)
        lines.append("GLOBAL WHITESPACE OPPORTUNITIES")
        lines.append("=" * 70)
        lines.append(f"Total Opportunities: {result.global_whitespace.total_opportunities:,}")
        lines.append(f"Total Opportunity Value: ${float(result.global_whitespace.total_opportunity_value):,.2f}")
        lines.append("")
        lines.append("Top Categories:")

        for cat in result.global_whitespace.get_top_categories(5):
            lines.append(
                f"  - {cat.category}: {cat.n_lookalikes} lookalikes "
                f"(${float(cat.total_opportunity_value):,.0f})"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def export_integrated_analysis(
    result: IntegratedAnalysisResult,
) -> dict[str, Any]:
    """
    Export integrated analysis to dictionary for JSON serialization.

    Args:
        result: IntegratedAnalysisResult

    Returns:
        Dictionary suitable for JSON export
    """
    def segment_to_dict(usable: UsableSegment) -> dict[str, Any]:
        seg_dict: dict[str, Any] = {
            "segment_id": usable.segment.segment_id,
            "name": usable.segment.name,
            "size": usable.segment.size,
            "total_clv": float(usable.segment.total_clv),
            "avg_clv": float(usable.segment.avg_clv),
            "is_usable": usable.is_usable,
            "priority_score": usable.priority_score,
            "usability_reasons": usable.usability_reasons,
            "recommended_actions": usable.recommended_actions,
            "defining_traits": usable.segment.defining_traits,
        }

        if usable.robustness:
            seg_dict["robustness"] = {
                "overall": usable.robustness.overall_robustness,
                "tier": usable.robustness.robustness_tier.value,
            }

        if usable.actionability:
            seg_dict["actionability"] = {
                "is_actionable": usable.actionability.is_actionable,
                "dimensions": [d.value for d in usable.actionability.actionability_dimensions],
                "recommended_action": usable.actionability.recommended_action,
            }

        if usable.whitespace:
            seg_dict["whitespace"] = {
                "total_opportunity_value": float(usable.whitespace.total_opportunity_value),
                "total_lookalike_count": usable.whitespace.total_lookalike_count,
                "top_category": usable.whitespace.top_category,
                "category_opportunities": [
                    {
                        "category": cat.category,
                        "n_lookalikes": cat.n_lookalikes,
                        "opportunity_value": float(cat.total_opportunity_value),
                        "avg_similarity": cat.avg_similarity_score,
                    }
                    for cat in usable.whitespace.category_opportunities[:5]
                ],
            }

        return seg_dict

    return {
        "summary": result.get_summary(),
        "usable_segments": [segment_to_dict(s) for s in result.usable_segments],
        "rejected_segments": [segment_to_dict(s) for s in result.rejected_segments],
        "global_whitespace": {
            "total_opportunities": result.global_whitespace.total_opportunities,
            "total_opportunity_value": float(result.global_whitespace.total_opportunity_value),
            "top_categories": [
                {
                    "category": cat.category,
                    "n_lookalikes": cat.n_lookalikes,
                    "opportunity_value": float(cat.total_opportunity_value),
                }
                for cat in result.global_whitespace.get_top_categories(10)
            ],
        } if result.global_whitespace else None,
    }
