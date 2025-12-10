"""
Module: segment_reporter

Purpose: Generate human-readable segment reports.

Key Functions:
- generate_report: Create comprehensive segment report
- export_to_json: Export report as JSON
- SegmentReport: Container for full report

Architecture Notes:
- Combines segment data, insights, and recommendations
- Includes robustness details and confidence levels
- Produces JSON output for downstream systems
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentExplanation,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import ReportGenerationError


# =============================================================================
# REPORT DATA CLASSES
# =============================================================================


@dataclass
class SegmentSummary:
    """Summary of a single segment."""

    segment_id: str
    name: str
    size: int
    total_clv: float
    avg_clv: float
    avg_order_value: float
    defining_traits: list[str]
    actionability_dimensions: list[str]
    strategic_goals: list[str]


@dataclass
class RobustnessReport:
    """Robustness information for a segment."""

    segment_id: str
    overall_robustness: float
    robustness_tier: str
    feature_stability: float
    time_window_consistency: float
    is_production_ready: bool
    sensitive_features: list[str]


@dataclass
class ActionabilityReport:
    """Actionability evaluation for a segment."""

    segment_id: str
    is_actionable: bool
    reasoning: str
    recommended_action: str | None
    confidence_level: str
    dimensions: list[str]


@dataclass
class ExplanationReport:
    """Business explanation for a segment."""

    segment_id: str
    executive_summary: str
    key_characteristics: list[str]
    recommended_campaign: str
    business_hypothesis: str
    expected_roi: str
    confidence_level: str
    confidence_justification: str


@dataclass
class ViabilityReport:
    """Viability assessment for a segment."""

    segment_id: str
    marketing_targetability: float
    sales_prioritization: float
    personalization_opportunity: float
    timing_optimization: float
    cost_to_exploit: float
    expected_roi: float
    revenue_impact: str
    retention_impact: str
    satisfaction_impact: str
    is_approved: bool


@dataclass
class SegmentReport:
    """Complete report for a segment."""

    segment_id: str
    name: str
    summary: SegmentSummary
    robustness: RobustnessReport | None = None
    actionability: ActionabilityReport | None = None
    explanation: ExplanationReport | None = None
    viability: ViabilityReport | None = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SegmentationReport:
    """Complete report for a segmentation run."""

    report_id: str
    title: str
    generated_at: str
    total_segments: int
    total_customers: int
    actionable_segments: int
    segments: list[SegmentReport]
    summary_stats: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================


def segment_to_summary(segment: Segment) -> SegmentSummary:
    """
    Convert Segment to SegmentSummary.

    Args:
        segment: Segment to convert

    Returns:
        SegmentSummary
    """
    return SegmentSummary(
        segment_id=segment.segment_id,
        name=segment.name,
        size=segment.size,
        total_clv=float(segment.total_clv),
        avg_clv=float(segment.avg_clv),
        avg_order_value=float(segment.avg_order_value),
        defining_traits=segment.defining_traits.copy(),
        actionability_dimensions=[
            dim.value for dim in segment.actionability_dimensions
        ],
        strategic_goals=[goal.value for goal in segment.strategic_goals],
    )


def robustness_to_report(robustness: RobustnessScore) -> RobustnessReport:
    """
    Convert RobustnessScore to RobustnessReport.

    Args:
        robustness: RobustnessScore to convert

    Returns:
        RobustnessReport
    """
    return RobustnessReport(
        segment_id=robustness.segment_id,
        overall_robustness=robustness.overall_robustness,
        robustness_tier=robustness.robustness_tier.value,
        feature_stability=robustness.feature_stability,
        time_window_consistency=robustness.time_window_consistency,
        is_production_ready=robustness.is_production_ready,
        sensitive_features=[],  # Not in current schema, empty for compatibility
    )


def actionability_to_report(evaluation: ActionabilityEvaluation) -> ActionabilityReport:
    """
    Convert ActionabilityEvaluation to ActionabilityReport.

    Args:
        evaluation: ActionabilityEvaluation to convert

    Returns:
        ActionabilityReport
    """
    return ActionabilityReport(
        segment_id=evaluation.segment_id,
        is_actionable=evaluation.is_actionable,
        reasoning=evaluation.reasoning,
        recommended_action=evaluation.recommended_action,
        confidence_level=evaluation.confidence_level.value,
        dimensions=[dim.value for dim in evaluation.actionability_dimensions],
    )


def explanation_to_report(explanation: SegmentExplanation) -> ExplanationReport:
    """
    Convert SegmentExplanation to ExplanationReport.

    Args:
        explanation: SegmentExplanation to convert

    Returns:
        ExplanationReport
    """
    return ExplanationReport(
        segment_id=explanation.segment_id,
        executive_summary=explanation.executive_summary,
        key_characteristics=explanation.key_characteristics.copy(),
        recommended_campaign=explanation.recommended_campaign,
        business_hypothesis=explanation.business_hypothesis,
        expected_roi=explanation.expected_roi,
        confidence_level=explanation.confidence_level.value,
        confidence_justification=explanation.confidence_justification,
    )


def viability_to_report(viability: SegmentViability) -> ViabilityReport:
    """
    Convert SegmentViability to ViabilityReport.

    Args:
        viability: SegmentViability to convert

    Returns:
        ViabilityReport
    """
    return ViabilityReport(
        segment_id=viability.segment_id,
        marketing_targetability=viability.marketing_targetability,
        sales_prioritization=viability.sales_prioritization,
        personalization_opportunity=viability.personalization_opportunity,
        timing_optimization=viability.timing_optimization,
        cost_to_exploit=float(viability.cost_to_exploit),
        expected_roi=viability.expected_roi,
        revenue_impact=viability.revenue_impact,
        retention_impact=viability.retention_impact,
        satisfaction_impact=viability.satisfaction_impact,
        is_approved=viability.is_approved,
    )


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_segment_report(
    segment: Segment,
    *,
    robustness: RobustnessScore | None = None,
    actionability: ActionabilityEvaluation | None = None,
    explanation: SegmentExplanation | None = None,
    viability: SegmentViability | None = None,
) -> SegmentReport:
    """
    Generate a complete report for a single segment.

    Args:
        segment: Segment to report on
        robustness: Optional robustness score
        actionability: Optional actionability evaluation
        explanation: Optional business explanation
        viability: Optional viability assessment

    Returns:
        SegmentReport
    """
    return SegmentReport(
        segment_id=segment.segment_id,
        name=segment.name,
        summary=segment_to_summary(segment),
        robustness=robustness_to_report(robustness) if robustness else None,
        actionability=actionability_to_report(actionability) if actionability else None,
        explanation=explanation_to_report(explanation) if explanation else None,
        viability=viability_to_report(viability) if viability else None,
    )


def generate_segmentation_report(
    segments: list[Segment],
    *,
    robustness_scores: dict[str, RobustnessScore] | None = None,
    actionability_evaluations: dict[str, ActionabilityEvaluation] | None = None,
    explanations: dict[str, SegmentExplanation] | None = None,
    viabilities: dict[str, SegmentViability] | None = None,
    title: str = "Segmentation Analysis Report",
    metadata: dict[str, Any] | None = None,
) -> SegmentationReport:
    """
    Generate a complete segmentation report.

    Args:
        segments: List of segments to report on
        robustness_scores: Optional mapping of segment_id to RobustnessScore
        actionability_evaluations: Optional mapping of segment_id to ActionabilityEvaluation
        explanations: Optional mapping of segment_id to SegmentExplanation
        viabilities: Optional mapping of segment_id to SegmentViability
        title: Report title
        metadata: Optional metadata to include

    Returns:
        SegmentationReport
    """
    robustness_scores = robustness_scores or {}
    actionability_evaluations = actionability_evaluations or {}
    explanations = explanations or {}
    viabilities = viabilities or {}
    metadata = metadata or {}

    # Generate individual segment reports
    segment_reports: list[SegmentReport] = []
    for segment in segments:
        report = generate_segment_report(
            segment,
            robustness=robustness_scores.get(segment.segment_id),
            actionability=actionability_evaluations.get(segment.segment_id),
            explanation=explanations.get(segment.segment_id),
            viability=viabilities.get(segment.segment_id),
        )
        segment_reports.append(report)

    # Calculate summary statistics
    total_customers = sum(s.size for s in segments)
    total_clv = sum(float(s.total_clv) for s in segments)
    actionable_count = sum(
        1 for e in actionability_evaluations.values()
        if e.is_actionable
    )

    # Robustness distribution
    robustness_tiers: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in robustness_scores.values():
        robustness_tiers[r.robustness_tier.value] += 1

    # Dimension distribution
    dimension_counts: dict[str, int] = {}
    for e in actionability_evaluations.values():
        for dim in e.actionability_dimensions:
            dimension_counts[dim.value] = dimension_counts.get(dim.value, 0) + 1

    summary_stats: dict[str, Any] = {
        "total_segments": len(segments),
        "total_customers": total_customers,
        "total_clv": total_clv,
        "avg_segment_size": total_customers / len(segments) if segments else 0,
        "avg_segment_clv": total_clv / len(segments) if segments else 0,
        "actionable_segments": actionable_count,
        "actionability_rate": actionable_count / len(segments) if segments else 0,
        "robustness_distribution": robustness_tiers,
        "dimension_distribution": dimension_counts,
    }

    return SegmentationReport(
        report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        title=title,
        generated_at=datetime.now().isoformat(),
        total_segments=len(segments),
        total_customers=total_customers,
        actionable_segments=actionable_count,
        segments=segment_reports,
        summary_stats=summary_stats,
        metadata=metadata,
    )


# =============================================================================
# JSON EXPORT
# =============================================================================


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def report_to_dict(report: SegmentReport | SegmentationReport) -> dict[str, Any]:
    """
    Convert report to dictionary.

    Args:
        report: Report to convert

    Returns:
        Dictionary representation
    """
    return asdict(report)


def export_report_to_json(
    report: SegmentReport | SegmentationReport,
    filepath: str | Path | None = None,
    *,
    indent: int = 2,
) -> str:
    """
    Export report to JSON.

    Args:
        report: Report to export
        filepath: Optional filepath to write to
        indent: JSON indentation level

    Returns:
        JSON string

    Raises:
        ReportGenerationError: If export fails
    """
    try:
        report_dict = report_to_dict(report)
        json_str = json.dumps(report_dict, indent=indent, cls=DecimalEncoder)

        if filepath:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)

        return json_str

    except Exception as e:
        raise ReportGenerationError(
            f"Failed to export report to JSON: {e}",
            report_type="json",
        ) from e


def load_report_from_json(filepath: str | Path) -> dict[str, Any]:
    """
    Load report from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Report dictionary

    Raises:
        ReportGenerationError: If load fails
    """
    try:
        path = Path(filepath)
        result: dict[str, Any] = json.loads(path.read_text())
        return result
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to load report from JSON: {e}",
            report_type="json",
        ) from e


# =============================================================================
# TEXT REPORT GENERATION
# =============================================================================


def generate_text_summary(report: SegmentationReport) -> str:
    """
    Generate a text summary of the segmentation report.

    Args:
        report: SegmentationReport to summarize

    Returns:
        Text summary string
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append(f"{report.title}")
    lines.append("=" * 60)
    lines.append(f"Generated: {report.generated_at}")
    lines.append("")

    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Segments: {report.total_segments}")
    lines.append(f"Total Customers: {report.total_customers:,}")
    lines.append(f"Actionable Segments: {report.actionable_segments}")
    lines.append(
        f"Actionability Rate: {report.summary_stats.get('actionability_rate', 0):.1%}"
    )
    lines.append(f"Total CLV: ${report.summary_stats.get('total_clv', 0):,.2f}")
    lines.append("")

    # Robustness Distribution
    robustness_dist = report.summary_stats.get("robustness_distribution", {})
    if robustness_dist:
        lines.append("ROBUSTNESS DISTRIBUTION")
        lines.append("-" * 40)
        for tier, count in robustness_dist.items():
            lines.append(f"  {tier}: {count} segments")
        lines.append("")

    # Dimension Distribution
    dimension_dist = report.summary_stats.get("dimension_distribution", {})
    if dimension_dist:
        lines.append("ACTIONABILITY DIMENSIONS")
        lines.append("-" * 40)
        for dim, count in dimension_dist.items():
            lines.append(f"  {dim}: {count} segments")
        lines.append("")

    # Individual Segments
    lines.append("SEGMENT DETAILS")
    lines.append("-" * 40)

    for seg_report in report.segments:
        lines.append(f"\n[{seg_report.segment_id}] {seg_report.name}")
        lines.append(f"  Size: {seg_report.summary.size} customers")
        lines.append(f"  Total CLV: ${seg_report.summary.total_clv:,.2f}")
        lines.append(f"  Avg CLV: ${seg_report.summary.avg_clv:,.2f}")

        if seg_report.robustness:
            lines.append(
                f"  Robustness: {seg_report.robustness.robustness_tier} "
                f"({seg_report.robustness.overall_robustness:.1%})"
            )

        if seg_report.actionability:
            status = "Actionable" if seg_report.actionability.is_actionable else "Not Actionable"
            lines.append(f"  Status: {status}")
            if seg_report.actionability.dimensions:
                lines.append(f"  Dimensions: {', '.join(seg_report.actionability.dimensions)}")

        if seg_report.explanation:
            lines.append(f"  Summary: {seg_report.explanation.executive_summary[:100]}...")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


def export_text_report(
    report: SegmentationReport,
    filepath: str | Path,
) -> None:
    """
    Export report as text file.

    Args:
        report: Report to export
        filepath: Filepath to write to

    Raises:
        ReportGenerationError: If export fails
    """
    try:
        text = generate_text_summary(report)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to export text report: {e}",
            report_type="text",
        ) from e


# =============================================================================
# SEGMENT REPORTER CLASS
# =============================================================================


class SegmentReporter:
    """
    Reporter class for generating segment reports.

    Coordinates report generation and export.
    """

    def __init__(
        self,
        *,
        title: str = "Segmentation Analysis Report",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize SegmentReporter.

        Args:
            title: Default report title
            metadata: Default metadata
        """
        self.title = title
        self.metadata = metadata or {}

        self._segments: list[Segment] = []
        self._robustness_scores: dict[str, RobustnessScore] = {}
        self._actionability_evaluations: dict[str, ActionabilityEvaluation] = {}
        self._explanations: dict[str, SegmentExplanation] = {}
        self._viabilities: dict[str, SegmentViability] = {}
        self._last_report: SegmentationReport | None = None

    def add_segment(
        self,
        segment: Segment,
        *,
        robustness: RobustnessScore | None = None,
        actionability: ActionabilityEvaluation | None = None,
        explanation: SegmentExplanation | None = None,
        viability: SegmentViability | None = None,
    ) -> None:
        """
        Add a segment with associated data.

        Args:
            segment: Segment to add
            robustness: Optional robustness score
            actionability: Optional actionability evaluation
            explanation: Optional business explanation
            viability: Optional viability assessment
        """
        self._segments.append(segment)

        if robustness:
            self._robustness_scores[segment.segment_id] = robustness
        if actionability:
            self._actionability_evaluations[segment.segment_id] = actionability
        if explanation:
            self._explanations[segment.segment_id] = explanation
        if viability:
            self._viabilities[segment.segment_id] = viability

    def add_segments_batch(
        self,
        segments: list[Segment],
        *,
        robustness_scores: dict[str, RobustnessScore] | None = None,
        actionability_evaluations: dict[str, ActionabilityEvaluation] | None = None,
        explanations: dict[str, SegmentExplanation] | None = None,
        viabilities: dict[str, SegmentViability] | None = None,
    ) -> None:
        """
        Add multiple segments with associated data.

        Args:
            segments: Segments to add
            robustness_scores: Optional robustness scores
            actionability_evaluations: Optional actionability evaluations
            explanations: Optional explanations
            viabilities: Optional viabilities
        """
        self._segments.extend(segments)

        if robustness_scores:
            self._robustness_scores.update(robustness_scores)
        if actionability_evaluations:
            self._actionability_evaluations.update(actionability_evaluations)
        if explanations:
            self._explanations.update(explanations)
        if viabilities:
            self._viabilities.update(viabilities)

    def generate_report(
        self,
        *,
        title: str | None = None,
    ) -> SegmentationReport:
        """
        Generate the segmentation report.

        Args:
            title: Optional override title

        Returns:
            SegmentationReport
        """
        report = generate_segmentation_report(
            self._segments,
            robustness_scores=self._robustness_scores,
            actionability_evaluations=self._actionability_evaluations,
            explanations=self._explanations,
            viabilities=self._viabilities,
            title=title or self.title,
            metadata=self.metadata,
        )
        self._last_report = report
        return report

    def export_json(self, filepath: str | Path) -> str:
        """
        Export report to JSON file.

        Args:
            filepath: Path to write JSON

        Returns:
            JSON string
        """
        if not self._last_report:
            self._last_report = self.generate_report()

        return export_report_to_json(self._last_report, filepath)

    def export_text(self, filepath: str | Path) -> None:
        """
        Export report to text file.

        Args:
            filepath: Path to write text
        """
        if not self._last_report:
            self._last_report = self.generate_report()

        export_text_report(self._last_report, filepath)

    def get_summary(self) -> str:
        """
        Get text summary of the report.

        Returns:
            Text summary string
        """
        if not self._last_report:
            self._last_report = self.generate_report()

        return generate_text_summary(self._last_report)

    def clear(self) -> None:
        """Clear all stored data."""
        self._segments.clear()
        self._robustness_scores.clear()
        self._actionability_evaluations.clear()
        self._explanations.clear()
        self._viabilities.clear()
        self._last_report = None

    @property
    def last_report(self) -> SegmentationReport | None:
        """Get the last generated report."""
        return self._last_report

    @property
    def segment_count(self) -> int:
        """Get count of segments added."""
        return len(self._segments)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_report(
    segments: list[Segment],
    *,
    include_text_summary: bool = True,
) -> tuple[SegmentationReport, str | None]:
    """
    Quick report generation with minimal setup.

    Args:
        segments: Segments to report on
        include_text_summary: Whether to include text summary

    Returns:
        Tuple of (SegmentationReport, optional text summary)
    """
    report = generate_segmentation_report(segments)
    text_summary = generate_text_summary(report) if include_text_summary else None
    return report, text_summary
