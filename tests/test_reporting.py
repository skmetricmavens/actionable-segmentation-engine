"""
Tests for reporting and visualization modules.

Tests segment reporter, JSON export, and visualization functions.
"""

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend for tests
matplotlib.use("Agg")

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    ConfidenceLevel,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentExplanation,
    SegmentMember,
    SegmentViability,
    StrategicGoal,
)
from src.exceptions import ReportGenerationError
from src.reporting.segment_reporter import (
    ActionabilityReport,
    DecimalEncoder,
    ExplanationReport,
    RobustnessReport,
    SegmentationReport,
    SegmentReport,
    SegmentReporter,
    SegmentSummary,
    ViabilityReport,
    actionability_to_report,
    explanation_to_report,
    export_report_to_json,
    export_text_report,
    generate_segment_report,
    generate_segmentation_report,
    generate_text_summary,
    load_report_from_json,
    quick_report,
    report_to_dict,
    robustness_to_report,
    segment_to_summary,
    viability_to_report,
)
from src.reporting.visuals import (
    close_figure,
    plot_actionability_by_segment,
    plot_actionability_dimensions,
    plot_report_summary,
    plot_robustness_heatmap,
    plot_robustness_scores,
    plot_segment_dashboard,
    plot_segment_distribution,
    plot_segment_sizes_pie,
    plot_viability_scores,
    save_figure,
    set_style,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_segment() -> Segment:
    """Create a sample segment for testing."""
    return Segment(
        segment_id="test_segment_1",
        name="High Value Weekend Shoppers",
        description="Customers who shop on weekends with high order values",
        members=[
            SegmentMember(internal_customer_id=f"cust_{i}", membership_score=1.0)
            for i in range(50)
        ],
        size=50,
        defining_traits=[
            "Weekend shopping preference",
            "High average order value",
            "Mobile device users",
        ],
        total_clv=Decimal("25000"),
        avg_clv=Decimal("500"),
        avg_order_value=Decimal("150"),
        actionability_dimensions=[
            ActionabilityDimension.WHAT,
            ActionabilityDimension.WHEN,
            ActionabilityDimension.HOW,
        ],
        strategic_goals=[StrategicGoal.INCREASE_REVENUE],
    )


@pytest.fixture
def sample_segments() -> list[Segment]:
    """Create multiple sample segments."""
    return [
        Segment(
            segment_id=f"segment_{i}",
            name=f"Segment {i}",
            description=f"Test segment {i}",
            size=50 + i * 20,
            total_clv=Decimal(str(5000 + i * 2000)),
            avg_clv=Decimal(str(100 + i * 20)),
            avg_order_value=Decimal(str(50 + i * 10)),
            actionability_dimensions=[ActionabilityDimension.WHO] if i % 2 == 0 else [],
            strategic_goals=[StrategicGoal.INCREASE_REVENUE] if i % 2 == 0 else [],
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_robustness() -> RobustnessScore:
    """Create sample robustness score."""
    return RobustnessScore.calculate(
        segment_id="test_segment_1",
        feature_stability=0.85,
        time_window_consistency=0.90,
    )


@pytest.fixture
def sample_robustness_scores(sample_segments: list[Segment]) -> dict[str, RobustnessScore]:
    """Create robustness scores for multiple segments."""
    scores = {}
    for i, segment in enumerate(sample_segments):
        stability = 0.3 + i * 0.15
        scores[segment.segment_id] = RobustnessScore.calculate(
            segment_id=segment.segment_id,
            feature_stability=min(stability, 0.95),
            time_window_consistency=min(stability + 0.05, 0.95),
        )
    return scores


@pytest.fixture
def sample_actionability() -> ActionabilityEvaluation:
    """Create sample actionability evaluation."""
    return ActionabilityEvaluation(
        segment_id="test_segment_1",
        is_actionable=True,
        reasoning="Segment shows clear actionability across multiple dimensions",
        recommended_action="Target with personalized product recommendations",
        confidence_level=ConfidenceLevel.HIGH,
        actionability_dimensions=[
            ActionabilityDimension.WHAT,
            ActionabilityDimension.WHEN,
            ActionabilityDimension.HOW,
        ],
    )


@pytest.fixture
def sample_actionability_evaluations(sample_segments: list[Segment]) -> dict[str, ActionabilityEvaluation]:
    """Create actionability evaluations for multiple segments."""
    evaluations = {}
    for i, segment in enumerate(sample_segments):
        evaluations[segment.segment_id] = ActionabilityEvaluation(
            segment_id=segment.segment_id,
            is_actionable=i % 2 == 0,
            reasoning=f"Evaluation for segment {i}",
            recommended_action="Take action" if i % 2 == 0 else None,
            confidence_level=ConfidenceLevel.HIGH if i < 2 else ConfidenceLevel.MEDIUM,
            actionability_dimensions=[ActionabilityDimension.WHO] if i % 2 == 0 else [],
        )
    return evaluations


@pytest.fixture
def sample_explanation() -> SegmentExplanation:
    """Create sample segment explanation."""
    return SegmentExplanation(
        segment_id="test_segment_1",
        executive_summary="High-value weekend shoppers with significant CLV",
        key_characteristics=[
            "Premium customer value",
            "Weekend preference",
            "Mobile engagement",
        ],
        recommended_campaign="Launch weekend promotion campaign",
        business_hypothesis="Targeting will increase revenue by 15%",
        expected_roi="Strong positive ROI expected",
        confidence_level=ConfidenceLevel.HIGH,
        confidence_justification="High data quality supports confidence",
    )


@pytest.fixture
def sample_viability(sample_robustness: RobustnessScore) -> SegmentViability:
    """Create sample viability assessment."""
    return SegmentViability(
        segment_id="test_segment_1",
        size=50,
        total_clv=Decimal("25000"),
        marketing_targetability=0.8,
        sales_prioritization=0.7,
        personalization_opportunity=0.75,
        timing_optimization=0.6,
        cost_to_exploit=Decimal("500"),
        expected_roi=1.5,
        revenue_impact="high",
        retention_impact="medium",
        satisfaction_impact="medium",
        robustness_score=sample_robustness,
        confidence_level=ConfidenceLevel.HIGH,
        recommended_action="Target with personalized offers",
        business_hypothesis="Will increase revenue",
        is_approved=True,
    )


@pytest.fixture
def sample_viabilities(
    sample_segments: list[Segment],
    sample_robustness_scores: dict[str, RobustnessScore],
) -> dict[str, SegmentViability]:
    """Create viability assessments for multiple segments."""
    viabilities = {}
    for i, segment in enumerate(sample_segments):
        robustness = sample_robustness_scores.get(segment.segment_id)
        if robustness is None:
            robustness = RobustnessScore.calculate(
                segment_id=segment.segment_id,
                feature_stability=0.5,
                time_window_consistency=0.5,
            )
        viabilities[segment.segment_id] = SegmentViability(
            segment_id=segment.segment_id,
            size=segment.size,
            total_clv=segment.total_clv,
            marketing_targetability=0.5 + i * 0.1,
            sales_prioritization=0.4 + i * 0.1,
            personalization_opportunity=0.6 + i * 0.05,
            timing_optimization=0.3 + i * 0.1,
            cost_to_exploit=Decimal("100") * (i + 1),
            expected_roi=0.5 + i * 0.3,
            revenue_impact="high" if i > 2 else "medium",
            retention_impact="medium",
            satisfaction_impact="medium",
            robustness_score=robustness,
            confidence_level=ConfidenceLevel.MEDIUM,
            recommended_action="Take action",
            business_hypothesis="Will improve metrics",
            is_approved=i % 2 == 0,
        )
    return viabilities


# =============================================================================
# CONVERSION FUNCTION TESTS
# =============================================================================


class TestConversionFunctions:
    """Tests for data conversion functions."""

    def test_segment_to_summary(self, sample_segment: Segment) -> None:
        """Test converting Segment to SegmentSummary."""
        summary = segment_to_summary(sample_segment)

        assert isinstance(summary, SegmentSummary)
        assert summary.segment_id == sample_segment.segment_id
        assert summary.name == sample_segment.name
        assert summary.size == sample_segment.size
        assert summary.total_clv == float(sample_segment.total_clv)
        assert len(summary.defining_traits) == len(sample_segment.defining_traits)

    def test_robustness_to_report(self, sample_robustness: RobustnessScore) -> None:
        """Test converting RobustnessScore to RobustnessReport."""
        report = robustness_to_report(sample_robustness)

        assert isinstance(report, RobustnessReport)
        assert report.segment_id == sample_robustness.segment_id
        assert report.overall_robustness == sample_robustness.overall_robustness
        assert report.robustness_tier == sample_robustness.robustness_tier.value

    def test_actionability_to_report(self, sample_actionability: ActionabilityEvaluation) -> None:
        """Test converting ActionabilityEvaluation to ActionabilityReport."""
        report = actionability_to_report(sample_actionability)

        assert isinstance(report, ActionabilityReport)
        assert report.segment_id == sample_actionability.segment_id
        assert report.is_actionable == sample_actionability.is_actionable
        assert len(report.dimensions) == len(sample_actionability.actionability_dimensions)

    def test_explanation_to_report(self, sample_explanation: SegmentExplanation) -> None:
        """Test converting SegmentExplanation to ExplanationReport."""
        report = explanation_to_report(sample_explanation)

        assert isinstance(report, ExplanationReport)
        assert report.segment_id == sample_explanation.segment_id
        assert report.executive_summary == sample_explanation.executive_summary

    def test_viability_to_report(self, sample_viability: SegmentViability) -> None:
        """Test converting SegmentViability to ViabilityReport."""
        report = viability_to_report(sample_viability)

        assert isinstance(report, ViabilityReport)
        assert report.segment_id == sample_viability.segment_id
        assert report.expected_roi == sample_viability.expected_roi


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================


class TestReportGeneration:
    """Tests for report generation functions."""

    def test_generate_segment_report_minimal(self, sample_segment: Segment) -> None:
        """Test generating segment report with minimal data."""
        report = generate_segment_report(sample_segment)

        assert isinstance(report, SegmentReport)
        assert report.segment_id == sample_segment.segment_id
        assert report.name == sample_segment.name
        assert report.summary is not None
        assert report.robustness is None
        assert report.actionability is None

    def test_generate_segment_report_full(
        self,
        sample_segment: Segment,
        sample_robustness: RobustnessScore,
        sample_actionability: ActionabilityEvaluation,
        sample_explanation: SegmentExplanation,
        sample_viability: SegmentViability,
    ) -> None:
        """Test generating segment report with all data."""
        report = generate_segment_report(
            sample_segment,
            robustness=sample_robustness,
            actionability=sample_actionability,
            explanation=sample_explanation,
            viability=sample_viability,
        )

        assert report.robustness is not None
        assert report.actionability is not None
        assert report.explanation is not None
        assert report.viability is not None

    def test_generate_segmentation_report(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
    ) -> None:
        """Test generating full segmentation report."""
        report = generate_segmentation_report(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            actionability_evaluations=sample_actionability_evaluations,
            title="Test Report",
        )

        assert isinstance(report, SegmentationReport)
        assert report.title == "Test Report"
        assert report.total_segments == len(sample_segments)
        assert len(report.segments) == len(sample_segments)
        assert "total_clv" in report.summary_stats
        assert "robustness_distribution" in report.summary_stats

    def test_generate_segmentation_report_empty(self) -> None:
        """Test generating report with empty segments."""
        report = generate_segmentation_report([])

        assert report.total_segments == 0
        assert report.total_customers == 0
        assert len(report.segments) == 0


# =============================================================================
# JSON EXPORT TESTS
# =============================================================================


class TestJSONExport:
    """Tests for JSON export functionality."""

    def test_report_to_dict(self, sample_segment: Segment) -> None:
        """Test converting report to dictionary."""
        report = generate_segment_report(sample_segment)
        report_dict = report_to_dict(report)

        assert isinstance(report_dict, dict)
        assert "segment_id" in report_dict
        assert "summary" in report_dict

    def test_export_report_to_json_string(self, sample_segment: Segment) -> None:
        """Test exporting report to JSON string."""
        report = generate_segment_report(sample_segment)
        json_str = export_report_to_json(report)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["segment_id"] == sample_segment.segment_id

    def test_export_report_to_json_file(
        self, sample_segments: list[Segment]
    ) -> None:
        """Test exporting report to JSON file."""
        report = generate_segmentation_report(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            json_str = export_report_to_json(report, filepath)

            assert Path(filepath).exists()
            loaded = json.loads(Path(filepath).read_text())
            assert loaded["total_segments"] == len(sample_segments)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_report_from_json(self, sample_segments: list[Segment]) -> None:
        """Test loading report from JSON file."""
        report = generate_segmentation_report(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            export_report_to_json(report, filepath)
            loaded = load_report_from_json(filepath)

            assert isinstance(loaded, dict)
            assert loaded["total_segments"] == report.total_segments
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_decimal_encoder(self) -> None:
        """Test DecimalEncoder handles Decimal types."""
        data = {"value": Decimal("123.45")}
        json_str = json.dumps(data, cls=DecimalEncoder)

        assert "123.45" in json_str
        parsed = json.loads(json_str)
        assert parsed["value"] == 123.45


# =============================================================================
# TEXT REPORT TESTS
# =============================================================================


class TestTextReport:
    """Tests for text report generation."""

    def test_generate_text_summary(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
    ) -> None:
        """Test generating text summary."""
        report = generate_segmentation_report(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            actionability_evaluations=sample_actionability_evaluations,
        )
        text = generate_text_summary(report)

        assert isinstance(text, str)
        assert "EXECUTIVE SUMMARY" in text
        assert "Total Segments" in text
        assert "ROBUSTNESS DISTRIBUTION" in text
        assert "SEGMENT DETAILS" in text

    def test_export_text_report(self, sample_segments: list[Segment]) -> None:
        """Test exporting text report to file."""
        report = generate_segmentation_report(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            export_text_report(report, filepath)

            assert Path(filepath).exists()
            content = Path(filepath).read_text()
            assert "EXECUTIVE SUMMARY" in content
        finally:
            Path(filepath).unlink(missing_ok=True)


# =============================================================================
# SEGMENT REPORTER CLASS TESTS
# =============================================================================


class TestSegmentReporter:
    """Tests for SegmentReporter class."""

    def test_reporter_initialization(self) -> None:
        """Test reporter initialization."""
        reporter = SegmentReporter(title="Test Report")

        assert reporter.title == "Test Report"
        assert reporter.segment_count == 0

    def test_add_segment(
        self,
        sample_segment: Segment,
        sample_robustness: RobustnessScore,
        sample_actionability: ActionabilityEvaluation,
    ) -> None:
        """Test adding segment with data."""
        reporter = SegmentReporter()
        reporter.add_segment(
            sample_segment,
            robustness=sample_robustness,
            actionability=sample_actionability,
        )

        assert reporter.segment_count == 1

    def test_add_segments_batch(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
    ) -> None:
        """Test batch adding segments."""
        reporter = SegmentReporter()
        reporter.add_segments_batch(
            sample_segments,
            robustness_scores=sample_robustness_scores,
        )

        assert reporter.segment_count == len(sample_segments)

    def test_generate_report(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
    ) -> None:
        """Test generating report."""
        reporter = SegmentReporter(title="My Report")
        reporter.add_segments_batch(
            sample_segments,
            robustness_scores=sample_robustness_scores,
        )
        report = reporter.generate_report()

        assert isinstance(report, SegmentationReport)
        assert report.title == "My Report"
        assert reporter.last_report is not None

    def test_export_json(self, sample_segments: list[Segment]) -> None:
        """Test exporting JSON via reporter."""
        reporter = SegmentReporter()
        reporter.add_segments_batch(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            json_str = reporter.export_json(filepath)

            assert Path(filepath).exists()
            assert len(json_str) > 0
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_text(self, sample_segments: list[Segment]) -> None:
        """Test exporting text via reporter."""
        reporter = SegmentReporter()
        reporter.add_segments_batch(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            reporter.export_text(filepath)

            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_summary(self, sample_segments: list[Segment]) -> None:
        """Test getting text summary."""
        reporter = SegmentReporter()
        reporter.add_segments_batch(sample_segments)
        summary = reporter.get_summary()

        assert isinstance(summary, str)
        assert "Total Segments" in summary

    def test_clear(self, sample_segments: list[Segment]) -> None:
        """Test clearing reporter data."""
        reporter = SegmentReporter()
        reporter.add_segments_batch(sample_segments)
        reporter.generate_report()

        reporter.clear()

        assert reporter.segment_count == 0
        assert reporter.last_report is None


# =============================================================================
# QUICK REPORT TESTS
# =============================================================================


class TestQuickReport:
    """Tests for quick_report function."""

    def test_quick_report_with_summary(self, sample_segments: list[Segment]) -> None:
        """Test quick report with text summary."""
        report, text = quick_report(sample_segments, include_text_summary=True)

        assert isinstance(report, SegmentationReport)
        assert text is not None
        assert "EXECUTIVE SUMMARY" in text

    def test_quick_report_without_summary(self, sample_segments: list[Segment]) -> None:
        """Test quick report without text summary."""
        report, text = quick_report(sample_segments, include_text_summary=False)

        assert isinstance(report, SegmentationReport)
        assert text is None


# =============================================================================
# VISUALIZATION TESTS
# =============================================================================


class TestVisualization:
    """Tests for visualization functions."""

    def test_set_style(self) -> None:
        """Test setting plot style."""
        set_style("darkgrid")
        # Just verify no exceptions
        assert True

    def test_plot_segment_distribution_size(self, sample_segments: list[Segment]) -> None:
        """Test plotting segment distribution by size."""
        fig = plot_segment_distribution(sample_segments, by="size")

        assert fig is not None
        close_figure(fig)

    def test_plot_segment_distribution_clv(self, sample_segments: list[Segment]) -> None:
        """Test plotting segment distribution by CLV."""
        fig = plot_segment_distribution(sample_segments, by="clv")

        assert fig is not None
        close_figure(fig)

    def test_plot_segment_distribution_avg_clv(self, sample_segments: list[Segment]) -> None:
        """Test plotting segment distribution by avg CLV."""
        fig = plot_segment_distribution(sample_segments, by="avg_clv")

        assert fig is not None
        close_figure(fig)

    def test_plot_segment_distribution_aov(self, sample_segments: list[Segment]) -> None:
        """Test plotting segment distribution by AOV."""
        fig = plot_segment_distribution(sample_segments, by="aov")

        assert fig is not None
        close_figure(fig)

    def test_plot_segment_sizes_pie(self, sample_segments: list[Segment]) -> None:
        """Test plotting segment sizes pie chart."""
        fig = plot_segment_sizes_pie(sample_segments)

        assert fig is not None
        close_figure(fig)

    def test_plot_robustness_scores(
        self, sample_robustness_scores: dict[str, RobustnessScore]
    ) -> None:
        """Test plotting robustness scores."""
        fig = plot_robustness_scores(sample_robustness_scores)

        assert fig is not None
        close_figure(fig)

    def test_plot_robustness_heatmap(
        self, sample_robustness_scores: dict[str, RobustnessScore]
    ) -> None:
        """Test plotting robustness heatmap."""
        fig = plot_robustness_heatmap(sample_robustness_scores)

        assert fig is not None
        close_figure(fig)

    def test_plot_actionability_dimensions(
        self, sample_actionability_evaluations: dict[str, ActionabilityEvaluation]
    ) -> None:
        """Test plotting actionability dimensions."""
        fig = plot_actionability_dimensions(sample_actionability_evaluations)

        assert fig is not None
        close_figure(fig)

    def test_plot_actionability_by_segment(
        self, sample_actionability_evaluations: dict[str, ActionabilityEvaluation]
    ) -> None:
        """Test plotting actionability by segment."""
        fig = plot_actionability_by_segment(sample_actionability_evaluations)

        assert fig is not None
        close_figure(fig)

    def test_plot_viability_scores(
        self, sample_viabilities: dict[str, SegmentViability]
    ) -> None:
        """Test plotting viability scores."""
        fig = plot_viability_scores(sample_viabilities)

        assert fig is not None
        close_figure(fig)

    def test_plot_segment_dashboard(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
    ) -> None:
        """Test plotting segment dashboard."""
        fig = plot_segment_dashboard(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            evaluations=sample_actionability_evaluations,
        )

        assert fig is not None
        close_figure(fig)

    def test_plot_report_summary(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
    ) -> None:
        """Test plotting report summary."""
        report = generate_segmentation_report(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            actionability_evaluations=sample_actionability_evaluations,
        )
        fig = plot_report_summary(report)

        assert fig is not None
        close_figure(fig)

    def test_save_figure(self, sample_segments: list[Segment]) -> None:
        """Test saving figure to file."""
        fig = plot_segment_distribution(sample_segments)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            save_figure(fig, filepath)
            assert Path(filepath).exists()
        finally:
            close_figure(fig)
            Path(filepath).unlink(missing_ok=True)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_load_nonexistent_json(self) -> None:
        """Test loading nonexistent JSON raises error."""
        with pytest.raises(ReportGenerationError):
            load_report_from_json("/nonexistent/path/report.json")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestReportingIntegration:
    """Integration tests for full reporting workflow."""

    def test_full_reporting_workflow(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
        sample_viabilities: dict[str, SegmentViability],
    ) -> None:
        """Test complete reporting workflow."""
        # Create reporter
        reporter = SegmentReporter(title="Integration Test Report")

        # Add data
        reporter.add_segments_batch(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            actionability_evaluations=sample_actionability_evaluations,
            viabilities=sample_viabilities,
        )

        # Generate report
        report = reporter.generate_report()

        # Verify report structure
        assert report.total_segments == len(sample_segments)
        assert all(sr.robustness is not None for sr in report.segments)
        assert all(sr.actionability is not None for sr in report.segments)
        assert all(sr.viability is not None for sr in report.segments)

        # Export to different formats
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            text_path = Path(tmpdir) / "report.txt"

            reporter.export_json(json_path)
            reporter.export_text(text_path)

            assert json_path.exists()
            assert text_path.exists()

            # Verify JSON content
            loaded = load_report_from_json(json_path)
            assert loaded["total_segments"] == report.total_segments

    def test_visualization_workflow(
        self,
        sample_segments: list[Segment],
        sample_robustness_scores: dict[str, RobustnessScore],
        sample_actionability_evaluations: dict[str, ActionabilityEvaluation],
    ) -> None:
        """Test visualization workflow."""
        # Generate report
        report = generate_segmentation_report(
            sample_segments,
            robustness_scores=sample_robustness_scores,
            actionability_evaluations=sample_actionability_evaluations,
        )

        # Create multiple visualizations
        fig1 = plot_segment_distribution(sample_segments)
        fig2 = plot_robustness_scores(sample_robustness_scores)
        fig3 = plot_actionability_dimensions(sample_actionability_evaluations)
        fig4 = plot_report_summary(report)

        # Verify all figures created
        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4])

        # Cleanup
        for fig in [fig1, fig2, fig3, fig4]:
            close_figure(fig)
