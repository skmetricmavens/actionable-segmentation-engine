"""
Module: pipeline

Purpose: Main orchestrator for the end-to-end segmentation pipeline.

Key Functions:
- run_pipeline: Execute complete pipeline from data to insights
- PipelineConfig: Configuration for pipeline execution
- PipelineResult: Container for pipeline outputs

Architecture Notes:
- Orchestrates all pipeline stages
- Supports both synthetic and real data sources
- Produces comprehensive segment reports
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Literal, Protocol

from src.data.joiner import MergeMap, resolve_customer_merges
from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    CustomerIdHistory,
    CustomerProfile,
    EventRecord,
    RobustnessScore,
    Segment,
    SegmentExplanation,
    SegmentViability,
    StrategicGoal,
    SyntheticDataset,
)
from src.data.synthetic_generator import (
    SyntheticDataGenerator,
    generate_small_dataset,
)
from src.exceptions import (
    PipelineError,
    ProfileBuildError,
)
from src.features.profile_builder import (
    ProfileBuilder,
    build_profiles_batch,
    profile_summary_stats,
)
from src.llm.actionability_filter import (
    ActionabilityFilter,
    evaluate_actionability,
)
from src.llm.segment_explainer import (
    SegmentExplainer,
    explain_segment,
    format_segment_for_presentation,
)
from src.reporting.segment_reporter import (
    SegmentationReport,
    SegmentReport,
    generate_segmentation_report,
)
from src.segmentation.clusterer import (
    ClusteringResult,
    CustomerClusterer,
    get_cluster_summary,
)
from src.segmentation.segment_validator import (
    SegmentValidator,
    ValidationCriteria,
    ValidationResult,
    build_segment_viability,
)
from src.segmentation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityAnalysisResult,
    get_sensitivity_summary,
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Data generation
    n_customers: int = 1000
    data_seed: int = 42
    merge_probability: float = 0.15
    date_range: tuple[datetime, datetime] | None = None

    # Profile building
    min_events_per_customer: int = 1
    reference_date: datetime | None = None

    # Clustering
    n_clusters: int | None = None  # None = auto-select
    cluster_seed: int = 42
    auto_select_k: bool = True
    k_range: tuple[int, int] = (3, 10)

    # Sensitivity analysis
    run_sensitivity: bool = True
    include_sampling_stability: bool = True

    # Validation
    validation_criteria: ValidationCriteria | None = None

    # LLM / Actionability
    use_llm: bool = False  # Use mock by default for MVP
    min_actionability_dimensions: int = 1

    # Output options
    generate_report: bool = True
    verbose: bool = False


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""

    stage_name: str
    success: bool
    duration_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    # Core outputs
    profiles: list[CustomerProfile]
    segments: list[Segment]
    robustness_scores: dict[str, RobustnessScore]
    viabilities: dict[str, SegmentViability]
    actionability_evaluations: dict[str, ActionabilityEvaluation]
    explanations: dict[str, SegmentExplanation]

    # Intermediate results
    merge_map: MergeMap
    clustering_result: ClusteringResult | None
    sensitivity_result: SensitivityAnalysisResult | None
    validation_results: dict[str, ValidationResult]

    # Report
    report: SegmentationReport | None

    # Metadata
    config: PipelineConfig
    stage_results: list[PipelineStageResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True
    error_message: str | None = None

    @property
    def actionable_segments(self) -> list[Segment]:
        """Get only actionable segments."""
        actionable = []
        for seg in self.segments:
            eval_result = self.actionability_evaluations.get(seg.segment_id)
            if eval_result and eval_result.is_actionable:
                actionable.append(seg)
        return actionable

    @property
    def valid_segments(self) -> list[Segment]:
        """Get only validated segments."""
        return [
            seg
            for seg in self.segments
            if self.validation_results.get(seg.segment_id, ValidationResult(is_valid=False)).is_valid
        ]

    @property
    def production_ready_segments(self) -> list[Segment]:
        """Get segments that are both valid and actionable."""
        valid_ids = {seg.segment_id for seg in self.valid_segments}
        actionable_ids = {seg.segment_id for seg in self.actionable_segments}
        ready_ids = valid_ids & actionable_ids

        return [seg for seg in self.segments if seg.segment_id in ready_ids]

    def get_segment_details(self, segment_id: str) -> dict[str, Any] | None:
        """Get complete details for a segment."""
        segment = next((s for s in self.segments if s.segment_id == segment_id), None)
        if not segment:
            return None

        return {
            "segment": segment,
            "robustness": self.robustness_scores.get(segment_id),
            "viability": self.viabilities.get(segment_id),
            "actionability": self.actionability_evaluations.get(segment_id),
            "explanation": self.explanations.get(segment_id),
            "validation": self.validation_results.get(segment_id),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of pipeline results."""
        return {
            "total_customers": len(self.profiles),
            "total_segments": len(self.segments),
            "valid_segments": len(self.valid_segments),
            "actionable_segments": len(self.actionable_segments),
            "production_ready_segments": len(self.production_ready_segments),
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "stages": [
                {
                    "name": s.stage_name,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                }
                for s in self.stage_results
            ],
        }


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================


def _time_stage(
    stage_name: str,
    func: Callable[[], Any],
    verbose: bool = False,
) -> tuple[Any, PipelineStageResult]:
    """Execute a stage and time it."""
    import time

    if verbose:
        print(f"[Pipeline] Starting: {stage_name}")

    start = time.perf_counter()
    try:
        result = func()
        duration = (time.perf_counter() - start) * 1000

        stage_result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            duration_ms=duration,
        )

        if verbose:
            print(f"[Pipeline] Completed: {stage_name} ({duration:.1f}ms)")

        return result, stage_result

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        stage_result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            duration_ms=duration,
            error_message=str(e),
        )

        if verbose:
            print(f"[Pipeline] Failed: {stage_name} - {e}")

        raise


def run_pipeline(
    config: PipelineConfig | None = None,
    *,
    events: list[EventRecord] | None = None,
    id_history: list[CustomerIdHistory] | None = None,
    dataset: SyntheticDataset | None = None,
) -> PipelineResult:
    """
    Execute the complete segmentation pipeline.

    Can accept data in three ways:
    1. events + id_history: Raw event data and merge history
    2. dataset: Pre-generated SyntheticDataset
    3. Neither: Generate synthetic data based on config

    Args:
        config: Pipeline configuration
        events: Optional list of event records
        id_history: Optional list of customer ID history records
        dataset: Optional pre-generated synthetic dataset

    Returns:
        PipelineResult with all outputs

    Raises:
        PipelineError: If pipeline execution fails
    """
    import time

    config = config or PipelineConfig()
    start_time = time.perf_counter()
    stage_results: list[PipelineStageResult] = []

    try:
        # Stage 1: Data Acquisition
        def acquire_data() -> tuple[list[EventRecord], list[CustomerIdHistory]]:
            if events is not None and id_history is not None:
                return events, id_history
            elif dataset is not None:
                return dataset.events, dataset.id_history
            else:
                # Generate synthetic data
                date_range = config.date_range or (
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 6, 30, tzinfo=timezone.utc),
                )
                generator = SyntheticDataGenerator(seed=config.data_seed)
                generated = generator.generate_dataset(
                    n_customers=config.n_customers,
                    date_range=date_range,
                    merge_probability=config.merge_probability,
                )
                return generated.events, generated.id_history

        (event_data, history_data), stage = _time_stage(
            "Data Acquisition",
            acquire_data,
            config.verbose,
        )
        stage.metrics = {"n_events": len(event_data), "n_merges": len(history_data)}
        stage_results.append(stage)

        # Stage 2: Customer ID Resolution
        def resolve_merges() -> MergeMap:
            return resolve_customer_merges(history_data)

        merge_map, stage = _time_stage(
            "ID Resolution",
            resolve_merges,
            config.verbose,
        )
        stage.metrics = {"n_mappings": len(merge_map)}
        stage_results.append(stage)

        # Stage 3: Profile Building
        def build_profiles() -> list[CustomerProfile]:
            builder = ProfileBuilder(
                merge_map=merge_map,
                reference_date=config.reference_date,
                min_events=config.min_events_per_customer,
            )
            return builder.build_all(event_data)

        profiles, stage = _time_stage(
            "Profile Building",
            build_profiles,
            config.verbose,
        )
        stage.metrics = {
            "n_profiles": len(profiles),
            **profile_summary_stats(profiles),
        }
        stage_results.append(stage)

        if not profiles:
            raise PipelineError(
                "No customer profiles built",
                stage="Profile Building",
            )

        # Stage 4: Clustering
        def cluster_customers() -> tuple[list[Segment], ClusteringResult]:
            n_clusters = config.n_clusters
            if n_clusters is None and config.auto_select_k:
                n_clusters = 5  # Will be auto-selected

            clusterer = CustomerClusterer(
                n_clusters=n_clusters or 5,
                random_seed=config.cluster_seed,
                auto_select_k=config.auto_select_k,
                k_range=config.k_range,
            )

            segments = clusterer.create_segments(profiles)
            return segments, clusterer.last_result

        (segments, clustering_result), stage = _time_stage(
            "Clustering",
            cluster_customers,
            config.verbose,
        )
        stage.metrics = get_cluster_summary(clustering_result) if clustering_result else {}
        stage_results.append(stage)

        # Stage 5: Sensitivity Analysis
        sensitivity_result: SensitivityAnalysisResult | None = None
        robustness_scores: dict[str, RobustnessScore] = {}

        if config.run_sensitivity:
            def analyze_sensitivity() -> SensitivityAnalysisResult:
                analyzer = SensitivityAnalyzer(
                    n_clusters=clustering_result.n_clusters if clustering_result else 5,
                    random_seed=config.cluster_seed,
                )
                return analyzer.analyze(profiles)

            sensitivity_result, stage = _time_stage(
                "Sensitivity Analysis",
                analyze_sensitivity,
                config.verbose,
            )
            stage.metrics = get_sensitivity_summary(sensitivity_result)
            stage_results.append(stage)

            # Get per-segment robustness scores
            def compute_robustness() -> dict[str, RobustnessScore]:
                analyzer = SensitivityAnalyzer(
                    n_clusters=clustering_result.n_clusters if clustering_result else 5,
                    random_seed=config.cluster_seed,
                )
                return analyzer.analyze_segments(profiles, segments)

            robustness_scores, stage = _time_stage(
                "Robustness Scoring",
                compute_robustness,
                config.verbose,
            )
            stage.metrics = {"n_scores": len(robustness_scores)}
            stage_results.append(stage)

        # Stage 6: Validation
        def validate_segments() -> dict[str, ValidationResult]:
            validator = SegmentValidator(
                criteria=config.validation_criteria,
                total_customers=len(profiles),
            )
            return validator.validate_batch(segments, robustness_scores=robustness_scores)

        validation_results, stage = _time_stage(
            "Validation",
            validate_segments,
            config.verbose,
        )
        stage.metrics = {
            "valid": sum(1 for r in validation_results.values() if r.is_valid),
            "invalid": sum(1 for r in validation_results.values() if not r.is_valid),
        }
        stage_results.append(stage)

        # Stage 7: Viability Assessment
        def assess_viability() -> dict[str, SegmentViability]:
            validator = SegmentValidator(
                criteria=config.validation_criteria,
                total_customers=len(profiles),
            )
            viabilities: dict[str, SegmentViability] = {}
            for segment in segments:
                robustness = robustness_scores.get(segment.segment_id)
                viabilities[segment.segment_id] = validator.build_viability(
                    segment,
                    robustness=robustness,
                )
            return viabilities

        viabilities, stage = _time_stage(
            "Viability Assessment",
            assess_viability,
            config.verbose,
        )
        stage.metrics = {"n_assessments": len(viabilities)}
        stage_results.append(stage)

        # Stage 8: Actionability Evaluation
        def evaluate_actionability_all() -> dict[str, ActionabilityEvaluation]:
            filter_obj = ActionabilityFilter(
                min_dimensions=config.min_actionability_dimensions,
                use_llm=config.use_llm,
            )
            return filter_obj.evaluate_batch(segments, robustness_scores=robustness_scores)

        actionability_evaluations, stage = _time_stage(
            "Actionability Evaluation",
            evaluate_actionability_all,
            config.verbose,
        )
        stage.metrics = {
            "actionable": sum(1 for e in actionability_evaluations.values() if e.is_actionable),
            "not_actionable": sum(1 for e in actionability_evaluations.values() if not e.is_actionable),
        }
        stage_results.append(stage)

        # Stage 9: Explanation Generation
        def generate_explanations() -> dict[str, SegmentExplanation]:
            explainer = SegmentExplainer(use_llm=config.use_llm)
            return explainer.explain_batch(
                segments,
                robustness_scores=robustness_scores,
                viabilities=viabilities,
                actionabilities=actionability_evaluations,
            )

        explanations, stage = _time_stage(
            "Explanation Generation",
            generate_explanations,
            config.verbose,
        )
        stage.metrics = {"n_explanations": len(explanations)}
        stage_results.append(stage)

        # Stage 10: Report Generation
        report: SegmentationReport | None = None
        if config.generate_report:
            def create_report() -> SegmentationReport:
                return generate_segmentation_report(
                    segments=segments,
                    robustness_scores=robustness_scores,
                    actionability_evaluations=actionability_evaluations,
                    explanations=explanations,
                    viabilities=viabilities,
                )

            report, stage = _time_stage(
                "Report Generation",
                create_report,
                config.verbose,
            )
            stage.metrics = {"n_segment_reports": len(report.segments) if report else 0}
            stage_results.append(stage)

        # Calculate total duration
        total_duration = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            profiles=profiles,
            segments=segments,
            robustness_scores=robustness_scores,
            viabilities=viabilities,
            actionability_evaluations=actionability_evaluations,
            explanations=explanations,
            merge_map=merge_map,
            clustering_result=clustering_result,
            sensitivity_result=sensitivity_result,
            validation_results=validation_results,
            report=report,
            config=config,
            stage_results=stage_results,
            total_duration_ms=total_duration,
            success=True,
        )

    except Exception as e:
        total_duration = (time.perf_counter() - start_time) * 1000

        # Return partial result on failure
        return PipelineResult(
            profiles=[],
            segments=[],
            robustness_scores={},
            viabilities={},
            actionability_evaluations={},
            explanations={},
            merge_map={},
            clustering_result=None,
            sensitivity_result=None,
            validation_results={},
            report=None,
            config=config,
            stage_results=stage_results,
            total_duration_ms=total_duration,
            success=False,
            error_message=str(e),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_segmentation(
    n_customers: int = 500,
    n_clusters: int = 5,
    *,
    seed: int = 42,
    verbose: bool = False,
) -> PipelineResult:
    """
    Run quick segmentation with minimal configuration.

    Useful for demos and quick exploration.

    Args:
        n_customers: Number of synthetic customers
        n_clusters: Number of segments
        seed: Random seed
        verbose: Print progress

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        n_customers=n_customers,
        n_clusters=n_clusters,
        data_seed=seed,
        cluster_seed=seed,
        auto_select_k=False,
        run_sensitivity=True,
        generate_report=True,
        verbose=verbose,
    )
    return run_pipeline(config)


def run_pipeline_on_dataset(
    dataset: SyntheticDataset,
    *,
    n_clusters: int | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """
    Run pipeline on a pre-generated dataset.

    Args:
        dataset: SyntheticDataset to process
        n_clusters: Optional number of clusters
        verbose: Print progress

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        n_clusters=n_clusters,
        auto_select_k=n_clusters is None,
        verbose=verbose,
    )
    return run_pipeline(config, dataset=dataset)


def get_pipeline_metrics(result: PipelineResult) -> dict[str, Any]:
    """
    Extract key metrics from pipeline result.

    Args:
        result: PipelineResult

    Returns:
        Dictionary of metrics
    """
    metrics: dict[str, Any] = {
        "success": result.success,
        "total_duration_ms": result.total_duration_ms,
        "n_profiles": len(result.profiles),
        "n_segments": len(result.segments),
        "n_valid_segments": len(result.valid_segments),
        "n_actionable_segments": len(result.actionable_segments),
        "n_production_ready": len(result.production_ready_segments),
    }

    # Add stage timings
    stage_timings = {
        f"stage_{s.stage_name.lower().replace(' ', '_')}_ms": s.duration_ms
        for s in result.stage_results
    }
    metrics.update(stage_timings)

    # Add clustering metrics if available
    if result.clustering_result:
        metrics["silhouette_score"] = result.clustering_result.silhouette
        metrics["inertia"] = result.clustering_result.inertia

    # Add sensitivity metrics if available
    if result.sensitivity_result:
        metrics["overall_robustness"] = result.sensitivity_result.overall_robustness

    return metrics


def format_pipeline_summary(result: PipelineResult) -> str:
    """
    Format pipeline result as human-readable summary.

    Args:
        result: PipelineResult

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "SEGMENTATION PIPELINE RESULTS",
        "=" * 60,
        "",
        f"Status: {'SUCCESS' if result.success else 'FAILED'}",
        f"Duration: {result.total_duration_ms:.1f}ms",
        "",
        "DATA:",
        f"  - Customer profiles: {len(result.profiles)}",
        f"  - Merge mappings: {len(result.merge_map)}",
        "",
        "SEGMENTATION:",
        f"  - Total segments: {len(result.segments)}",
        f"  - Valid segments: {len(result.valid_segments)}",
        f"  - Actionable segments: {len(result.actionable_segments)}",
        f"  - Production ready: {len(result.production_ready_segments)}",
        "",
    ]

    if result.clustering_result:
        lines.extend([
            "CLUSTERING:",
            f"  - Silhouette score: {result.clustering_result.silhouette:.3f}" if result.clustering_result.silhouette else "  - Silhouette score: N/A",
            f"  - Inertia: {result.clustering_result.inertia:.1f}",
            "",
        ])

    if result.sensitivity_result:
        lines.extend([
            "ROBUSTNESS:",
            f"  - Overall: {result.sensitivity_result.overall_robustness:.3f}",
            f"  - Feature stability: {result.sensitivity_result.feature_sensitivity.feature_stability:.3f}",
            f"  - Time consistency: {result.sensitivity_result.time_window_sensitivity.time_consistency:.3f}",
            "",
        ])

    lines.extend([
        "STAGE TIMINGS:",
    ])
    for stage in result.stage_results:
        status = "✓" if stage.success else "✗"
        lines.append(f"  {status} {stage.stage_name}: {stage.duration_ms:.1f}ms")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def print_segment_details(
    result: PipelineResult,
    segment_id: str,
) -> None:
    """
    Print detailed information about a specific segment.

    Args:
        result: PipelineResult
        segment_id: ID of segment to display
    """
    details = result.get_segment_details(segment_id)
    if not details:
        print(f"Segment '{segment_id}' not found")
        return

    segment: Segment = details["segment"]
    robustness: RobustnessScore | None = details["robustness"]
    viability: SegmentViability | None = details["viability"]
    actionability: ActionabilityEvaluation | None = details["actionability"]
    explanation: SegmentExplanation | None = details["explanation"]
    validation: ValidationResult | None = details["validation"]

    print(f"\n{'=' * 60}")
    print(f"SEGMENT: {segment.name}")
    print(f"{'=' * 60}")

    print(f"\nBASIC INFO:")
    print(f"  ID: {segment.segment_id}")
    print(f"  Size: {segment.size} customers")
    print(f"  Total CLV: ${segment.total_clv:,.2f}")
    print(f"  Avg CLV: ${segment.avg_clv:,.2f}")
    print(f"  Avg AOV: ${segment.avg_order_value:,.2f}")

    print(f"\nDEFINING TRAITS:")
    for trait in segment.defining_traits[:5]:
        print(f"  - {trait}")

    if validation:
        print(f"\nVALIDATION:")
        print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
        if validation.rejection_reasons:
            print(f"  Rejection reasons:")
            for reason in validation.rejection_reasons:
                print(f"    - {reason}")

    if robustness:
        print(f"\nROBUSTNESS:")
        print(f"  Overall: {robustness.overall_robustness:.3f}")
        print(f"  Tier: {robustness.robustness_tier.value}")
        print(f"  Production ready: {'Yes' if robustness.is_production_ready else 'No'}")

    if actionability:
        print(f"\nACTIONABILITY:")
        print(f"  Actionable: {'Yes' if actionability.is_actionable else 'No'}")
        print(f"  Dimensions: {', '.join(d.value for d in actionability.actionability_dimensions)}")
        print(f"  Recommended action: {actionability.recommended_action}")

    if explanation:
        print(f"\nEXPLANATION:")
        print(f"  Summary: {explanation.executive_summary}")
        print(f"  Campaign: {explanation.recommended_campaign}")
        print(f"  Expected ROI: {explanation.expected_roi}")

    print(f"\n{'=' * 60}\n")


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_results_to_dict(result: PipelineResult) -> dict[str, Any]:
    """
    Export pipeline results to a dictionary for serialization.

    Args:
        result: PipelineResult

    Returns:
        Dictionary suitable for JSON serialization
    """
    segments_data = []
    for segment in result.segments:
        seg_dict = {
            "segment_id": segment.segment_id,
            "name": segment.name,
            "size": segment.size,
            "total_clv": float(segment.total_clv),
            "avg_clv": float(segment.avg_clv),
            "avg_order_value": float(segment.avg_order_value),
            "defining_traits": segment.defining_traits,
        }

        # Add additional data
        if segment.segment_id in result.robustness_scores:
            rob = result.robustness_scores[segment.segment_id]
            seg_dict["robustness"] = {
                "overall": rob.overall_robustness,
                "tier": rob.robustness_tier.value,
                "production_ready": rob.is_production_ready,
            }

        if segment.segment_id in result.validation_results:
            val = result.validation_results[segment.segment_id]
            seg_dict["validation"] = {
                "valid": val.is_valid,
                "rejection_reasons": val.rejection_reasons,
            }

        if segment.segment_id in result.actionability_evaluations:
            act = result.actionability_evaluations[segment.segment_id]
            seg_dict["actionability"] = {
                "actionable": act.is_actionable,
                "dimensions": [d.value for d in act.actionability_dimensions],
                "recommended_action": act.recommended_action,
            }

        if segment.segment_id in result.explanations:
            exp = result.explanations[segment.segment_id]
            seg_dict["explanation"] = {
                "summary": exp.executive_summary,
                "campaign": exp.recommended_campaign,
                "roi": exp.expected_roi,
            }

        segments_data.append(seg_dict)

    return {
        "success": result.success,
        "duration_ms": result.total_duration_ms,
        "n_profiles": len(result.profiles),
        "n_segments": len(result.segments),
        "segments": segments_data,
        "summary": result.get_summary(),
    }
