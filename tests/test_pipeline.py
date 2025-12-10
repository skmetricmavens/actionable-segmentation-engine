"""
Tests for the end-to-end segmentation pipeline.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.data.schemas import (
    ActionabilityDimension,
    CustomerIdHistory,
    EventRecord,
    EventType,
    EventProperties,
    SyntheticDataset,
)
from src.data.synthetic_generator import (
    SyntheticDataGenerator,
    generate_small_dataset,
)
from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStageResult,
    export_results_to_dict,
    format_pipeline_summary,
    get_pipeline_metrics,
    quick_segmentation,
    run_pipeline,
    run_pipeline_on_dataset,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def small_dataset() -> SyntheticDataset:
    """Generate a small dataset for testing."""
    return generate_small_dataset(seed=42)


@pytest.fixture
def minimal_config() -> PipelineConfig:
    """Minimal configuration for fast tests."""
    return PipelineConfig(
        n_customers=100,
        n_clusters=3,
        auto_select_k=False,
        run_sensitivity=False,
        generate_report=False,
        data_seed=42,
        cluster_seed=42,
    )


@pytest.fixture
def full_config() -> PipelineConfig:
    """Full configuration with all features enabled."""
    return PipelineConfig(
        n_customers=200,
        n_clusters=4,
        auto_select_k=False,
        run_sensitivity=True,
        include_sampling_stability=True,
        generate_report=True,
        data_seed=42,
        cluster_seed=42,
    )


# =============================================================================
# PIPELINE CONFIG TESTS
# =============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.n_customers == 1000
        assert config.data_seed == 42
        assert config.n_clusters is None
        assert config.auto_select_k is True
        assert config.run_sensitivity is True
        assert config.generate_report is True
        assert config.use_llm is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PipelineConfig(
            n_customers=500,
            n_clusters=6,
            auto_select_k=False,
            run_sensitivity=False,
            verbose=True,
        )

        assert config.n_customers == 500
        assert config.n_clusters == 6
        assert config.auto_select_k is False
        assert config.run_sensitivity is False
        assert config.verbose is True

    def test_config_date_range(self) -> None:
        """Test configuration with date range."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        config = PipelineConfig(
            date_range=(start, end),
        )

        assert config.date_range == (start, end)


# =============================================================================
# PIPELINE EXECUTION TESTS
# =============================================================================


class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_run_pipeline_minimal(self, minimal_config: PipelineConfig) -> None:
        """Test pipeline with minimal configuration."""
        result = run_pipeline(minimal_config)

        assert result.success is True
        assert len(result.profiles) > 0
        assert len(result.segments) == 3  # n_clusters=3
        assert result.merge_map is not None
        assert result.clustering_result is not None
        assert result.error_message is None

    def test_run_pipeline_full(self, full_config: PipelineConfig) -> None:
        """Test pipeline with full configuration."""
        result = run_pipeline(full_config)

        assert result.success is True
        assert len(result.profiles) > 0
        assert len(result.segments) == 4  # n_clusters=4
        assert len(result.robustness_scores) == len(result.segments)
        assert len(result.viabilities) == len(result.segments)
        assert len(result.actionability_evaluations) == len(result.segments)
        assert len(result.explanations) == len(result.segments)
        assert result.report is not None
        assert result.sensitivity_result is not None

    def test_run_pipeline_with_dataset(self, small_dataset: SyntheticDataset) -> None:
        """Test pipeline with pre-generated dataset."""
        config = PipelineConfig(
            n_clusters=4,
            auto_select_k=False,
            run_sensitivity=False,
            generate_report=False,
        )

        result = run_pipeline(config, dataset=small_dataset)

        assert result.success is True
        assert len(result.profiles) > 0
        assert len(result.segments) == 4

    def test_run_pipeline_verbose(self, minimal_config: PipelineConfig, capsys: pytest.CaptureFixture) -> None:
        """Test verbose output."""
        minimal_config.verbose = True
        result = run_pipeline(minimal_config)

        captured = capsys.readouterr()
        assert "[Pipeline] Starting:" in captured.out
        assert "[Pipeline] Completed:" in captured.out
        assert result.success is True

    def test_run_pipeline_stages(self, minimal_config: PipelineConfig) -> None:
        """Test that all expected stages are executed."""
        result = run_pipeline(minimal_config)

        stage_names = [s.stage_name for s in result.stage_results]
        assert "Data Acquisition" in stage_names
        assert "ID Resolution" in stage_names
        assert "Profile Building" in stage_names
        assert "Clustering" in stage_names
        assert "Validation" in stage_names
        assert "Viability Assessment" in stage_names
        assert "Actionability Evaluation" in stage_names
        assert "Explanation Generation" in stage_names

    def test_run_pipeline_with_sensitivity(self) -> None:
        """Test pipeline with sensitivity analysis enabled."""
        config = PipelineConfig(
            n_customers=150,
            n_clusters=3,
            auto_select_k=False,
            run_sensitivity=True,
            generate_report=False,
        )

        result = run_pipeline(config)

        assert result.success is True
        assert result.sensitivity_result is not None
        assert len(result.robustness_scores) == 3

        # Check robustness stage was run
        stage_names = [s.stage_name for s in result.stage_results]
        assert "Sensitivity Analysis" in stage_names
        assert "Robustness Scoring" in stage_names

    def test_run_pipeline_generates_report(self) -> None:
        """Test report generation."""
        config = PipelineConfig(
            n_customers=100,
            n_clusters=3,
            auto_select_k=False,
            run_sensitivity=False,
            generate_report=True,
        )

        result = run_pipeline(config)

        assert result.success is True
        assert result.report is not None
        assert len(result.report.segments) == 3

    def test_pipeline_timing(self, minimal_config: PipelineConfig) -> None:
        """Test that timings are recorded."""
        result = run_pipeline(minimal_config)

        assert result.total_duration_ms > 0
        for stage in result.stage_results:
            assert stage.duration_ms >= 0

    def test_pipeline_determinism(self, minimal_config: PipelineConfig) -> None:
        """Test that pipeline produces deterministic results."""
        result1 = run_pipeline(minimal_config)
        result2 = run_pipeline(minimal_config)

        assert len(result1.segments) == len(result2.segments)
        assert len(result1.profiles) == len(result2.profiles)

        # Segment sizes should match
        sizes1 = sorted(s.size for s in result1.segments)
        sizes2 = sorted(s.size for s in result2.segments)
        assert sizes1 == sizes2


# =============================================================================
# PIPELINE RESULT TESTS
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_actionable_segments(self, full_config: PipelineConfig) -> None:
        """Test actionable_segments property."""
        result = run_pipeline(full_config)

        actionable = result.actionable_segments
        assert isinstance(actionable, list)
        # All segments should have actionability evaluations
        for seg in actionable:
            assert result.actionability_evaluations[seg.segment_id].is_actionable is True

    def test_valid_segments(self, full_config: PipelineConfig) -> None:
        """Test valid_segments property."""
        result = run_pipeline(full_config)

        valid = result.valid_segments
        assert isinstance(valid, list)
        for seg in valid:
            assert result.validation_results[seg.segment_id].is_valid is True

    def test_production_ready_segments(self, full_config: PipelineConfig) -> None:
        """Test production_ready_segments property."""
        result = run_pipeline(full_config)

        ready = result.production_ready_segments
        assert isinstance(ready, list)

        # Production ready should be subset of both valid and actionable
        for seg in ready:
            assert seg in result.valid_segments
            assert seg in result.actionable_segments

    def test_get_segment_details(self, full_config: PipelineConfig) -> None:
        """Test get_segment_details method."""
        result = run_pipeline(full_config)

        segment = result.segments[0]
        details = result.get_segment_details(segment.segment_id)

        assert details is not None
        assert details["segment"] == segment
        assert "robustness" in details
        assert "viability" in details
        assert "actionability" in details
        assert "explanation" in details
        assert "validation" in details

    def test_get_segment_details_not_found(self, minimal_config: PipelineConfig) -> None:
        """Test get_segment_details with invalid ID."""
        result = run_pipeline(minimal_config)

        details = result.get_segment_details("nonexistent_id")
        assert details is None

    def test_get_summary(self, full_config: PipelineConfig) -> None:
        """Test get_summary method."""
        result = run_pipeline(full_config)

        summary = result.get_summary()

        assert "total_customers" in summary
        assert "total_segments" in summary
        assert "valid_segments" in summary
        assert "actionable_segments" in summary
        assert "production_ready_segments" in summary
        assert "total_duration_ms" in summary
        assert "success" in summary
        assert "stages" in summary


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_quick_segmentation(self) -> None:
        """Test quick_segmentation function."""
        result = quick_segmentation(
            n_customers=100,
            n_clusters=3,
            seed=42,
        )

        assert result.success is True
        assert len(result.segments) == 3
        assert len(result.profiles) > 0

    def test_quick_segmentation_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Test quick_segmentation with verbose output."""
        result = quick_segmentation(
            n_customers=50,
            n_clusters=2,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "[Pipeline]" in captured.out
        assert result.success is True

    def test_run_pipeline_on_dataset(self, small_dataset: SyntheticDataset) -> None:
        """Test run_pipeline_on_dataset function."""
        result = run_pipeline_on_dataset(
            small_dataset,
            n_clusters=5,
        )

        assert result.success is True
        assert len(result.segments) == 5

    def test_get_pipeline_metrics(self, full_config: PipelineConfig) -> None:
        """Test get_pipeline_metrics function."""
        result = run_pipeline(full_config)
        metrics = get_pipeline_metrics(result)

        assert "success" in metrics
        assert "total_duration_ms" in metrics
        assert "n_profiles" in metrics
        assert "n_segments" in metrics
        assert "n_valid_segments" in metrics
        assert "n_actionable_segments" in metrics
        assert "n_production_ready" in metrics

        # Should include stage timings
        assert any("stage_" in key for key in metrics.keys())

    def test_format_pipeline_summary(self, full_config: PipelineConfig) -> None:
        """Test format_pipeline_summary function."""
        result = run_pipeline(full_config)
        summary = format_pipeline_summary(result)

        assert "SEGMENTATION PIPELINE RESULTS" in summary
        assert "SUCCESS" in summary
        assert "Customer profiles:" in summary
        assert "Total segments:" in summary

    def test_export_results_to_dict(self, full_config: PipelineConfig) -> None:
        """Test export_results_to_dict function."""
        result = run_pipeline(full_config)
        export = export_results_to_dict(result)

        assert export["success"] is True
        assert export["n_profiles"] > 0
        assert len(export["segments"]) == len(result.segments)

        # Check segment data
        seg_data = export["segments"][0]
        assert "segment_id" in seg_data
        assert "name" in seg_data
        assert "size" in seg_data
        assert "total_clv" in seg_data
        assert "robustness" in seg_data
        assert "validation" in seg_data
        assert "actionability" in seg_data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in pipeline."""

    def test_empty_events_returns_failed_result(self) -> None:
        """Test that empty events returns failed result."""
        config = PipelineConfig()
        result = run_pipeline(config, events=[], id_history=[])

        # Should fail gracefully during profile building
        assert result.success is False
        assert result.error_message is not None

    def test_failed_result_has_partial_stages(self) -> None:
        """Test that failed pipeline has stage results up to failure."""
        config = PipelineConfig()
        result = run_pipeline(config, events=[], id_history=[])

        # Should have at least data acquisition and ID resolution
        assert len(result.stage_results) >= 2
        assert result.stage_results[0].stage_name == "Data Acquisition"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_integration(self) -> None:
        """Test complete pipeline integration."""
        config = PipelineConfig(
            n_customers=300,
            n_clusters=5,
            auto_select_k=False,
            run_sensitivity=True,
            generate_report=True,
            verbose=False,
        )

        result = run_pipeline(config)

        # Basic assertions
        assert result.success is True
        assert len(result.profiles) > 200  # Should have most customers
        assert len(result.segments) == 5

        # Check all segments have associated data
        for segment in result.segments:
            assert segment.segment_id in result.validation_results
            assert segment.segment_id in result.viabilities
            assert segment.segment_id in result.actionability_evaluations
            assert segment.segment_id in result.explanations
            assert segment.segment_id in result.robustness_scores

        # Check report
        assert result.report is not None
        assert len(result.report.segments) == 5

        # Check summary
        summary = result.get_summary()
        assert summary["total_customers"] == len(result.profiles)
        assert summary["total_segments"] == 5

    def test_pipeline_with_auto_k_selection(self) -> None:
        """Test pipeline with automatic k selection."""
        config = PipelineConfig(
            n_customers=200,
            auto_select_k=True,
            k_range=(3, 6),
            run_sensitivity=False,
            generate_report=False,
        )

        result = run_pipeline(config)

        assert result.success is True
        assert 3 <= len(result.segments) <= 6

    def test_pipeline_roundtrip_export(self) -> None:
        """Test that exported data can be used for analysis."""
        config = PipelineConfig(
            n_customers=150,
            n_clusters=4,
            auto_select_k=False,
            run_sensitivity=True,
            generate_report=True,
        )

        result = run_pipeline(config)
        export = export_results_to_dict(result)

        # Verify export is JSON-serializable
        import json

        json_str = json.dumps(export)
        reimported = json.loads(json_str)

        assert reimported["n_segments"] == len(result.segments)
        assert len(reimported["segments"]) == len(result.segments)

    def test_pipeline_multiple_runs_consistent(self) -> None:
        """Test multiple pipeline runs produce consistent results."""
        config = PipelineConfig(
            n_customers=100,
            n_clusters=3,
            data_seed=12345,
            cluster_seed=12345,
            auto_select_k=False,
            run_sensitivity=False,
            generate_report=False,
        )

        results = [run_pipeline(config) for _ in range(3)]

        # All runs should produce same number of profiles and segments
        n_profiles = [len(r.profiles) for r in results]
        n_segments = [len(r.segments) for r in results]

        assert len(set(n_profiles)) == 1
        assert len(set(n_segments)) == 1

        # Segment sizes should be consistent
        for i in range(1, len(results)):
            sizes_0 = sorted(s.size for s in results[0].segments)
            sizes_i = sorted(s.size for s in results[i].segments)
            assert sizes_0 == sizes_i
