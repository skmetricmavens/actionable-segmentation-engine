#!/usr/bin/env python3
"""Custom configuration example for the Actionable Segmentation Engine.

This example demonstrates how to customize pipeline behavior.

Usage:
    python examples/custom_config.py
"""

from decimal import Decimal

from src.pipeline import PipelineConfig, run_pipeline, format_pipeline_summary
from src.segmentation.segment_validator import ValidationCriteria


def main() -> None:
    """Run pipeline with custom configuration."""
    print("=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    # Define strict validation criteria
    strict_criteria = ValidationCriteria(
        min_segment_size=20,
        max_segment_size_pct=0.4,
        min_total_clv=Decimal("5000"),
        min_avg_clv=Decimal("100"),
        min_feature_stability=0.5,
        min_overall_robustness=0.5,
        min_expected_roi=1.0,
    )

    # Configure pipeline
    config = PipelineConfig(
        # Data generation
        n_customers=1000,
        data_seed=42,
        merge_probability=0.15,

        # Clustering - auto select k
        auto_select_k=True,
        k_range=(4, 8),
        cluster_seed=42,

        # Sensitivity analysis
        run_sensitivity=True,
        include_sampling_stability=True,

        # Validation
        validation_criteria=strict_criteria,

        # Output
        generate_report=True,
        use_llm=False,  # Use mock LLM
        verbose=True,
    )

    print("\nConfiguration:")
    print(f"  Customers: {config.n_customers}")
    print(f"  Auto k-selection: {config.auto_select_k} (range {config.k_range})")
    print(f"  Sensitivity analysis: {config.run_sensitivity}")
    print(f"  Min robustness: {strict_criteria.min_overall_robustness}")

    # Run pipeline
    print("\nRunning pipeline...")
    result = run_pipeline(config)

    # Print summary
    print("\n" + format_pipeline_summary(result))

    # Show validation results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for segment in result.segments:
        validation = result.validation_results.get(segment.segment_id)
        if validation:
            status = "VALID" if validation.is_valid else "REJECTED"
            print(f"\n{segment.name}: {status}")
            if not validation.is_valid:
                for reason in validation.rejection_reasons:
                    print(f"  - {reason}")

    # Show stage timing
    print("\n" + "=" * 60)
    print("STAGE TIMING")
    print("=" * 60)

    total_ms = 0
    for stage in result.stage_results:
        status = "OK" if stage.success else "FAILED"
        print(f"  {stage.stage_name}: {stage.duration_ms:.1f}ms [{status}]")
        total_ms += stage.duration_ms

    print(f"\n  Total: {total_ms:.1f}ms")


if __name__ == "__main__":
    main()
