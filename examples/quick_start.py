#!/usr/bin/env python3
"""Quick start example for the Actionable Segmentation Engine.

Run this script to see the pipeline in action with synthetic data.

Usage:
    python examples/quick_start.py
"""

from src.pipeline import quick_segmentation, format_pipeline_summary


def main() -> None:
    """Run a quick segmentation demo."""
    print("=" * 60)
    print("Actionable Segmentation Engine - Quick Start Demo")
    print("=" * 60)

    # Run segmentation with synthetic data
    print("\nRunning pipeline with 500 customers and 5 clusters...")
    result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

    # Print summary
    print("\n" + format_pipeline_summary(result))

    # Show segment details
    print("\n" + "=" * 60)
    print("SEGMENT DETAILS")
    print("=" * 60)

    for segment in result.segments:
        print(f"\n{segment.name}")
        print("-" * 40)
        print(f"  Size: {segment.size} customers")
        print(f"  Total CLV: ${float(segment.total_clv):,.2f}")
        print(f"  Avg CLV: ${float(segment.avg_clv):,.2f}")
        print(f"  Traits: {', '.join(segment.defining_traits[:3])}")

        # Robustness
        robustness = result.robustness_scores.get(segment.segment_id)
        if robustness:
            print(f"  Robustness: {robustness.overall_robustness:.2f} ({robustness.robustness_tier.value})")

        # Actionability
        evaluation = result.actionability_evaluations.get(segment.segment_id)
        if evaluation:
            print(f"  Actionable: {evaluation.is_actionable}")
            if evaluation.recommended_action:
                print(f"  Recommended: {evaluation.recommended_action}")

    # Production-ready segments
    print("\n" + "=" * 60)
    print("PRODUCTION-READY SEGMENTS")
    print("=" * 60)

    if result.production_ready_segments:
        for segment in result.production_ready_segments:
            explanation = result.explanations.get(segment.segment_id)
            print(f"\n{segment.name}")
            if explanation:
                print(f"  Campaign: {explanation.recommended_campaign}")
                print(f"  ROI: {explanation.expected_roi}")
    else:
        print("\nNo segments passed all validation criteria.")
        print("Try adjusting validation thresholds or using more data.")


if __name__ == "__main__":
    main()
