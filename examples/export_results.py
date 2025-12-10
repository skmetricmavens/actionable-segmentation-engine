#!/usr/bin/env python3
"""Export results example for the Actionable Segmentation Engine.

This example demonstrates how to export segmentation results.

Usage:
    python examples/export_results.py
"""

import json
from pathlib import Path

from src.pipeline import quick_segmentation, export_results_to_dict


def main() -> None:
    """Run pipeline and export results."""
    print("=" * 60)
    print("Export Results Example")
    print("=" * 60)

    # Run segmentation
    print("\nRunning pipeline...")
    result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

    # Export to dictionary
    print("\nExporting results...")
    data = export_results_to_dict(result)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save full results
    output_file = output_dir / "segmentation_results.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Full results: {output_file}")

    # Save segment summary
    summary_file = output_dir / "segments_summary.json"
    segment_summary = {
        "total_segments": len(result.segments),
        "valid_segments": len(result.valid_segments),
        "actionable_segments": len(result.actionable_segments),
        "production_ready": len(result.production_ready_segments),
        "segments": [
            {
                "id": s.segment_id,
                "name": s.name,
                "size": s.size,
                "total_clv": str(s.total_clv),
                "avg_clv": str(s.avg_clv),
                "traits": s.defining_traits[:3],
                "robustness": result.robustness_scores[s.segment_id].overall_robustness
                if s.segment_id in result.robustness_scores
                else None,
                "is_actionable": result.actionability_evaluations[s.segment_id].is_actionable
                if s.segment_id in result.actionability_evaluations
                else None,
            }
            for s in result.segments
        ],
    }
    with open(summary_file, "w") as f:
        json.dump(segment_summary, f, indent=2)
    print(f"  Segment summary: {summary_file}")

    # Save customer-segment mapping
    mapping_file = output_dir / "customer_segments.json"
    customer_mapping = {
        "mappings": [
            {"customer_id": cid, "segment_id": s.segment_id, "segment_name": s.name}
            for s in result.segments
            for cid in s.customer_ids
        ]
    }
    with open(mapping_file, "w") as f:
        json.dump(customer_mapping, f, indent=2)
    print(f"  Customer mapping: {mapping_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Total segments: {len(result.segments)}")
    print(f"  Production-ready: {len(result.production_ready_segments)}")
    print(f"  Total customers: {len(result.profiles)}")
    print(f"\nFiles saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
