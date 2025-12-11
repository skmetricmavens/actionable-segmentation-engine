#!/usr/bin/env python3
"""
Run the segmentation pipeline on locally stored sample data.

Usage:
    python scripts/run_local_pipeline.py [--data-dir DATA_DIR] [--clusters N] [--verbose]

Example:
    python scripts/run_local_pipeline.py --data-dir data/samples --clusters 6 --verbose
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.local_loader import load_events_only, LocalDataLoader
from src.pipeline import (
    run_pipeline,
    PipelineConfig,
    format_pipeline_summary,
    export_results_to_dict,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation pipeline on local sample data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/samples",
        help="Directory containing parquet sample files (default: data/samples)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="Number of clusters/segments to create (default: 6)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=3,
        help="Minimum events per customer to include (default: 3)",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Auto-select optimal number of clusters",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results JSON (optional)",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (faster)",
    )

    args = parser.parse_args()

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run sample extraction first or specify a different --data-dir")
        sys.exit(1)

    # Load data
    print("=" * 60)
    print("LOADING LOCAL DATA")
    print("=" * 60)

    loader = LocalDataLoader(data_dir)
    metadata = loader.load_metadata()

    if metadata:
        print(f"Dataset: {metadata.get('project_id')}.{metadata.get('dataset_id')}")
        print(f"Date range: {metadata.get('start_date')} to {metadata.get('end_date')}")
        print(f"Unique customers: {metadata.get('unique_customers')}")

    events, id_history = load_events_only(data_dir)
    print(f"Loaded {len(events):,} events and {len(id_history):,} ID history records")

    # Configure pipeline
    print("\n" + "=" * 60)
    print("RUNNING SEGMENTATION PIPELINE")
    print("=" * 60)

    config = PipelineConfig(
        data_source="synthetic",  # We're providing our own data
        min_events_per_customer=args.min_events,
        n_clusters=args.clusters if not args.auto_k else None,
        auto_select_k=args.auto_k,
        run_sensitivity=not args.no_sensitivity,
        run_integrated_analysis=True,
        generate_report=True,
        verbose=args.verbose,
    )

    # Run pipeline
    result = run_pipeline(
        config=config,
        events=events,
        id_history=id_history,
    )

    # Print results
    print("\n" + format_pipeline_summary(result))

    if result.success:
        print("\n" + "=" * 60)
        print("SEGMENT DETAILS")
        print("=" * 60)

        for segment in sorted(result.segments, key=lambda s: float(s.total_clv), reverse=True):
            rob = result.robustness_scores.get(segment.segment_id)
            act = result.actionability_evaluations.get(segment.segment_id)

            print(f"\n{segment.name}")
            print(f"  Size: {segment.size:,} customers ({segment.size / len(result.profiles) * 100:.1f}%)")
            print(f"  Total CLV: ${float(segment.total_clv):,.2f}")
            print(f"  Avg AOV: ${float(segment.avg_order_value):,.2f}")

            if rob:
                print(f"  Robustness: {rob.overall_robustness:.2f} ({rob.robustness_tier.value})")

            if act:
                status = "✓" if act.is_actionable else "✗"
                print(f"  Actionable: {status}")
                if act.is_actionable and act.recommended_action:
                    print(f"  → {act.recommended_action[:100]}...")

            if segment.defining_traits:
                print(f"  Traits: {', '.join(segment.defining_traits[:3])}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_clv = sum(float(s.total_clv) for s in result.segments)
        actionable_count = len(result.actionable_segments)

        print(f"Total customer profiles: {len(result.profiles):,}")
        print(f"Total segments: {len(result.segments)}")
        print(f"Actionable segments: {actionable_count}")
        print(f"Total CLV coverage: ${total_clv:,.2f}")

        if result.integrated_analysis:
            ia = result.integrated_analysis
            print(f"Whitespace opportunity: ${float(ia.total_whitespace_opportunity):,.2f}")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            results_dict = export_results_to_dict(result)
            results_dict["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "data_dir": str(data_dir),
                "config": {
                    "clusters": args.clusters,
                    "min_events": args.min_events,
                    "auto_k": args.auto_k,
                },
            }

            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2, default=str)

            print(f"\nResults saved to: {output_path}")

    else:
        print(f"\nPIPELINE FAILED: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
