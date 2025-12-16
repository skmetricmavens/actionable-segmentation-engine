#!/usr/bin/env python3
"""
Script: collect_trait_feedback.py

Purpose: Interactive CLI for collecting user feedback on trait recommendations.

This script runs trait discovery on sample data and then guides the user
through rating each trait. Feedback is persisted and used to adjust
future trait scores.

Usage:
    python scripts/collect_trait_feedback.py
    python scripts/collect_trait_feedback.py --data-dir data/samples --top-n 10
    python scripts/collect_trait_feedback.py --show-summary
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trait_discovery import TraitDiscoveryResult, TraitValueAnalyzer, format_trait_report
from src.analysis.trait_feedback import FeedbackStore, TraitFeedback
from src.data.joiner import resolve_customer_merges
from src.data.local_loader import load_local_data
from src.features.profile_builder import build_profiles_batch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FEEDBACK_TYPES = {
    "u": "useful",
    "n": "not_useful",
    "o": "obvious",
    "i": "incorrect",
    "c": "needs_context",
}


def print_trait_summary(trait, index: int) -> None:
    """Print a summary of a trait for user review."""
    print(f"\n{'='*60}")
    print(f"TRAIT {index}: {trait.trait_name}")
    print(f"{'='*60}")

    # Score info
    score_str = f"{trait.overall_score:.2f}"
    if trait.adjusted_overall_score is not None:
        score_str += f" (adjusted: {trait.adjusted_overall_score:.2f})"
    print(f"Overall Score: {score_str}")
    print(f"Type: {trait.trait_type}")
    print(f"Coverage: {trait.customer_coverage:.0%}")
    print(f"Distinct Values: {trait.distinct_values}")
    print()

    # Impact scores
    print("IMPACT SCORES:")
    print(f"  Revenue: {trait.revenue_impact:.2f} (p={trait.revenue_p_value:.4f})")
    print(f"  Retention: {trait.retention_impact:.2f} (p={trait.retention_p_value:.4f})")
    print(f"  Personalization: {trait.personalization_value:.2f}")
    print()

    # Explanation if available
    if trait.explanation:
        print("EXPLANATION:")
        if trait.explanation.overall_summary:
            print(f"  Summary: {trait.explanation.overall_summary}")
        if trait.explanation.revenue_explanation:
            print(f"  Revenue: {trait.explanation.revenue_explanation}")
        if trait.explanation.retention_explanation:
            print(f"  Retention: {trait.explanation.retention_explanation}")
        if trait.explanation.personalization_explanation:
            print(f"  Personalization: {trait.explanation.personalization_explanation}")
        print()

        print(f"  Actionability: {trait.explanation.actionability_assessment}")

        if trait.explanation.caveats:
            print("  Caveats:")
            for caveat in trait.explanation.caveats:
                print(f"    - {caveat}")

        if trait.explanation.suggested_actions:
            print("  Suggested Actions:")
            for action in trait.explanation.suggested_actions[:3]:
                print(f"    - {action}")
        print()

    # Recommended uses
    if trait.recommended_uses:
        print(f"Recommended Uses: {', '.join(trait.recommended_uses)}")
    print()


def collect_feedback_interactive(
    result: TraitDiscoveryResult,
    store: FeedbackStore,
    top_n: int = 10,
) -> int:
    """
    Interactive CLI for rating traits.

    Args:
        result: TraitDiscoveryResult with scored traits
        store: FeedbackStore for persisting feedback
        top_n: Number of top traits to review

    Returns:
        Number of feedback entries collected
    """
    print("\n" + "=" * 70)
    print("TRAIT FEEDBACK COLLECTION")
    print("=" * 70)
    print()
    print("Rate each trait recommendation to improve future analyses.")
    print("Options:")
    print("  u = useful (this trait is valuable for segmentation/personalization)")
    print("  n = not useful (this trait doesn't provide actionable insights)")
    print("  o = obvious (correlation is true but trivial/expected)")
    print("  i = incorrect (the analysis seems wrong)")
    print("  c = needs context (might be useful with more information)")
    print("  s = skip (don't provide feedback for this trait)")
    print("  q = quit (stop collecting feedback)")
    print()

    collected = 0
    traits_to_review = result.traits[:top_n]

    for i, trait in enumerate(traits_to_review, 1):
        print_trait_summary(trait, i)

        while True:
            response = input(f"Rating for '{trait.trait_name}' [u/n/o/i/c/s/q]: ").strip().lower()

            if response == "q":
                print("\nStopping feedback collection.")
                return collected

            if response == "s":
                print("  Skipped.")
                break

            if response in FEEDBACK_TYPES:
                # Optionally collect reason
                reason = None
                if response in ("n", "o", "i"):
                    reason_input = input("  Reason (optional, press Enter to skip): ").strip()
                    if reason_input:
                        reason = reason_input

                # Create and store feedback
                feedback = TraitFeedback(
                    feedback_id=str(uuid.uuid4()),
                    trait_name=trait.trait_name,
                    trait_path=trait.trait_path,
                    timestamp=datetime.now(timezone.utc),
                    feedback_type=FEEDBACK_TYPES[response],
                    reason=reason,
                )
                store.add_feedback(feedback)
                collected += 1
                print(f"  Recorded: {FEEDBACK_TYPES[response]}")
                break
            else:
                print("  Invalid input. Please enter u, n, o, i, c, s, or q.")

    return collected


def show_feedback_summary(store: FeedbackStore) -> None:
    """Display summary of accumulated feedback."""
    summary = store.get_summary()

    print("\n" + "=" * 70)
    print("FEEDBACK SUMMARY")
    print("=" * 70)
    print()

    print(f"Total Feedback Entries: {summary['total_feedback']}")
    print()

    if summary["by_type"]:
        print("By Type:")
        for feedback_type, count in sorted(summary["by_type"].items()):
            print(f"  {feedback_type}: {count}")
        print()

    if summary["by_trait"]:
        print("By Trait (top 10):")
        sorted_traits = sorted(summary["by_trait"].items(), key=lambda x: x[1], reverse=True)
        for trait_name, count in sorted_traits[:10]:
            print(f"  {trait_name}: {count}")
        print()

    if summary["learned_patterns"]:
        print("Learned Patterns:")
        for pattern, data in summary["learned_patterns"].items():
            adjustment = data["adjustment"]
            direction = "boost" if adjustment > 1 else "penalty"
            pct = abs(1 - adjustment) * 100
            print(f"  '{pattern}': {pct:.0f}% {direction} (from {data['sample_size']} samples)")
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect user feedback on trait recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run interactive feedback collection
    python scripts/collect_trait_feedback.py

    # Review top 15 traits
    python scripts/collect_trait_feedback.py --top-n 15

    # Show feedback summary only
    python scripts/collect_trait_feedback.py --show-summary

    # Use custom data directory
    python scripts/collect_trait_feedback.py --data-dir data/my_samples
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/samples"),
        help="Directory containing sample parquet files",
    )
    parser.add_argument(
        "--feedback-path",
        type=Path,
        default=Path("data/feedback/trait_feedback.json"),
        help="Path to feedback storage file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top traits to review (default: 10)",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show feedback summary and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize feedback store
    store = FeedbackStore(storage_path=args.feedback_path)

    # Show summary if requested
    if args.show_summary:
        show_feedback_summary(store)
        return 0

    # Load data and run trait discovery
    print("Loading data...")
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    try:
        data_result = load_local_data(args.data_dir)
        print(f"Loaded {len(data_result.events):,} events")

        merge_map = resolve_customer_merges(data_result.id_history)
        profiles = build_profiles_batch(data_result.events, merge_map=merge_map)
        print(f"Built {len(profiles):,} customer profiles")

        # Run trait discovery with feedback store
        print("Running trait discovery...")
        analyzer = TraitValueAnalyzer(
            min_coverage=0.01,
            max_cardinality=200,
            min_customers_per_value=3,
        )
        trait_result = analyzer.analyze(
            data_result.events,
            profiles,
            feedback_store=store,
            generate_explanations=True,
        )
        print(f"Discovered {len(trait_result.traits)} traits")

    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Collect feedback
    collected = collect_feedback_interactive(trait_result, store, top_n=args.top_n)

    print()
    print(f"Collected {collected} feedback entries.")
    print(f"Feedback saved to: {args.feedback_path}")

    # Show updated summary
    show_feedback_summary(store)

    return 0


if __name__ == "__main__":
    sys.exit(main())
