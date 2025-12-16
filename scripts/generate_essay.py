#!/usr/bin/env python3
"""
Generate Pudding-style visual essays from customer segmentation data.

This script generates interactive HTML essays that tell the story of your
customer data through scrollytelling visualizations.

Usage:
    # Generate all essays
    python scripts/generate_essay.py --data-dir data/samples --output-dir output/essays

    # Generate specific essays
    python scripts/generate_essay.py --essays loyalty,champions

    # Apply manual narrative overrides
    python scripts/generate_essay.py --overrides config/essay_overrides.yaml

    # Export auto-generated narratives (to edit and use as overrides)
    python scripts/generate_essay.py --export-narratives config/narratives_draft.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.local_loader import LocalDataLoader
from src.data.joiner import resolve_customer_merges
from src.data.schemas import CustomerProfile
from src.essays.base import Essay, EssayConfig, KeyInsight
from src.essays.data_queries import query_all_essay_data
from src.essays.narratives import export_narratives_to_dict, generate_all_narratives
from src.essays.overrides import (
    apply_overrides,
    create_template_override_file,
    export_narratives_to_yaml,
    load_overrides,
)
from src.essays.renderers.pudding_html import render_pudding_essay, save_pudding_essay
from src.essays.sections import (
    create_champion_fragility_section,
    create_churn_prediction_section,
    create_loyalty_journey_section,
    create_new_customer_quality_section,
    create_segment_robustness_section,
    create_trait_insights_section,
    create_whitespace_opportunities_section,
)
from src.features.profile_builder import build_profiles_batch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Pudding-style visual essays from customer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/samples",
        help="Directory containing customer profile data (default: data/samples)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/essays",
        help="Directory for output HTML files (default: output/essays)",
    )

    parser.add_argument(
        "--essays",
        type=str,
        default="all",
        help=(
            "Comma-separated list of essays to generate. "
            "Options: loyalty, champions, quality, churn, whitespace, traits, robustness, all (default: all)"
        ),
    )

    parser.add_argument(
        "--audience",
        type=str,
        choices=["executive", "marketing", "technical", "all"],
        default="all",
        help="Target audience layer to emphasize (default: all)",
    )

    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="Path to YAML file with narrative overrides",
    )

    parser.add_argument(
        "--export-narratives",
        type=str,
        default=None,
        help="Export auto-generated narratives to YAML file for editing",
    )

    parser.add_argument(
        "--create-template",
        type=str,
        default=None,
        help="Create a template override file with all narrative fields",
    )

    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Exclude technical appendix from output",
    )

    parser.add_argument(
        "--embed-assets",
        action="store_true",
        help="Embed CSS/JS assets instead of linking to CDN",
    )

    parser.add_argument(
        "--profile-file",
        type=str,
        default=None,
        help="Specific CSV file with customer profiles (overrides --data-dir)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_customer_data(args: argparse.Namespace) -> list[CustomerProfile]:
    """Load customer profiles from parquet data files.

    Args:
        args: Command line arguments containing data_dir and verbose flag

    Returns:
        List of CustomerProfile objects built from parquet event data
    """
    data_dir = Path(args.data_dir)

    # Check if parquet files exist
    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        if args.verbose:
            print(f"No parquet files found in {data_dir}, generating sample data...")
        return _generate_sample_profiles()

    if args.verbose:
        print(f"Loading data from {data_dir}...")

    # Load events from parquet files
    loader = LocalDataLoader(data_dir)
    result = loader.load()

    if result.errors:
        print(f"Warning: Errors during data loading: {result.errors}")

    if args.verbose:
        print(f"  Loaded {len(result.events):,} events from {len(result.tables_loaded)} tables")
        print(f"  ID history records: {len(result.id_history):,}")

    if not result.events:
        print("No events found in parquet files, generating sample data...")
        return _generate_sample_profiles()

    # Build merge map for customer ID unification
    merge_map = resolve_customer_merges(result.id_history)
    if args.verbose:
        print(f"  Merge map entries: {len(merge_map):,}")

    # Build customer profiles from events
    if args.verbose:
        print("Building customer profiles...")

    profiles = build_profiles_batch(
        result.events,
        merge_map=merge_map,
        min_events=1,
    )

    if args.verbose:
        print(f"  Built {len(profiles):,} customer profiles")

        # Show behavior distribution
        from collections import Counter
        behavior_counts = Counter(p.behavior_type.value for p in profiles)
        print("  Behavior distribution:")
        for bt, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
            print(f"    {bt}: {count:,} ({count/len(profiles)*100:.1f}%)")

    return profiles


def _generate_sample_profiles() -> list[CustomerProfile]:
    """Generate sample customer profiles for demo purposes when no parquet data available."""
    from datetime import timedelta
    from decimal import Decimal
    from random import gauss, random, randint

    from src.data.schemas import BehaviorType

    profiles = []
    behavior_weights = [
        (BehaviorType.NEW, 0.15),
        (BehaviorType.ONE_TIME, 0.35),
        (BehaviorType.IRREGULAR, 0.25),
        (BehaviorType.REGULAR, 0.15),
        (BehaviorType.LONG_CYCLE, 0.10),
    ]

    now = datetime.now()

    for i in range(1000):
        # Randomly select behavior type based on weights
        r = random()
        cumulative = 0
        behavior = BehaviorType.ONE_TIME
        for bt, weight in behavior_weights:
            cumulative += weight
            if r < cumulative:
                behavior = bt
                break

        # Generate profile based on behavior type
        if behavior == BehaviorType.NEW:
            purchases = 1
            days_since = randint(1, 30)
            revenue = Decimal(str(round(max(10, gauss(50, 20)), 2)))
        elif behavior == BehaviorType.ONE_TIME:
            purchases = 1
            days_since = randint(60, 365)
            revenue = Decimal(str(round(max(10, gauss(75, 30)), 2)))
        elif behavior == BehaviorType.IRREGULAR:
            purchases = randint(2, 4)
            days_since = randint(30, 120)
            revenue = Decimal(str(round(max(20, gauss(150, 50)) * purchases, 2)))
        elif behavior == BehaviorType.REGULAR:
            purchases = randint(5, 20)
            days_since = randint(1, 45)
            revenue = Decimal(str(round(max(30, gauss(100, 30)) * purchases, 2)))
        else:  # LONG_CYCLE
            purchases = randint(3, 8)
            days_since = randint(45, 180)
            revenue = Decimal(str(round(max(50, gauss(200, 50)) * purchases, 2)))

        # Calculate derived fields
        first_seen = now - timedelta(days=randint(90, 730))
        last_seen = now - timedelta(days=days_since)
        churn_risk = min(1.0, max(0.0, days_since / 100 + random() * 0.2))

        profile = CustomerProfile(
            internal_customer_id=f"CUST_{i:05d}",
            behavior_type=behavior,
            total_purchases=purchases,
            total_revenue=max(Decimal("0"), revenue),
            avg_order_value=revenue / purchases if purchases > 0 else Decimal("0"),
            days_since_last_purchase=days_since,
            purchase_frequency_per_month=purchases / max(1, (now - first_seen).days / 30),
            clv_estimate=revenue * Decimal(str(1.5 + random())),
            churn_risk_score=churn_risk,
            first_seen=first_seen,
            last_seen=last_seen,
        )
        profiles.append(profile)

    return profiles


def generate_essays(args: argparse.Namespace) -> None:
    """Main essay generation workflow."""
    # Create config
    config = EssayConfig(
        essays=args.essays.split(",") if args.essays != "all" else ["all"],
        audience=args.audience,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        overrides_path=args.overrides,
        include_appendix=not args.no_appendix,
        embed_assets=args.embed_assets,
    )

    # Load data
    print("Loading customer data...")
    profiles = load_customer_data(args)
    print(f"Loaded {len(profiles)} customer profiles")

    # Query essay data
    print("Analyzing data for essays...")
    data_bundle = query_all_essay_data(profiles)

    # Generate narratives
    print("Generating narratives...")
    narratives = generate_all_narratives(data_bundle)

    # Apply overrides if provided
    if config.overrides_path:
        print(f"Applying overrides from: {config.overrides_path}")
        overrides = load_overrides(config.overrides_path)
        narratives = apply_overrides(narratives, overrides)

    # Export narratives if requested
    if args.export_narratives:
        export_path = Path(args.export_narratives)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_narratives_to_yaml(narratives, str(export_path))
        print(f"Exported narratives to: {export_path}")

    # Create template if requested
    if args.create_template:
        template_path = Path(args.create_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        create_template_override_file(str(template_path))
        print(f"Created template override file: {template_path}")

    # Determine which essays to generate
    essay_map = {
        "loyalty": "loyalty_journey",
        "champions": "champion_fragility",
        "quality": "new_customer_quality",
        "churn": "churn_prediction",
        "whitespace": "whitespace_opportunities",
        "traits": "trait_insights",
        "robustness": "segment_robustness",
    }

    essays_to_generate = []
    if "all" in config.essays:
        essays_to_generate = list(essay_map.keys())
    else:
        essays_to_generate = [e.strip() for e in config.essays if e.strip() in essay_map]

    if not essays_to_generate:
        print("No valid essays specified. Use: loyalty, champions, quality, churn, whitespace, traits, robustness, or all")
        return

    # Generate essay sections
    print(f"Generating essays: {', '.join(essays_to_generate)}")
    sections = []

    if "loyalty" in essays_to_generate:
        sections.append(create_loyalty_journey_section(data_bundle.loyalty_journey, narratives))

    if "champions" in essays_to_generate:
        sections.append(create_champion_fragility_section(data_bundle.champion_fragility, narratives))

    if "quality" in essays_to_generate:
        sections.append(create_new_customer_quality_section(data_bundle.new_customer_quality, narratives))

    if "churn" in essays_to_generate:
        sections.append(create_churn_prediction_section(data_bundle.churn_prediction, narratives))

    if "whitespace" in essays_to_generate and data_bundle.whitespace:
        sections.append(create_whitespace_opportunities_section(data_bundle.whitespace, narratives))

    if "traits" in essays_to_generate and data_bundle.trait_discovery:
        sections.append(create_trait_insights_section(data_bundle.trait_discovery, narratives))

    if "robustness" in essays_to_generate and data_bundle.sensitivity:
        sections.append(create_segment_robustness_section(data_bundle.sensitivity, narratives))

    # Build key insights for banner
    key_insights = _extract_key_insights(sections)

    # Create essay
    essay = Essay(
        essay_id="customer-segmentation-story",
        title="Your Customer Segmentation Story",
        subtitle="A data-driven narrative of how your customers behave",
        sections=sections,
        key_insights=key_insights,
        generated_at=datetime.now(),
        data_period_start=data_bundle.data_period_start,
        data_period_end=data_bundle.data_period_end,
        customer_count=data_bundle.total_customers,
        event_count=data_bundle.total_events,
    )

    # Render HTML with Pudding-style scrollytelling
    print("Rendering Pudding-style HTML...")
    html = render_pudding_essay(essay, config)

    # Save output
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "index.html"
    save_pudding_essay(html, str(output_file))

    print(f"\nEssay generated successfully!")
    print(f"Output: {output_file.absolute()}")
    print(f"\nOpen in browser: file://{output_file.absolute()}")


def _extract_key_insights(sections: list) -> list[KeyInsight]:
    """Extract key insights from sections for executive banner."""
    insights = []

    for section in sections:
        metrics = section.key_metrics

        if section.section_id == "loyalty-journey":
            threshold = metrics.get("loyalty_threshold", 4)
            regular_pct = metrics.get("regular_percentage", 0)
            insights.append(KeyInsight(
                text=f"Order #{threshold} is the loyalty threshold",
                metric_value=f"#{threshold}",
                metric_label="loyalty threshold",
                category="loyalty",
            ))
            insights.append(KeyInsight(
                text=f"{regular_pct:.0f}% become loyal regulars",
                metric_value=f"{regular_pct:.0f}%",
                metric_label="become regulars",
                category="loyalty",
            ))

        elif section.section_id == "champion-fragility":
            fragile_pct = metrics.get("fragile_percentage", 0)
            revenue = metrics.get("revenue_at_risk", 0)
            insights.append(KeyInsight(
                text=f"{fragile_pct:.0f}% of Champions are fragile",
                metric_value=f"{fragile_pct:.0f}%",
                metric_label="Champions at risk",
                category="risk",
            ))
            if revenue > 0:
                insights.append(KeyInsight(
                    text=f"${revenue:,.0f} revenue at risk",
                    metric_value=f"${revenue:,.0f}",
                    metric_label="at risk",
                    category="risk",
                ))

        elif section.section_id == "new-customer-quality":
            churn = metrics.get("churn_rate", 0)
            insights.append(KeyInsight(
                text=f"{churn:.0f}% of new customers never return",
                metric_value=f"{churn:.0f}%",
                metric_label="never return",
                category="acquisition",
            ))

        elif section.section_id == "churn-prediction":
            warning_days = metrics.get("warning_days", 0)
            at_risk = metrics.get("at_risk_count", 0)
            insights.append(KeyInsight(
                text=f"Warning signs appear {warning_days:.0f} days before churn",
                metric_value=f"{warning_days:.0f}",
                metric_label="days warning",
                category="churn",
            ))
            insights.append(KeyInsight(
                text=f"{at_risk:,} customers currently at risk",
                metric_value=f"{at_risk:,}",
                metric_label="at risk",
                category="churn",
            ))

        elif section.section_id == "whitespace-opportunities":
            total_value = metrics.get("total_opportunity_value", 0)
            total_lookalikes = metrics.get("total_lookalikes", 0)
            insights.append(KeyInsight(
                text=f"${total_value:,.0f} in cross-sell opportunities",
                metric_value=f"${total_value:,.0f}",
                metric_label="opportunity",
                category="growth",
            ))
            insights.append(KeyInsight(
                text=f"{total_lookalikes:,} lookalike customers identified",
                metric_value=f"{total_lookalikes:,}",
                metric_label="lookalikes",
                category="growth",
            ))

        elif section.section_id == "trait-insights":
            significant = metrics.get("significant_traits", 0)
            total = metrics.get("total_traits", 0)
            insights.append(KeyInsight(
                text=f"{significant} of {total} traits drive business outcomes",
                metric_value=f"{significant}/{total}",
                metric_label="impactful traits",
                category="insights",
            ))

        elif section.section_id == "segment-robustness":
            high_pct = metrics.get("high_robustness_pct", 0)
            insights.append(KeyInsight(
                text=f"{high_pct:.0f}% of segments are highly robust",
                metric_value=f"{high_pct:.0f}%",
                metric_label="robust segments",
                category="quality",
            ))

    # Return top 6 most impactful insights (expanded from 4 to include new essays)
    return insights[:6]


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        print("Essay Generator")
        print("=" * 40)

    try:
        generate_essays(args)
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
