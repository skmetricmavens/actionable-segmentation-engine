#!/usr/bin/env python3
"""
Demo script to run trait discovery on sample data.

Demonstrates the client-agnostic trait value analyzer that discovers
all product traits from event data and scores them for:
- Revenue impact (ANOVA F-test)
- Retention impact (Chi-square test)
- Personalization value (Entropy-based)
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.local_loader import load_local_data
from src.data.joiner import resolve_customer_merges
from src.features.profile_builder import build_profiles_batch
from src.analysis.trait_discovery import TraitValueAnalyzer, format_trait_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run trait discovery demo."""
    data_dir = Path(__file__).parent.parent / "data" / "samples"

    print("=" * 70)
    print("TRAIT VALUE DISCOVERY DEMO")
    print("=" * 70)
    print()

    # Step 1: Load sample data
    print("1. Loading sample data...")
    result = load_local_data(data_dir)
    print(f"   Loaded {len(result.events):,} events from {len(result.tables_loaded)} tables")
    print(f"   Unique customers: {result.unique_customers:,}")
    print(f"   Events by type: {result.events_by_type}")
    print()

    # Step 2: Build merge map from ID history
    print("2. Building customer ID merge map...")
    merge_map = resolve_customer_merges(result.id_history)
    print(f"   Merge map entries: {len(merge_map):,}")
    print()

    # Step 3: Build customer profiles
    print("3. Building customer profiles...")
    profiles = build_profiles_batch(result.events, merge_map=merge_map)
    print(f"   Built {len(profiles):,} customer profiles")
    print()

    # Step 4: Run trait discovery
    print("4. Running trait discovery...")
    analyzer = TraitValueAnalyzer(
        min_coverage=0.01,  # Lower threshold for demo
        max_cardinality=200,  # Allow more cardinality
        min_customers_per_value=3,  # Lower for demo
    )
    trait_result = analyzer.analyze(result.events, profiles)
    print(f"   Discovered {trait_result.total_fields_found} fields")
    print(f"   Scored {len(trait_result.traits)} usable traits")
    print()

    # Step 5: Print detailed report
    print("5. Trait Discovery Results:")
    print()
    print(format_trait_report(trait_result))

    # Step 6: Print detailed trait info
    print()
    print("DETAILED TRAIT SCORES")
    print("-" * 70)
    for i, trait in enumerate(trait_result.traits[:15], 1):
        print(f"\n{i}. {trait.trait_name} (path: {trait.trait_path})")
        print(f"   Type: {trait.trait_type}")
        print(f"   Overall Score: {trait.overall_score:.3f}")
        print(f"   Revenue Impact: {trait.revenue_impact:.3f} (p={trait.revenue_p_value:.4f})")
        print(f"   Retention Impact: {trait.retention_impact:.3f} (p={trait.retention_p_value:.4f})")
        print(f"   Personalization: {trait.personalization_value:.3f} (entropy={trait.entropy:.2f})")
        print(f"   Distinct Values: {trait.distinct_values}")
        print(f"   Customer Coverage: {trait.customer_coverage:.1%}")
        print(f"   Concentration: {trait.concentration:.1%}")
        if trait.recommended_uses:
            print(f"   Recommended Uses: {', '.join(trait.recommended_uses)}")
        if trait.top_revenue_values:
            print(f"   Top Revenue Values:")
            for val, rev in trait.top_revenue_values[:3]:
                print(f"      - {val}: ${rev:,.0f} avg revenue")

    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
