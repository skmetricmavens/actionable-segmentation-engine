"""
Tests for essay data integrity.

These tests validate that data used in essay visualizations is:
1. Mathematically consistent (counts sum correctly, percentages are bounded)
2. Cross-essay consistent (same metrics across essays match)
3. Properly contextualized (percentages include baselines)

These tests catch data errors before they reach users, preventing
confusing visualizations like "42% of what?".
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.data.schemas import BehaviorType, CustomerProfile
from src.essays.charts.funnel import create_funnel_spec
from src.essays.charts.sankey import create_sankey_spec
from src.essays.data_queries import (
    EssayDataBundle,
    query_all_essay_data,
    query_loyalty_journey_data,
    query_new_customer_quality_data,
)
from src.essays.sections import (
    create_champion_fragility_section,
    create_loyalty_journey_section,
    create_new_customer_quality_section,
    create_segment_robustness_section,
)


# =============================================================================
# FIXTURES - Known Totals for Validation
# =============================================================================


@pytest.fixture
def known_totals() -> dict:
    """Expected values for validation testing."""
    return {
        "total_customers": 1000,
        "new_customers": 200,
        "one_time_customers": 300,
        "irregular_customers": 250,
        "regular_customers": 150,  # Champions
        "long_cycle_customers": 100,
        "customers_with_2_plus_purchases": 500,  # irregular + regular + long_cycle
        "customers_with_3_plus_purchases": 350,  # subset with 3+
    }


@pytest.fixture
def sample_profiles_with_known_totals(known_totals: dict) -> list[CustomerProfile]:
    """Create sample profiles with known, predictable totals."""
    now = datetime.now()
    profiles = []

    # New customers (200)
    for i in range(known_totals["new_customers"]):
        profiles.append(CustomerProfile(
            internal_customer_id=f"NEW_{i:03d}",
            behavior_type=BehaviorType.NEW,
            total_purchases=1,
            total_revenue=Decimal("50"),
            avg_order_value=Decimal("50"),
            days_since_last_purchase=15,
            purchase_frequency_per_month=0.5,
            clv_estimate=Decimal("75"),
            churn_risk_score=0.3,
            first_seen=now - timedelta(days=30),
            last_seen=now - timedelta(days=15),
        ))

    # One-time customers (300)
    for i in range(known_totals["one_time_customers"]):
        profiles.append(CustomerProfile(
            internal_customer_id=f"ONE_{i:03d}",
            behavior_type=BehaviorType.ONE_TIME,
            total_purchases=1,
            total_revenue=Decimal("75"),
            avg_order_value=Decimal("75"),
            days_since_last_purchase=120,
            purchase_frequency_per_month=0.1,
            clv_estimate=Decimal("75"),
            churn_risk_score=0.8,
            first_seen=now - timedelta(days=180),
            last_seen=now - timedelta(days=120),
        ))

    # Irregular customers (250) - 2 purchases
    for i in range(known_totals["irregular_customers"]):
        profiles.append(CustomerProfile(
            internal_customer_id=f"IRR_{i:03d}",
            behavior_type=BehaviorType.IRREGULAR,
            total_purchases=2,
            total_revenue=Decimal("150"),
            avg_order_value=Decimal("75"),
            days_since_last_purchase=45,
            purchase_frequency_per_month=0.3,
            clv_estimate=Decimal("200"),
            churn_risk_score=0.5,
            first_seen=now - timedelta(days=90),
            last_seen=now - timedelta(days=45),
        ))

    # Regular customers (150) - 5+ purchases
    for i in range(known_totals["regular_customers"]):
        profiles.append(CustomerProfile(
            internal_customer_id=f"REG_{i:03d}",
            behavior_type=BehaviorType.REGULAR,
            total_purchases=5,
            total_revenue=Decimal("500"),
            avg_order_value=Decimal("100"),
            days_since_last_purchase=10,
            purchase_frequency_per_month=1.5,
            clv_estimate=Decimal("1000"),
            churn_risk_score=0.15,
            first_seen=now - timedelta(days=365),
            last_seen=now - timedelta(days=10),
        ))

    # Long cycle customers (100) - 3 purchases
    for i in range(known_totals["long_cycle_customers"]):
        profiles.append(CustomerProfile(
            internal_customer_id=f"LC_{i:03d}",
            behavior_type=BehaviorType.LONG_CYCLE,
            total_purchases=3,
            total_revenue=Decimal("300"),
            avg_order_value=Decimal("100"),
            days_since_last_purchase=90,
            purchase_frequency_per_month=0.25,
            clv_estimate=Decimal("500"),
            churn_risk_score=0.35,
            first_seen=now - timedelta(days=400),
            last_seen=now - timedelta(days=90),
        ))

    return profiles


@pytest.fixture
def data_bundle_with_known_totals(
    sample_profiles_with_known_totals: list[CustomerProfile],
) -> EssayDataBundle:
    """Query all essay data from profiles with known totals."""
    return query_all_essay_data(sample_profiles_with_known_totals)


# =============================================================================
# PERCENTAGE SANITY CHECKS
# =============================================================================


class TestPercentageSanity:
    """Validate percentage calculations are bounded and sensible."""

    def test_percentages_bounded_0_to_100(
        self,
        data_bundle_with_known_totals: EssayDataBundle,
    ) -> None:
        """All percentages should be between 0 and 100."""
        data = data_bundle_with_known_totals

        # Loyalty journey percentages
        for pct in data.loyalty_journey.behavior_percentages.values():
            assert 0 <= pct <= 1, f"Behavior percentage {pct} out of [0,1] range"

        # New customer quality conversion rates
        assert 0 <= data.new_customer_quality.conversion_to_2nd <= 1
        assert 0 <= data.new_customer_quality.conversion_to_3rd <= 1
        assert 0 <= data.new_customer_quality.conversion_to_retained <= 1

        # Champion fragility percentages
        assert 0 <= data.champion_fragility.fragile_champion_percentage <= 1

        # Churn prediction at-risk percentage
        assert 0 <= data.churn_prediction.at_risk_percentage <= 1

    def test_funnel_stages_decrease_monotonically(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Funnel stages should decrease (or stay same) from top to bottom."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)
        spec = create_funnel_spec("test_funnel", data)

        stages = spec.data["stages"]
        for i in range(1, len(stages)):
            curr_value = stages[i]["value"]
            prev_value = stages[i - 1]["value"]
            assert curr_value <= prev_value, (
                f"Funnel stage {stages[i]['label']} ({curr_value}) should not be "
                f"greater than previous stage {stages[i-1]['label']} ({prev_value})"
            )

    def test_conversion_rates_are_bounded(
        self,
        data_bundle_with_known_totals: EssayDataBundle,
    ) -> None:
        """Conversion rates should never exceed 100% or be negative."""
        data = data_bundle_with_known_totals.new_customer_quality

        assert 0 <= data.conversion_to_2nd <= 1, (
            f"Conversion to 2nd: {data.conversion_to_2nd} out of bounds"
        )
        assert 0 <= data.conversion_to_3rd <= 1, (
            f"Conversion to 3rd: {data.conversion_to_3rd} out of bounds"
        )
        assert 0 <= data.new_customer_churn_rate <= 1, (
            f"Churn rate: {data.new_customer_churn_rate} out of bounds"
        )

    def test_funnel_has_baseline_context(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Funnel stages should include baseline for context."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)
        spec = create_funnel_spec("test_funnel", data)

        for stage in spec.data["stages"]:
            # Every stage should have baseline
            assert "baseline" in stage, f"Stage {stage['label']} missing baseline"
            assert stage["baseline"] > 0, f"Stage {stage['label']} has zero baseline"

            # Every stage should have display_text with context
            assert "display_text" in stage, f"Stage {stage['label']} missing display_text"
            # display_text should contain "of" to show baseline context
            assert " of " in stage["display_text"], (
                f"Stage {stage['label']} display_text lacks baseline context: "
                f"{stage['display_text']}"
            )


# =============================================================================
# COUNT CONSISTENCY CHECKS
# =============================================================================


class TestCountConsistency:
    """Validate that counts are consistent across calculations."""

    def test_total_customers_matches_source(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Total customers in data should match source profile count."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)

        assert data.total_customers == known_totals["total_customers"], (
            f"Total customers {data.total_customers} doesn't match "
            f"expected {known_totals['total_customers']}"
        )

    def test_segment_counts_sum_to_total(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Behavior segment counts should sum to total customers."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)

        segment_sum = sum(data.behavior_counts.values())

        assert segment_sum == data.total_customers, (
            f"Segment counts sum ({segment_sum}) doesn't match "
            f"total customers ({data.total_customers})"
        )

    def test_individual_segment_counts_match_expected(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Each segment count should match expected value."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)

        assert data.behavior_counts.get("new", 0) == known_totals["new_customers"]
        assert data.behavior_counts.get("one_time", 0) == known_totals["one_time_customers"]
        assert data.behavior_counts.get("irregular", 0) == known_totals["irregular_customers"]
        assert data.behavior_counts.get("regular", 0) == known_totals["regular_customers"]
        assert data.behavior_counts.get("long_cycle", 0) == known_totals["long_cycle_customers"]

    def test_funnel_counts_consistent(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Funnel stage counts should be consistent with profile data."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)

        # Total should match
        assert data.total_new_customers == known_totals["total_customers"]

        # 2nd purchase count should match profiles with 2+ purchases
        expected_2_plus = sum(
            1 for p in sample_profiles_with_known_totals if p.total_purchases >= 2
        )
        assert data.customers_with_2nd_purchase == expected_2_plus

        # 3rd purchase count should match profiles with 3+ purchases
        expected_3_plus = sum(
            1 for p in sample_profiles_with_known_totals if p.total_purchases >= 3
        )
        assert data.customers_with_3rd_purchase == expected_3_plus


# =============================================================================
# SANKEY FLOW VALIDATION
# =============================================================================


class TestSankeyFlowValidation:
    """Validate Sankey diagram data integrity."""

    def test_sankey_nodes_have_groups(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Sankey nodes must have group property for column positioning."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)
        spec = create_sankey_spec("test_sankey", data)

        for node in spec.data["nodes"]:
            assert "group" in node, f"Node {node['id']} missing group property"
            assert isinstance(node["group"], int), f"Node {node['id']} group not int"

    def test_sankey_has_multiple_column_groups(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Sankey should have nodes in multiple columns (groups)."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)
        spec = create_sankey_spec("test_sankey", data)

        groups = {node["group"] for node in spec.data["nodes"]}

        assert len(groups) > 1, (
            f"Sankey has only {len(groups)} group(s). "
            f"Nodes will stack vertically without multiple groups."
        )

    def test_sankey_links_have_valid_sources_and_targets(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Sankey links should reference valid node indices."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)
        spec = create_sankey_spec("test_sankey", data)

        node_indices = {node["index"] for node in spec.data["nodes"]}

        for link in spec.data["links"]:
            assert link["source"] in node_indices, (
                f"Link source {link['source']} not in node indices"
            )
            assert link["target"] in node_indices, (
                f"Link target {link['target']} not in node indices"
            )

    def test_sankey_links_have_positive_values(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Sankey links should have positive values."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)
        spec = create_sankey_spec("test_sankey", data)

        for link in spec.data["links"]:
            assert link["value"] > 0, (
                f"Link {link.get('source_id')} -> {link.get('target_id')} "
                f"has non-positive value {link['value']}"
            )


# =============================================================================
# CROSS-ESSAY CONSISTENCY
# =============================================================================


class TestCrossEssayConsistency:
    """Validate that metrics are consistent across essays."""

    def test_champion_count_consistent_across_essays(
        self,
        data_bundle_with_known_totals: EssayDataBundle,
        known_totals: dict,
    ) -> None:
        """Champion (regular) count should be consistent across essays."""
        # In loyalty journey
        loyalty_champions = data_bundle_with_known_totals.loyalty_journey.behavior_counts.get(
            "regular", 0
        )

        # Champions in fragility analysis should be >= regular count
        # (could include other high-value customers)
        fragility_champions = data_bundle_with_known_totals.champion_fragility.total_champions

        # Loyalty journey regulars should match known count
        assert loyalty_champions == known_totals["regular_customers"], (
            f"Loyalty journey regulars ({loyalty_champions}) doesn't match "
            f"expected ({known_totals['regular_customers']})"
        )

        # Fragility analysis should have at least that many
        assert fragility_champions >= loyalty_champions, (
            f"Fragility champions ({fragility_champions}) should be >= "
            f"loyalty regulars ({loyalty_champions})"
        )

    def test_total_customers_consistent_across_data_bundle(
        self,
        data_bundle_with_known_totals: EssayDataBundle,
        known_totals: dict,
    ) -> None:
        """Total customer count should be consistent across all data types."""
        bundle = data_bundle_with_known_totals

        assert bundle.total_customers == known_totals["total_customers"]
        assert bundle.loyalty_journey.total_customers == known_totals["total_customers"]
        assert bundle.new_customer_quality.total_new_customers == known_totals["total_customers"]


# =============================================================================
# NARRATIVE ACCURACY
# =============================================================================


class TestNarrativeAccuracy:
    """Validate that narratives contain accurate numbers."""

    def test_essay_section_metrics_match_data(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Essay section key metrics should match source data."""
        data = query_loyalty_journey_data(sample_profiles_with_known_totals)
        section = create_loyalty_journey_section(data)

        # Check key metrics are present
        assert "total_customers" in section.key_metrics
        # Note: The key is "regular_percentage" not "champion_percentage"
        assert "regular_percentage" in section.key_metrics

        # Check values match
        assert section.key_metrics["total_customers"] == known_totals["total_customers"]

        # Verify regular_percentage calculation is correct
        expected_regular_pct = (
            known_totals["regular_customers"] / known_totals["total_customers"] * 100
        )
        assert abs(section.key_metrics["regular_percentage"] - expected_regular_pct) < 0.1

    def test_funnel_section_metrics_accurate(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Funnel section metrics should accurately reflect data."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)
        section = create_new_customer_quality_section(data)

        # Check conversion metrics match
        assert abs(
            section.key_metrics["conversion_to_2nd"] - data.conversion_to_2nd * 100
        ) < 0.1
        assert abs(
            section.key_metrics["conversion_to_3rd"] - data.conversion_to_3rd * 100
        ) < 0.1

    def test_percentage_rounding_consistent(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
    ) -> None:
        """Displayed percentages should round consistently."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)
        spec = create_funnel_spec("test_funnel", data)

        for stage in spec.data["stages"]:
            raw_pct = stage["percentage"]
            # display_text should contain a rounded percentage
            if "display_text" in stage:
                # Extract percentage from display_text (e.g., "42% (4,200 of 10,000)")
                pct_str = stage["display_text"].split("%")[0]
                try:
                    displayed_pct = float(pct_str)
                    # Should be within 1% of raw value
                    assert abs(displayed_pct - raw_pct) <= 1, (
                        f"Displayed {displayed_pct}% differs from raw {raw_pct}% "
                        f"by more than 1%"
                    )
                except ValueError:
                    pass  # Skip if can't parse


# =============================================================================
# QUALITY DISTRIBUTION CONSISTENCY
# =============================================================================


class TestQualityDistributionConsistency:
    """Validate quality score distributions are consistent."""

    def test_quality_buckets_sum_to_total(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """Quality score buckets should sum to total customers."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)

        bucket_sum = sum(data.quality_score_distribution.values())

        assert bucket_sum == known_totals["total_customers"], (
            f"Quality buckets sum ({bucket_sum}) doesn't match "
            f"total customers ({known_totals['total_customers']})"
        )

    def test_ltv_distribution_sums_correctly(
        self,
        sample_profiles_with_known_totals: list[CustomerProfile],
        known_totals: dict,
    ) -> None:
        """LTV distribution bucket counts should sum to total."""
        data = query_new_customer_quality_data(sample_profiles_with_known_totals)

        ltv_sum = sum(bucket["count"] for bucket in data.ltv_distribution)

        assert ltv_sum == known_totals["total_customers"], (
            f"LTV distribution sum ({ltv_sum}) doesn't match "
            f"total customers ({known_totals['total_customers']})"
        )
