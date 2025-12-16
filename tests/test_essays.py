"""
Tests for the visual essays module.

Tests cover:
- Base dataclasses (ChartNarrative, ChartSpec, EssaySection, Essay)
- Data queries (all four essay data types)
- Narrative generation
- Override loading and application
- Chart specifications
- Section generation
- HTML rendering
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from src.data.schemas import BehaviorType, CustomerProfile
from src.essays.base import (
    ChartNarrative,
    ChartSpec,
    Essay,
    EssayConfig,
    EssaySection,
    KeyInsight,
    ScrollyStep,
)
from src.essays.data_queries import (
    ChampionFragilityData,
    ChurnPredictionData,
    EssayDataBundle,
    LoyaltyJourneyData,
    NewCustomerQualityData,
    SensitivityData,
    TraitDiscoveryData,
    WhitespaceOpportunityData,
    query_all_essay_data,
    query_champion_fragility_data,
    query_churn_prediction_data,
    query_loyalty_journey_data,
    query_new_customer_quality_data,
    query_sensitivity_data,
    query_trait_discovery_data,
    query_whitespace_data,
)
from src.essays.narratives import (
    export_narratives_to_dict,
    generate_all_narratives,
    generate_champion_scatter_narrative,
    generate_churn_timeline_narrative,
    generate_feature_stability_narrative,
    generate_funnel_narrative,
    generate_loyalty_sankey_narrative,
    generate_robustness_gauge_narrative,
    generate_trait_heatmap_narrative,
    generate_whitespace_opportunity_narrative,
)
from src.essays.overrides import apply_overrides, load_overrides


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_profiles() -> list[CustomerProfile]:
    """Create sample customer profiles for testing."""
    now = datetime.now()
    profiles = []

    # New customers
    for i in range(100):
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

    # One-time customers
    for i in range(200):
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

    # Irregular customers
    for i in range(150):
        profiles.append(CustomerProfile(
            internal_customer_id=f"IRR_{i:03d}",
            behavior_type=BehaviorType.IRREGULAR,
            total_purchases=3,
            total_revenue=Decimal("200"),
            avg_order_value=Decimal("66.67"),
            days_since_last_purchase=45,
            purchase_frequency_per_month=0.5,
            clv_estimate=Decimal("300"),
            churn_risk_score=0.4,
            first_seen=now - timedelta(days=120),
            last_seen=now - timedelta(days=45),
        ))

    # Regular customers (champions)
    for i in range(100):
        profiles.append(CustomerProfile(
            internal_customer_id=f"REG_{i:03d}",
            behavior_type=BehaviorType.REGULAR,
            total_purchases=10,
            total_revenue=Decimal("1000"),
            avg_order_value=Decimal("100"),
            days_since_last_purchase=10,
            purchase_frequency_per_month=2.0,
            clv_estimate=Decimal("2000"),
            churn_risk_score=0.1,
            first_seen=now - timedelta(days=365),
            last_seen=now - timedelta(days=10),
        ))

    # Long cycle customers
    for i in range(50):
        profiles.append(CustomerProfile(
            internal_customer_id=f"LC_{i:03d}",
            behavior_type=BehaviorType.LONG_CYCLE,
            total_purchases=4,
            total_revenue=Decimal("500"),
            avg_order_value=Decimal("125"),
            days_since_last_purchase=90,
            purchase_frequency_per_month=0.3,
            clv_estimate=Decimal("750"),
            churn_risk_score=0.3,
            first_seen=now - timedelta(days=400),
            last_seen=now - timedelta(days=90),
        ))

    return profiles


@pytest.fixture
def loyalty_data(sample_profiles: list[CustomerProfile]) -> LoyaltyJourneyData:
    """Query loyalty journey data from sample profiles."""
    return query_loyalty_journey_data(sample_profiles)


@pytest.fixture
def fragility_data(sample_profiles: list[CustomerProfile]) -> ChampionFragilityData:
    """Query champion fragility data from sample profiles."""
    return query_champion_fragility_data(sample_profiles)


@pytest.fixture
def quality_data(sample_profiles: list[CustomerProfile]) -> NewCustomerQualityData:
    """Query new customer quality data from sample profiles."""
    return query_new_customer_quality_data(sample_profiles)


@pytest.fixture
def churn_data(sample_profiles: list[CustomerProfile]) -> ChurnPredictionData:
    """Query churn prediction data from sample profiles."""
    return query_churn_prediction_data(sample_profiles)


@pytest.fixture
def data_bundle(sample_profiles: list[CustomerProfile]) -> EssayDataBundle:
    """Query all essay data from sample profiles."""
    return query_all_essay_data(sample_profiles)


# =============================================================================
# CHART NARRATIVE TESTS
# =============================================================================


class TestChartNarrative:
    """Tests for ChartNarrative dataclass."""

    def test_auto_narrative_used_by_default(self) -> None:
        """Test that auto-generated narrative is used when no override."""
        narrative = ChartNarrative(
            chart_id="test_chart",
            auto_headline="Auto Headline",
            auto_insight="Auto Insight",
            auto_callout="Auto Callout",
        )

        assert narrative.headline == "Auto Headline"
        assert narrative.insight == "Auto Insight"
        assert narrative.callout == "Auto Callout"

    def test_override_takes_precedence(self) -> None:
        """Test that override values take precedence over auto."""
        narrative = ChartNarrative(
            chart_id="test_chart",
            auto_headline="Auto Headline",
            auto_insight="Auto Insight",
            auto_callout="Auto Callout",
            override_headline="Override Headline",
            override_insight="Override Insight",
        )

        assert narrative.headline == "Override Headline"
        assert narrative.insight == "Override Insight"
        assert narrative.callout == "Auto Callout"  # No override

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        narrative = ChartNarrative(
            chart_id="test_chart",
            auto_headline="Auto",
            override_headline="Override",
        )

        d = narrative.to_dict()

        assert d["chart_id"] == "test_chart"
        assert d["headline"] == "Override"
        assert d["auto_headline"] == "Auto"
        assert d["has_overrides"] is True


class TestChartSpec:
    """Tests for ChartSpec dataclass."""

    def test_basic_chart_spec(self) -> None:
        """Test basic chart specification."""
        spec = ChartSpec(
            chart_id="test_sankey",
            chart_type="sankey",
            data={"nodes": [], "links": []},
            config={"width": 800, "height": 500},
        )

        assert spec.chart_id == "test_sankey"
        assert spec.chart_type == "sankey"
        assert "nodes" in spec.data

    def test_scrolly_triggers(self) -> None:
        """Test scrolly trigger configuration."""
        spec = ChartSpec(
            chart_id="test_chart",
            chart_type="scatter",
            data={},
            scrolly_triggers=[
                {"step": 1, "action": "highlight", "params": {"id": "foo"}},
                {"step": 2, "action": "showAll", "params": {}},
            ],
        )

        assert len(spec.scrolly_triggers) == 2
        assert spec.scrolly_triggers[0]["action"] == "highlight"

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        spec = ChartSpec(
            chart_id="test",
            chart_type="bar",
            data={"bars": [1, 2, 3]},
            config={"color": "blue"},
        )

        d = spec.to_dict()

        assert d["chart_id"] == "test"
        assert d["chart_type"] == "bar"
        assert d["data"]["bars"] == [1, 2, 3]


# =============================================================================
# DATA QUERY TESTS
# =============================================================================


class TestLoyaltyJourneyQuery:
    """Tests for loyalty journey data queries."""

    def test_behavior_counts(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test behavior type counting."""
        counts = loyalty_data.behavior_counts

        assert counts["new"] == 100
        assert counts["one_time"] == 200
        assert counts["irregular"] == 150
        assert counts["regular"] == 100
        assert counts["long_cycle"] == 50

    def test_behavior_percentages(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test behavior percentage calculation."""
        percentages = loyalty_data.behavior_percentages

        assert abs(percentages["one_time"] - 0.333) < 0.01  # 200/600
        assert abs(percentages["regular"] - 0.167) < 0.01  # 100/600

    def test_total_customers(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test total customer count."""
        assert loyalty_data.total_customers == 600

    def test_sankey_nodes_generated(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test Sankey nodes are generated."""
        assert len(loyalty_data.sankey_nodes) > 0
        node_ids = [n["id"] for n in loyalty_data.sankey_nodes]
        assert "new" in node_ids
        assert "regular" in node_ids

    def test_sankey_links_generated(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test Sankey links are generated."""
        assert len(loyalty_data.sankey_links) > 0
        # All links should have positive values
        assert all(l["value"] > 0 for l in loyalty_data.sankey_links)

    def test_loyalty_threshold_detected(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test loyalty threshold detection."""
        # Should identify some threshold
        assert loyalty_data.loyalty_threshold_order >= 1

    def test_empty_profiles_handled(self) -> None:
        """Test handling of empty profile list."""
        data = query_loyalty_journey_data([])

        assert data.total_customers == 0
        assert data.behavior_counts == {}


class TestChampionFragilityQuery:
    """Tests for champion fragility data queries."""

    def test_champions_identified(self, fragility_data: ChampionFragilityData) -> None:
        """Test champion identification."""
        assert fragility_data.total_champions > 0

    def test_fragility_calculated(self, fragility_data: ChampionFragilityData) -> None:
        """Test fragility score calculation."""
        for champion in fragility_data.champions:
            assert 0 <= champion["fragility_score"] <= 1

    def test_fragility_buckets(self, fragility_data: ChampionFragilityData) -> None:
        """Test fragility bucket distribution."""
        buckets = fragility_data.fragility_buckets

        assert "low" in buckets
        assert "medium" in buckets
        assert "high" in buckets

    def test_revenue_at_risk_calculated(self, fragility_data: ChampionFragilityData) -> None:
        """Test revenue at risk calculation."""
        assert fragility_data.champion_revenue_total > 0

    def test_quadrant_data_generated(self, fragility_data: ChampionFragilityData) -> None:
        """Test quadrant scatter data generation."""
        assert len(fragility_data.quadrant_data) > 0

        for point in fragility_data.quadrant_data:
            assert "x" in point  # revenue
            assert "y" in point  # fragility
            assert "is_fragile" in point


class TestNewCustomerQualityQuery:
    """Tests for new customer quality data queries."""

    def test_funnel_counts(self, quality_data: NewCustomerQualityData) -> None:
        """Test funnel conversion counts."""
        assert quality_data.total_new_customers == 600

    def test_conversion_rates(self, quality_data: NewCustomerQualityData) -> None:
        """Test conversion rate calculation."""
        assert 0 <= quality_data.conversion_to_2nd <= 1
        assert 0 <= quality_data.conversion_to_3rd <= 1

    def test_quality_distribution(self, quality_data: NewCustomerQualityData) -> None:
        """Test quality score distribution."""
        dist = quality_data.quality_score_distribution

        assert "low" in dist
        assert "medium" in dist
        assert "high" in dist

    def test_ltv_distribution(self, quality_data: NewCustomerQualityData) -> None:
        """Test LTV distribution buckets."""
        assert len(quality_data.ltv_distribution) > 0


class TestChurnPredictionQuery:
    """Tests for churn prediction data queries."""

    def test_at_risk_identified(self, churn_data: ChurnPredictionData) -> None:
        """Test at-risk customer identification."""
        # With our sample data, one_time customers should be at risk
        assert churn_data.total_at_risk > 0

    def test_leading_indicators(self, churn_data: ChurnPredictionData) -> None:
        """Test leading indicators generation."""
        assert len(churn_data.leading_indicators) > 0

        for indicator in churn_data.leading_indicators:
            assert "indicator" in indicator
            assert "importance" in indicator
            assert "avg_days_before_churn" in indicator

    def test_top_predictors(self, churn_data: ChurnPredictionData) -> None:
        """Test top predictor extraction."""
        assert len(churn_data.top_predictors) > 0

    def test_radar_dimensions(self, churn_data: ChurnPredictionData) -> None:
        """Test radar chart dimension data."""
        assert len(churn_data.radar_dimensions) > 0

        for dim in churn_data.radar_dimensions:
            assert "dimension" in dim
            assert "all_customers" in dim
            assert "at_risk" in dim


class TestEssayDataBundle:
    """Tests for combined essay data bundle."""

    def test_all_data_types_present(self, data_bundle: EssayDataBundle) -> None:
        """Test all data types are populated."""
        assert data_bundle.loyalty_journey is not None
        assert data_bundle.champion_fragility is not None
        assert data_bundle.new_customer_quality is not None
        assert data_bundle.churn_prediction is not None

    def test_metadata_populated(self, data_bundle: EssayDataBundle) -> None:
        """Test metadata is populated."""
        assert data_bundle.total_customers == 600
        assert data_bundle.data_period_start is not None
        assert data_bundle.data_period_end is not None


# =============================================================================
# NARRATIVE GENERATION TESTS
# =============================================================================


class TestNarrativeGeneration:
    """Tests for auto-narrative generation."""

    def test_loyalty_narrative_generated(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test loyalty sankey narrative generation."""
        narrative = generate_loyalty_sankey_narrative(loyalty_data)

        assert narrative.chart_id == "loyalty_sankey"
        assert len(narrative.auto_headline) > 0
        assert len(narrative.auto_insight) > 0
        assert len(narrative.auto_callout) > 0

    def test_champion_narrative_generated(self, fragility_data: ChampionFragilityData) -> None:
        """Test champion scatter narrative generation."""
        narrative = generate_champion_scatter_narrative(fragility_data)

        assert narrative.chart_id == "champion_scatter"
        assert "%" in narrative.auto_headline  # Should mention percentage

    def test_funnel_narrative_generated(self, quality_data: NewCustomerQualityData) -> None:
        """Test funnel narrative generation."""
        narrative = generate_funnel_narrative(quality_data)

        assert narrative.chart_id == "new_customer_funnel"
        assert "purchase" in narrative.auto_insight.lower()

    def test_churn_narrative_generated(self, churn_data: ChurnPredictionData) -> None:
        """Test churn timeline narrative generation."""
        narrative = generate_churn_timeline_narrative(churn_data)

        assert narrative.chart_id == "churn_timeline"
        assert "days" in narrative.auto_headline.lower()

    def test_all_narratives_generated(self, data_bundle: EssayDataBundle) -> None:
        """Test all narratives are generated."""
        narratives = generate_all_narratives(data_bundle)

        expected_charts = [
            "loyalty_sankey",
            "loyalty_threshold",
            "champion_scatter",
            "champion_thermometer",
            "new_customer_funnel",
            "quality_distribution",
            "churn_timeline",
            "churn_radar",
        ]

        for chart_id in expected_charts:
            assert chart_id in narratives
            assert isinstance(narratives[chart_id], ChartNarrative)

    def test_narratives_export_to_dict(self, data_bundle: EssayDataBundle) -> None:
        """Test narrative export to dictionary format."""
        narratives = generate_all_narratives(data_bundle)
        exported = export_narratives_to_dict(narratives)

        assert "loyalty_sankey" in exported
        assert "headline" in exported["loyalty_sankey"]
        assert "insight" in exported["loyalty_sankey"]


# =============================================================================
# OVERRIDE TESTS
# =============================================================================


class TestOverrides:
    """Tests for narrative override functionality."""

    def test_apply_empty_overrides(self, data_bundle: EssayDataBundle) -> None:
        """Test applying empty overrides doesn't change narratives."""
        narratives = generate_all_narratives(data_bundle)
        original_headlines = {k: v.headline for k, v in narratives.items()}

        updated = apply_overrides(narratives, {})

        for chart_id, narrative in updated.items():
            assert narrative.headline == original_headlines[chart_id]

    def test_apply_headline_override(self, data_bundle: EssayDataBundle) -> None:
        """Test applying a headline override."""
        narratives = generate_all_narratives(data_bundle)

        # apply_overrides expects chart_id -> overrides dict (not nested under "narratives")
        overrides = {
            "loyalty_sankey": {
                "headline": "Custom Headline",
                "insight": None,
                "callout": None,
            }
        }

        updated = apply_overrides(narratives, overrides)

        assert updated["loyalty_sankey"].headline == "Custom Headline"
        # Other fields should still use auto
        assert updated["loyalty_sankey"].insight == narratives["loyalty_sankey"].auto_insight

    def test_apply_multiple_overrides(self, data_bundle: EssayDataBundle) -> None:
        """Test applying multiple field overrides."""
        narratives = generate_all_narratives(data_bundle)

        # apply_overrides expects chart_id -> overrides dict (not nested under "narratives")
        overrides = {
            "champion_scatter": {
                "headline": "Custom Champion Headline",
                "insight": None,
                "callout": "Custom Champion Callout",
            }
        }

        updated = apply_overrides(narratives, overrides)

        assert updated["champion_scatter"].headline == "Custom Champion Headline"
        assert updated["champion_scatter"].callout == "Custom Champion Callout"
        # Insight should still be auto
        assert updated["champion_scatter"].insight == narratives["champion_scatter"].auto_insight


# =============================================================================
# ESSAY SECTION TESTS
# =============================================================================


class TestEssaySections:
    """Tests for essay section creation."""

    def test_loyalty_section_creation(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test loyalty journey section creation."""
        from src.essays.sections import create_loyalty_journey_section

        section = create_loyalty_journey_section(loyalty_data)

        assert section.section_id == "loyalty-journey"
        assert section.chart.chart_type == "sankey"
        assert len(section.scrolly_steps) > 0
        assert section.executive_summary
        assert section.marketing_narrative

    def test_champion_section_creation(self, fragility_data: ChampionFragilityData) -> None:
        """Test champion fragility section creation."""
        from src.essays.sections import create_champion_fragility_section

        section = create_champion_fragility_section(fragility_data)

        assert section.section_id == "champion-fragility"
        assert section.chart.chart_type == "scatter"
        assert "fragile" in section.executive_summary.lower()

    def test_quality_section_creation(self, quality_data: NewCustomerQualityData) -> None:
        """Test new customer quality section creation."""
        from src.essays.sections import create_new_customer_quality_section

        section = create_new_customer_quality_section(quality_data)

        assert section.section_id == "new-customer-quality"
        assert section.chart.chart_type == "funnel"

    def test_churn_section_creation(self, churn_data: ChurnPredictionData) -> None:
        """Test churn prediction section creation."""
        from src.essays.sections import create_churn_prediction_section

        section = create_churn_prediction_section(churn_data)

        assert section.section_id == "churn-prediction"
        assert section.chart.chart_type == "timeline"
        assert len(section.supporting_charts) > 0


# =============================================================================
# ESSAY TESTS
# =============================================================================


class TestEssay:
    """Tests for Essay dataclass."""

    def test_essay_creation(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test basic essay creation."""
        from src.essays.sections import create_loyalty_journey_section

        section = create_loyalty_journey_section(loyalty_data)

        essay = Essay(
            essay_id="test-essay",
            title="Test Essay",
            subtitle="A test subtitle",
            sections=[section],
            customer_count=600,
        )

        assert essay.essay_id == "test-essay"
        assert len(essay.sections) == 1
        assert len(essay.toc_entries) == 1

    def test_key_insights(self) -> None:
        """Test key insight creation."""
        insight = KeyInsight(
            text="23% of Champions are fragile",
            metric_value="23%",
            metric_label="Champions at risk",
            category="risk",
        )

        assert insight.text == "23% of Champions are fragile"
        assert insight.category == "risk"

    def test_essay_to_dict(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test essay serialization."""
        from src.essays.sections import create_loyalty_journey_section

        section = create_loyalty_journey_section(loyalty_data)

        essay = Essay(
            essay_id="test",
            title="Test",
            subtitle="Subtitle",
            sections=[section],
        )

        d = essay.to_dict()

        assert d["essay_id"] == "test"
        assert "sections" in d
        assert len(d["sections"]) == 1


# =============================================================================
# ESSAY CONFIG TESTS
# =============================================================================


class TestEssayConfig:
    """Tests for EssayConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EssayConfig()

        assert config.essays == ["all"]
        assert config.audience == "all"
        assert config.include_appendix is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EssayConfig(
            essays=["loyalty", "champions"],
            audience="executive",
            include_appendix=False,
        )

        assert config.essays == ["loyalty", "champions"]
        assert config.audience == "executive"
        assert config.include_appendix is False


# =============================================================================
# SCROLLY STEP TESTS
# =============================================================================


class TestScrollyStep:
    """Tests for ScrollyStep dataclass."""

    def test_step_creation(self) -> None:
        """Test basic step creation."""
        step = ScrollyStep(
            step_number=1,
            narrative_text="This is the first step",
            chart_action="highlight",
            action_params={"nodeId": "foo"},
        )

        assert step.step_number == 1
        assert step.chart_action == "highlight"

    def test_step_to_dict(self) -> None:
        """Test step serialization."""
        step = ScrollyStep(
            step_number=2,
            narrative_text="Second step",
            chart_action="showAll",
            action_params={},
        )

        d = step.to_dict()

        assert d["step"] == 2
        assert d["text"] == "Second step"
        assert d["action"] == "showAll"


# =============================================================================
# HTML RENDERER TESTS
# =============================================================================


class TestHTMLRenderer:
    """Tests for HTML rendering."""

    def test_render_essay_html(self, data_bundle: EssayDataBundle) -> None:
        """Test HTML rendering of complete essay."""
        from src.essays.renderers.html import render_essay_html
        from src.essays.sections import create_loyalty_journey_section

        section = create_loyalty_journey_section(data_bundle.loyalty_journey)

        essay = Essay(
            essay_id="test",
            title="Test Essay",
            subtitle="Test",
            sections=[section],
        )

        config = EssayConfig()
        html = render_essay_html(essay, config)

        assert "<!DOCTYPE html>" in html
        assert "Test Essay" in html
        assert "ESSAY_DATA" in html  # JavaScript data
        assert "scrollama" in html.lower() or "scrolly" in html.lower()

    def test_render_includes_d3_scripts(self, data_bundle: EssayDataBundle) -> None:
        """Test that D3 scripts are included."""
        from src.essays.renderers.html import render_essay_html
        from src.essays.sections import create_loyalty_journey_section

        section = create_loyalty_journey_section(data_bundle.loyalty_journey)

        essay = Essay(
            essay_id="test",
            title="Test",
            subtitle="Test",
            sections=[section],
        )

        config = EssayConfig()
        html = render_essay_html(essay, config)

        assert "d3" in html.lower()


# =============================================================================
# CHART SPEC TESTS
# =============================================================================


class TestChartSpecs:
    """Tests for chart specification generators."""

    def test_sankey_spec(self, loyalty_data: LoyaltyJourneyData) -> None:
        """Test Sankey chart spec generation."""
        from src.essays.charts.sankey import create_sankey_spec

        spec = create_sankey_spec("test_sankey", loyalty_data)

        assert spec.chart_type == "sankey"
        assert "nodes" in spec.data
        assert "links" in spec.data
        assert "width" in spec.config

    def test_scatter_spec(self, fragility_data: ChampionFragilityData) -> None:
        """Test scatter chart spec generation."""
        from src.essays.charts.scatter import create_scatter_spec

        spec = create_scatter_spec("test_scatter", fragility_data)

        assert spec.chart_type == "scatter"
        assert "points" in spec.data

    def test_funnel_spec(self, quality_data: NewCustomerQualityData) -> None:
        """Test funnel chart spec generation."""
        from src.essays.charts.funnel import create_funnel_spec

        spec = create_funnel_spec("test_funnel", quality_data)

        assert spec.chart_type == "funnel"
        assert "stages" in spec.data

    def test_timeline_spec(self, churn_data: ChurnPredictionData) -> None:
        """Test timeline chart spec generation."""
        from src.essays.charts.timeline import create_timeline_spec

        spec = create_timeline_spec("test_timeline", churn_data)

        assert spec.chart_type == "timeline"
        assert "indicators" in spec.data

    def test_radar_spec(self, churn_data: ChurnPredictionData) -> None:
        """Test radar chart spec generation."""
        from src.essays.charts.radar import create_radar_spec

        spec = create_radar_spec("test_radar", churn_data)

        assert spec.chart_type == "radar"
        assert "axes" in spec.data
        assert "series" in spec.data


# =============================================================================
# ESSAY 5-7: EXTENDED ESSAY TESTS
# =============================================================================


@pytest.fixture
def whitespace_data(sample_profiles: list[CustomerProfile]) -> WhitespaceOpportunityData:
    """Query whitespace data from sample profiles."""
    return query_whitespace_data(profiles=sample_profiles)


@pytest.fixture
def trait_data(sample_profiles: list[CustomerProfile]) -> TraitDiscoveryData:
    """Query trait discovery data from sample profiles."""
    return query_trait_discovery_data(profiles=sample_profiles)


@pytest.fixture
def sensitivity_data(sample_profiles: list[CustomerProfile]) -> SensitivityData:
    """Query sensitivity data from sample profiles."""
    return query_sensitivity_data(profiles=sample_profiles)


class TestWhitespaceQuery:
    """Tests for whitespace opportunity data queries."""

    def test_whitespace_data_structure(self, whitespace_data: WhitespaceOpportunityData) -> None:
        """Test whitespace data structure is correct."""
        assert isinstance(whitespace_data.top_opportunities, list)
        assert isinstance(whitespace_data.similarity_distribution, list)
        assert isinstance(whitespace_data.buyer_profile, dict)
        assert isinstance(whitespace_data.lookalike_profile, dict)

    def test_empty_profiles_handled(self) -> None:
        """Test handling of empty profile list."""
        data = query_whitespace_data(profiles=[])

        assert data.total_opportunities == 0
        assert data.total_lookalikes == 0
        assert data.top_opportunities == []

    def test_opportunity_values_positive(self, whitespace_data: WhitespaceOpportunityData) -> None:
        """Test that opportunity values are non-negative."""
        assert whitespace_data.total_opportunity_value >= 0
        for opp in whitespace_data.top_opportunities:
            assert opp.get("opportunity_value", 0) >= 0


class TestTraitDiscoveryQuery:
    """Tests for trait discovery data queries."""

    def test_trait_data_structure(self, trait_data: TraitDiscoveryData) -> None:
        """Test trait data structure is correct."""
        assert isinstance(trait_data.top_traits, list)
        assert isinstance(trait_data.impact_heatmap, list)
        assert isinstance(trait_data.trait_coverage, list)

    def test_traits_have_scores(self, trait_data: TraitDiscoveryData) -> None:
        """Test that traits have valid impact scores."""
        for trait in trait_data.top_traits:
            assert "revenue_impact" in trait
            assert "retention_impact" in trait
            assert 0 <= trait["revenue_impact"] <= 1
            assert 0 <= trait["retention_impact"] <= 1

    def test_significant_traits_counted(self, trait_data: TraitDiscoveryData) -> None:
        """Test significant trait counting."""
        assert trait_data.total_traits_found > 0
        assert trait_data.significant_trait_count >= 0
        assert trait_data.significant_trait_count <= trait_data.total_traits_found

    def test_recommended_traits_populated(self, trait_data: TraitDiscoveryData) -> None:
        """Test recommended trait lists are populated."""
        assert isinstance(trait_data.segmentation_traits, list)
        assert isinstance(trait_data.personalization_traits, list)
        assert isinstance(trait_data.retention_traits, list)

    def test_empty_profiles_uses_sample_data(self) -> None:
        """Test that empty profiles still generates sample trait data."""
        data = query_trait_discovery_data(profiles=[])

        # Should generate sample traits even with no profiles
        assert data.total_traits_found > 0
        assert len(data.top_traits) > 0


class TestSensitivityQuery:
    """Tests for sensitivity analysis data queries."""

    def test_sensitivity_data_structure(self, sensitivity_data: SensitivityData) -> None:
        """Test sensitivity data structure is correct."""
        assert isinstance(sensitivity_data.robustness_tiers, dict)
        assert isinstance(sensitivity_data.feature_stability, list)
        assert isinstance(sensitivity_data.critical_features, list)
        assert isinstance(sensitivity_data.time_consistency, list)

    def test_robustness_tiers_valid(self, sensitivity_data: SensitivityData) -> None:
        """Test that robustness tiers contain valid values."""
        valid_tiers = {"HIGH", "MEDIUM", "LOW"}
        for tier in sensitivity_data.robustness_tiers.keys():
            assert tier.upper() in valid_tiers

    def test_stability_scores_in_range(self, sensitivity_data: SensitivityData) -> None:
        """Test that stability scores are in valid range."""
        assert 0 <= sensitivity_data.avg_feature_stability <= 1
        for item in sensitivity_data.feature_stability:
            assert 0 <= item["avg_stability"] <= 1

    def test_empty_profiles_uses_sample_data(self) -> None:
        """Test that empty profiles still generates sample sensitivity data."""
        data = query_sensitivity_data(profiles=[])

        # Should generate sample segments even with no profiles
        assert data.total_segments > 0


class TestExtendedNarrativeGeneration:
    """Tests for narrative generation for essays 5-7."""

    def test_whitespace_narrative_generated(self, whitespace_data: WhitespaceOpportunityData) -> None:
        """Test whitespace opportunity narrative generation."""
        narrative = generate_whitespace_opportunity_narrative(whitespace_data)

        assert narrative.chart_id == "whitespace_opportunity"
        assert len(narrative.auto_headline) > 0
        assert len(narrative.auto_insight) > 0
        assert len(narrative.auto_callout) > 0

    def test_trait_narrative_generated(self, trait_data: TraitDiscoveryData) -> None:
        """Test trait heatmap narrative generation."""
        narrative = generate_trait_heatmap_narrative(trait_data)

        assert narrative.chart_id == "trait_heatmap"
        assert len(narrative.auto_headline) > 0
        assert "trait" in narrative.auto_insight.lower()

    def test_robustness_narrative_generated(self, sensitivity_data: SensitivityData) -> None:
        """Test robustness gauge narrative generation."""
        narrative = generate_robustness_gauge_narrative(sensitivity_data)

        assert narrative.chart_id == "robustness_gauge"
        assert "%" in narrative.auto_headline

    def test_feature_stability_narrative_generated(self, sensitivity_data: SensitivityData) -> None:
        """Test feature stability narrative generation."""
        narrative = generate_feature_stability_narrative(sensitivity_data)

        assert narrative.chart_id == "feature_stability"
        assert len(narrative.auto_headline) > 0

    def test_all_narratives_include_extended_essays(self, data_bundle: EssayDataBundle) -> None:
        """Test all narratives include extended essay charts."""
        narratives = generate_all_narratives(data_bundle)

        # Check extended essay narratives are included
        extended_charts = [
            "whitespace_opportunity",
            "whitespace_similarity",
            "whitespace_comparison",
            "trait_heatmap",
            "trait_impact",
            "trait_coverage",
            "robustness_gauge",
            "feature_stability",
            "time_consistency",
        ]

        for chart_id in extended_charts:
            assert chart_id in narratives, f"Missing narrative for {chart_id}"
            assert isinstance(narratives[chart_id], ChartNarrative)


class TestExtendedEssaySections:
    """Tests for extended essay section creation (Essays 5-7)."""

    def test_whitespace_section_creation(self, whitespace_data: WhitespaceOpportunityData) -> None:
        """Test whitespace opportunities section creation."""
        from src.essays.sections import create_whitespace_opportunities_section

        section = create_whitespace_opportunities_section(whitespace_data)

        assert section.section_id == "whitespace-opportunities"
        assert section.chart.chart_type == "bar"
        assert len(section.scrolly_steps) > 0
        assert section.executive_summary
        assert section.marketing_narrative
        assert "opportunities" in section.title.lower()  # "Hidden Cross-Sell Opportunities"

    def test_trait_section_creation(self, trait_data: TraitDiscoveryData) -> None:
        """Test trait insights section creation."""
        from src.essays.sections import create_trait_insights_section

        section = create_trait_insights_section(trait_data)

        assert section.section_id == "trait-insights"
        assert section.chart.chart_type == "heatmap"
        assert len(section.scrolly_steps) > 0
        assert section.executive_summary
        assert "customer" in section.title.lower()  # "What Makes Customers Different"

    def test_robustness_section_creation(self, sensitivity_data: SensitivityData) -> None:
        """Test segment robustness section creation."""
        from src.essays.sections import create_segment_robustness_section

        section = create_segment_robustness_section(sensitivity_data)

        assert section.section_id == "segment-robustness"
        assert section.chart.chart_type == "donut"
        assert len(section.scrolly_steps) > 0
        assert section.executive_summary
        assert "robust" in section.title.lower()

    def test_whitespace_key_metrics(self, whitespace_data: WhitespaceOpportunityData) -> None:
        """Test whitespace section key metrics."""
        from src.essays.sections import create_whitespace_opportunities_section

        section = create_whitespace_opportunities_section(whitespace_data)

        assert "total_opportunity_value" in section.key_metrics
        assert "total_lookalikes" in section.key_metrics
        assert "top_category" in section.key_metrics

    def test_trait_key_metrics(self, trait_data: TraitDiscoveryData) -> None:
        """Test trait section key metrics."""
        from src.essays.sections import create_trait_insights_section

        section = create_trait_insights_section(trait_data)

        assert "total_traits" in section.key_metrics
        assert "significant_traits" in section.key_metrics
        assert "top_revenue_trait" in section.key_metrics

    def test_robustness_key_metrics(self, sensitivity_data: SensitivityData) -> None:
        """Test robustness section key metrics."""
        from src.essays.sections import create_segment_robustness_section

        section = create_segment_robustness_section(sensitivity_data)

        assert "total_segments" in section.key_metrics
        assert "high_robustness_pct" in section.key_metrics
        assert "avg_feature_stability" in section.key_metrics


class TestExtendedEssayDataBundle:
    """Tests for extended essay data in bundle."""

    def test_extended_data_in_bundle(self, data_bundle: EssayDataBundle) -> None:
        """Test extended essay data is present in bundle."""
        assert data_bundle.whitespace is not None
        assert data_bundle.trait_discovery is not None
        assert data_bundle.sensitivity is not None

    def test_extended_data_types(self, data_bundle: EssayDataBundle) -> None:
        """Test extended data types are correct."""
        assert isinstance(data_bundle.whitespace, WhitespaceOpportunityData)
        assert isinstance(data_bundle.trait_discovery, TraitDiscoveryData)
        assert isinstance(data_bundle.sensitivity, SensitivityData)
