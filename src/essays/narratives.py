"""
Auto-narrative generation from data insights.

This module generates human-readable narrative text that explains the insights
shown in each chart. Narratives are data-driven and update automatically
based on the actual metrics.
"""

from src.essays.base import ChartNarrative
from src.essays.data_queries import (
    ChampionFragilityData,
    ChurnPredictionData,
    EssayDataBundle,
    LoyaltyJourneyData,
    NewCustomerQualityData,
    SensitivityData,
    TraitDiscoveryData,
    WhitespaceOpportunityData,
)


# =============================================================================
# NARRATIVE GENERATORS
# =============================================================================


def generate_loyalty_sankey_narrative(data: LoyaltyJourneyData) -> ChartNarrative:
    """Generate narrative for the loyalty journey Sankey chart."""
    # Calculate key insights
    one_time_pct = data.behavior_percentages.get("one_time", 0) * 100
    regular_pct = data.behavior_percentages.get("regular", 0) * 100
    threshold = data.loyalty_threshold_order
    retention_jump = data.loyalty_threshold_retention_jump * 100

    # Generate headline
    if threshold > 0:
        headline = f"Order #{threshold} is your loyalty threshold"
    else:
        headline = "The path to loyalty is narrower than you think"

    # Generate insight
    if one_time_pct > 50:
        insight = (
            f"Most customers ({one_time_pct:.0f}%) never make it past their first purchase. "
            f"Only {regular_pct:.1f}% become regular customers. "
        )
    else:
        insight = (
            f"{one_time_pct:.0f}% of customers make only one purchase, "
            f"while {regular_pct:.1f}% develop into loyal regulars. "
        )

    if threshold > 0 and retention_jump > 0:
        insight += (
            f"The magic number is {threshold} orders - customers who reach this "
            f"point show a {retention_jump:.0f}% jump in retention probability."
        )

    # Generate callout
    if data.average_orders_to_champion > 0:
        callout = (
            f"Focus retention efforts on customers approaching order #{threshold}. "
            f"Champions average {data.average_orders_to_champion:.1f} orders."
        )
    else:
        callout = "Early engagement is critical - most churn happens before order 3."

    return ChartNarrative(
        chart_id="loyalty_sankey",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_loyalty_threshold_narrative(data: LoyaltyJourneyData) -> ChartNarrative:
    """Generate narrative for the loyalty threshold line chart."""
    threshold = data.loyalty_threshold_order
    retention_at_threshold = data.retention_by_order_count.get(threshold, 0) * 100
    retention_before = data.retention_by_order_count.get(max(1, threshold - 1), 0) * 100

    headline = f"Retention jumps dramatically at order {threshold}"

    insight = (
        f"Customers with {threshold} orders have {retention_at_threshold:.0f}% retention, "
        f"compared to {retention_before:.0f}% for those with {threshold - 1} orders. "
        f"This {retention_at_threshold - retention_before:.0f} percentage point jump "
        f"represents the 'loyalty inflection point'."
    )

    callout = (
        f"Invest in getting customers to order #{threshold}. "
        f"The ROI of retention efforts multiplies past this point."
    )

    return ChartNarrative(
        chart_id="loyalty_threshold",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_champion_scatter_narrative(data: ChampionFragilityData) -> ChartNarrative:
    """Generate narrative for the champion fragility scatter plot."""
    fragile_pct = data.fragile_champion_percentage * 100
    fragile_count = data.fragile_champion_count
    total = data.total_champions
    at_risk_revenue = float(data.fragile_revenue_at_risk)

    headline = f"{fragile_pct:.0f}% of your Champions are fragile"

    insight = (
        f"Out of {total:,} high-value customers, {fragile_count:,} show warning signs. "
        f"These fragile Champions represent ${at_risk_revenue:,.0f} in revenue at risk. "
        f"A Champion who hasn't purchased in {data.days_threshold_for_risk} days "
        f"is already showing decay signals."
    )

    callout = (
        f"Don't wait for RFM scores to drop. "
        f"These {fragile_count:,} customers need proactive outreach now."
    )

    return ChartNarrative(
        chart_id="champion_scatter",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_champion_thermometer_narrative(data: ChampionFragilityData) -> ChartNarrative:
    """Generate narrative for the risk thermometer visualization."""
    high_risk = data.fragility_buckets.get("high", 0)
    medium_risk = data.fragility_buckets.get("medium", 0)
    low_risk = data.fragility_buckets.get("low", 0)

    headline = f"{high_risk} Champions need immediate attention"

    insight = (
        f"Risk distribution: {high_risk} high-risk, {medium_risk} medium-risk, "
        f"{low_risk} low-risk. High-risk Champions have either not purchased in "
        f"{data.days_threshold_for_risk}+ days or show elevated churn scores."
    )

    callout = "Prioritize the high-risk group for win-back campaigns this week."

    return ChartNarrative(
        chart_id="champion_thermometer",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_funnel_narrative(data: NewCustomerQualityData) -> ChartNarrative:
    """Generate narrative for the new customer funnel chart."""
    total = data.total_new_customers
    conv_2nd = data.conversion_to_2nd * 100
    conv_3rd = data.conversion_to_3rd * 100
    churn_rate = data.new_customer_churn_rate * 100

    headline = f"Only {conv_3rd:.0f}% of new customers reach their 3rd purchase"

    insight = (
        f"We acquired {total:,} new customers. "
        f"{conv_2nd:.0f}% made a second purchase, "
        f"and just {conv_3rd:.0f}% reached the critical third order. "
        f"{churn_rate:.0f}% never returned after their first purchase."
    )

    # Calculate drop-off between stages
    drop_1_to_2 = 100 - conv_2nd
    drop_2_to_3 = conv_2nd - conv_3rd

    if drop_1_to_2 > drop_2_to_3:
        callout = (
            f"The biggest leak is between orders 1 and 2 ({drop_1_to_2:.0f}% drop). "
            f"Focus on first-purchase follow-up."
        )
    else:
        callout = (
            f"Significant drop-off continues between orders 2 and 3. "
            f"Sustained engagement is key."
        )

    return ChartNarrative(
        chart_id="new_customer_funnel",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_quality_distribution_narrative(data: NewCustomerQualityData) -> ChartNarrative:
    """Generate narrative for the new customer quality distribution."""
    low = data.quality_score_distribution.get("low", 0)
    medium = data.quality_score_distribution.get("medium", 0)
    high = data.quality_score_distribution.get("high", 0)
    total = low + medium + high

    low_pct = (low / total * 100) if total > 0 else 0
    high_pct = (high / total * 100) if total > 0 else 0

    headline = f"{high_pct:.0f}% of new customers show high potential"

    insight = (
        f"Based on predicted lifetime value: {high} high-potential, "
        f"{medium} medium-potential, {low} low-potential new customers. "
        f"Average new customer LTV: ${float(data.average_new_customer_ltv):,.0f}."
    )

    if low_pct > 40:
        callout = (
            f"Consider acquisition channel mix - {low_pct:.0f}% of new customers "
            f"have low predicted value."
        )
    else:
        callout = "Acquisition quality looks healthy. Focus on conversion to repeat."

    return ChartNarrative(
        chart_id="quality_distribution",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_churn_timeline_narrative(data: ChurnPredictionData) -> ChartNarrative:
    """Generate narrative for the churn leading indicators timeline."""
    warning_days = data.average_warning_days
    top_predictor = data.top_predictors[0] if data.top_predictors else "inactivity"

    headline = f"Warning signs appear {warning_days:.0f} days before churn"

    indicators_desc = []
    for ind in data.leading_indicators[:3]:
        indicators_desc.append(
            f"{ind['indicator']} (signals ~{ind['avg_days_before_churn']} days ahead)"
        )

    insight = (
        f"The top leading indicators are: {', '.join(indicators_desc)}. "
        f"{top_predictor} is the strongest predictor of churn."
    )

    callout = (
        f"Set up automated alerts for these signals. "
        f"You have a {warning_days:.0f}-day window to intervene."
    )

    return ChartNarrative(
        chart_id="churn_timeline",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_churn_radar_narrative(data: ChurnPredictionData) -> ChartNarrative:
    """Generate narrative for the at-risk customer radar chart."""
    at_risk_pct = data.at_risk_percentage * 100
    at_risk_count = data.total_at_risk
    revenue_at_risk = float(data.potential_revenue_loss)

    headline = f"{at_risk_count:,} customers ({at_risk_pct:.1f}%) are at risk"

    insight = (
        f"At-risk customers show elevated scores across multiple risk dimensions. "
        f"This group represents ${revenue_at_risk:,.0f} in potential revenue loss. "
    )

    if data.radar_dimensions:
        worst_dim = max(data.radar_dimensions, key=lambda x: x.get("at_risk", 0))
        insight += f"Highest risk factor: {worst_dim['dimension']}."

    callout = (
        f"Prioritize retention efforts on the highest-value at-risk customers. "
        f"Early intervention can save 20-40% of churning customers."
    )

    return ChartNarrative(
        chart_id="churn_radar",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


# =============================================================================
# ESSAY 5: WHITESPACE OPPORTUNITIES NARRATIVES
# =============================================================================


def generate_whitespace_opportunity_narrative(data: WhitespaceOpportunityData) -> ChartNarrative:
    """Generate narrative for the whitespace opportunity bar chart."""
    total_value = float(data.total_opportunity_value)
    total_lookalikes = data.total_lookalikes
    top_category = data.top_category
    top_value = float(data.top_category_value)

    headline = f"${total_value:,.0f} in hidden cross-sell opportunities"

    insight = (
        f"We identified {total_lookalikes:,} customers who look like buyers "
        f"but haven't purchased yet. The largest opportunity is in {top_category}, "
        f"representing ${top_value:,.0f} in potential revenue. "
        f"These lookalikes match buyer behavior patterns with {data.avg_similarity_score:.0%} average similarity."
    )

    callout = (
        f"Target {top_category} lookalikes first - they have the highest conversion potential "
        f"based on behavioral similarity to existing buyers."
    )

    return ChartNarrative(
        chart_id="whitespace_opportunity",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_whitespace_similarity_narrative(data: WhitespaceOpportunityData) -> ChartNarrative:
    """Generate narrative for the similarity distribution scatter plot."""
    avg_similarity = data.avg_similarity_score
    total_lookalikes = data.total_lookalikes

    # Count high-similarity lookalikes from distribution
    high_similarity_count = 0
    for bucket in data.similarity_distribution:
        if bucket["bucket"].startswith("0.9") or bucket["bucket"].startswith("0.8"):
            high_similarity_count += bucket["count"]

    high_sim_pct = (high_similarity_count / total_lookalikes * 100) if total_lookalikes > 0 else 0

    headline = f"{high_sim_pct:.0f}% of lookalikes are high-confidence matches"

    insight = (
        f"Similarity scores range from 0.5 to 1.0, with an average of {avg_similarity:.2f}. "
        f"{high_similarity_count:,} customers score 0.8 or above - these are your "
        f"highest-confidence conversion targets."
    )

    callout = (
        f"Prioritize lookalikes with 0.8+ similarity scores for immediate campaigns. "
        f"They're statistically most likely to convert."
    )

    return ChartNarrative(
        chart_id="whitespace_similarity",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_whitespace_comparison_narrative(data: WhitespaceOpportunityData) -> ChartNarrative:
    """Generate narrative for the buyer vs lookalike comparison chart."""
    buyer_clv = data.buyer_profile.get("avg_clv", 0)
    lookalike_clv = data.lookalike_profile.get("avg_clv", 0)
    buyer_sessions = data.buyer_profile.get("avg_sessions", 0)
    lookalike_sessions = data.lookalike_profile.get("avg_sessions", 0)

    headline = "Lookalikes mirror buyer engagement patterns"

    if lookalike_sessions > 0:
        session_ratio = buyer_sessions / lookalike_sessions if lookalike_sessions > 0 else 0
        insight = (
            f"Buyers average {buyer_sessions:.1f} sessions vs {lookalike_sessions:.1f} for lookalikes "
            f"({session_ratio:.1f}x difference). Current buyer CLV is ${buyer_clv:,.0f}, "
            f"while lookalikes show ${lookalike_clv:,.0f} predicted potential."
        )
    else:
        insight = (
            f"Buyers show strong engagement patterns with an average CLV of ${buyer_clv:,.0f}. "
            f"Lookalikes demonstrate similar browsing behavior but haven't converted yet."
        )

    callout = (
        f"Bridge the conversion gap with targeted incentives. "
        f"These customers already show buyer-like engagement."
    )

    return ChartNarrative(
        chart_id="whitespace_comparison",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


# =============================================================================
# ESSAY 6: TRAIT DISCOVERY NARRATIVES
# =============================================================================


def generate_trait_heatmap_narrative(data: TraitDiscoveryData) -> ChartNarrative:
    """Generate narrative for the trait impact heatmap."""
    total_traits = data.total_traits_found
    significant_traits = data.significant_trait_count
    top_revenue = data.top_revenue_trait
    top_retention = data.top_retention_trait

    headline = f"{significant_traits} of {total_traits} traits significantly impact business metrics"

    insight = (
        f"Trait analysis reveals {top_revenue} has the highest revenue impact, "
        f"while {top_retention} drives retention most strongly. "
        f"Average trait coverage across your customer base is {data.avg_coverage:.0%}."
    )

    callout = (
        f"Focus segmentation on high-impact traits like {top_revenue} for revenue "
        f"and {top_retention} for retention campaigns."
    )

    return ChartNarrative(
        chart_id="trait_heatmap",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_trait_impact_narrative(data: TraitDiscoveryData) -> ChartNarrative:
    """Generate narrative for the trait impact bar chart."""
    if not data.top_traits:
        return ChartNarrative(
            chart_id="trait_impact",
            auto_headline="Analyzing trait impacts...",
            auto_insight="No traits analyzed yet.",
            auto_callout="Run trait discovery to identify impactful customer characteristics.",
        )

    top_trait = data.top_traits[0]
    trait_name = top_trait["name"].replace("_", " ").title()
    revenue_impact = top_trait["revenue_impact"]
    retention_impact = top_trait["retention_impact"]

    headline = f"{trait_name} drives both revenue and retention"

    # Build insight about trait distribution
    top_3_traits = [t["name"].replace("_", " ").title() for t in data.top_traits[:3]]
    insight = (
        f"The most impactful traits are: {', '.join(top_3_traits)}. "
        f"{trait_name} shows {revenue_impact:.0%} revenue correlation and "
        f"{retention_impact:.0%} retention correlation."
    )

    # Callout based on recommended uses
    if data.segmentation_traits:
        seg_traits = [t.replace("_", " ").title() for t in data.segmentation_traits[:2]]
        callout = f"Use {', '.join(seg_traits)} for customer segmentation."
    else:
        callout = "Apply high-impact traits to personalization and targeting strategies."

    return ChartNarrative(
        chart_id="trait_impact",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_trait_coverage_narrative(data: TraitDiscoveryData) -> ChartNarrative:
    """Generate narrative for trait coverage distribution."""
    avg_coverage = data.avg_coverage
    total_traits = data.total_traits_found

    # Find trait with highest coverage
    max_coverage_trait = None
    max_coverage = 0
    for coverage in data.trait_coverage:
        if coverage["coverage"] > max_coverage:
            max_coverage = coverage["coverage"]
            max_coverage_trait = coverage["trait"]

    headline = f"Average trait coverage: {avg_coverage:.0%} of customers"

    insight = (
        f"Across {total_traits} discovered traits, average coverage is {avg_coverage:.0%}. "
    )
    if max_coverage_trait:
        insight += (
            f"{max_coverage_trait.replace('_', ' ').title()} has the highest coverage at {max_coverage:.0f}%, "
            f"making it reliable for broad segmentation."
        )

    callout = (
        f"Traits with >80% coverage are best for primary segmentation. "
        f"Lower-coverage traits can identify niche segments."
    )

    return ChartNarrative(
        chart_id="trait_coverage",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


# =============================================================================
# ESSAY 7: SEGMENT ROBUSTNESS NARRATIVES
# =============================================================================


def generate_robustness_gauge_narrative(data: SensitivityData) -> ChartNarrative:
    """Generate narrative for the robustness tier gauge."""
    high_pct = data.high_robustness_percentage
    high_count = data.high_robustness_count
    total = data.total_segments

    headline = f"{high_pct:.0f}% of segments are highly robust"

    tier_summary = []
    for tier, count in data.robustness_tiers.items():
        tier_summary.append(f"{count} {tier.lower()}")

    insight = (
        f"Segment robustness analysis: {', '.join(tier_summary)}. "
        f"{high_count} of {total} segments maintain stable boundaries "
        f"across feature perturbations and time windows."
    )

    if high_pct >= 60:
        callout = "Your segmentation is stable. Safe to build campaigns on these segments."
    elif high_pct >= 40:
        callout = "Some segments are unstable. Review low-robustness segments before targeting."
    else:
        callout = "Warning: Many segments are fragile. Consider simplifying your segmentation model."

    return ChartNarrative(
        chart_id="robustness_gauge",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_feature_stability_narrative(data: SensitivityData) -> ChartNarrative:
    """Generate narrative for feature stability rankings."""
    avg_stability = data.avg_feature_stability
    most_critical = data.most_critical_feature

    headline = f"{most_critical.replace('_', ' ').title()} is your most critical segmentation feature"

    # Build insight from feature stability data
    if data.feature_stability:
        top_stable = data.feature_stability[0]
        least_stable = data.feature_stability[-1] if len(data.feature_stability) > 1 else None

        insight = (
            f"Average feature stability is {avg_stability:.0%}. "
            f"{top_stable['feature'].replace('_', ' ').title()} is most stable "
            f"({top_stable['avg_stability']:.0%})"
        )
        if least_stable:
            insight += (
                f", while {least_stable['feature'].replace('_', ' ').title()} is least stable "
                f"({least_stable['avg_stability']:.0%})."
            )
    else:
        insight = f"Average feature stability across all segments is {avg_stability:.0%}."

    callout = (
        f"Monitor {most_critical.replace('_', ' ').title()} closely - "
        f"changes to this feature will most affect segment composition."
    )

    return ChartNarrative(
        chart_id="feature_stability",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


def generate_time_consistency_narrative(data: SensitivityData) -> ChartNarrative:
    """Generate narrative for time window consistency."""
    # Find segments with lowest time consistency
    if not data.time_consistency:
        return ChartNarrative(
            chart_id="time_consistency",
            auto_headline="Analyzing time stability...",
            auto_insight="Time consistency data not yet available.",
            auto_callout="Run sensitivity analysis with multiple time windows.",
        )

    sorted_consistency = sorted(data.time_consistency, key=lambda x: x["consistency"])
    least_consistent = sorted_consistency[0]
    most_consistent = sorted_consistency[-1]

    headline = f"Segment stability varies from {least_consistent['consistency']:.0%} to {most_consistent['consistency']:.0%}"

    insight = (
        f"'{most_consistent['segment']}' is most stable over time ({most_consistent['consistency']:.0%}), "
        f"while '{least_consistent['segment']}' shows more variability ({least_consistent['consistency']:.0%}). "
        f"High variability suggests the segment definition may be capturing noise."
    )

    callout = (
        f"Be cautious with campaigns targeting '{least_consistent['segment']}' - "
        f"customer membership in this segment fluctuates significantly."
    )

    return ChartNarrative(
        chart_id="time_consistency",
        auto_headline=headline,
        auto_insight=insight,
        auto_callout=callout,
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================


def generate_all_narratives(data: EssayDataBundle) -> dict[str, ChartNarrative]:
    """Generate all narratives for the essay.

    Args:
        data: EssayDataBundle with all essay data

    Returns:
        Dictionary mapping chart_id to ChartNarrative
    """
    narratives = {}

    # Essay 1: Loyalty Journey
    narratives["loyalty_sankey"] = generate_loyalty_sankey_narrative(data.loyalty_journey)
    narratives["loyalty_threshold"] = generate_loyalty_threshold_narrative(data.loyalty_journey)

    # Essay 2: Champion Fragility
    narratives["champion_scatter"] = generate_champion_scatter_narrative(data.champion_fragility)
    narratives["champion_thermometer"] = generate_champion_thermometer_narrative(data.champion_fragility)

    # Essay 3: New Customer Quality
    narratives["new_customer_funnel"] = generate_funnel_narrative(data.new_customer_quality)
    narratives["quality_distribution"] = generate_quality_distribution_narrative(data.new_customer_quality)

    # Essay 4: Churn Prediction
    narratives["churn_timeline"] = generate_churn_timeline_narrative(data.churn_prediction)
    narratives["churn_radar"] = generate_churn_radar_narrative(data.churn_prediction)

    # Essay 5: Whitespace Opportunities (if data available)
    if data.whitespace:
        narratives["whitespace_opportunity"] = generate_whitespace_opportunity_narrative(data.whitespace)
        narratives["whitespace_similarity"] = generate_whitespace_similarity_narrative(data.whitespace)
        narratives["whitespace_comparison"] = generate_whitespace_comparison_narrative(data.whitespace)

    # Essay 6: Trait Discovery (if data available)
    if data.trait_discovery:
        narratives["trait_heatmap"] = generate_trait_heatmap_narrative(data.trait_discovery)
        narratives["trait_impact"] = generate_trait_impact_narrative(data.trait_discovery)
        narratives["trait_coverage"] = generate_trait_coverage_narrative(data.trait_discovery)

    # Essay 7: Segment Robustness (if data available)
    if data.sensitivity:
        narratives["robustness_gauge"] = generate_robustness_gauge_narrative(data.sensitivity)
        narratives["feature_stability"] = generate_feature_stability_narrative(data.sensitivity)
        narratives["time_consistency"] = generate_time_consistency_narrative(data.sensitivity)

    return narratives


def export_narratives_to_dict(narratives: dict[str, ChartNarrative]) -> dict[str, dict[str, str]]:
    """Export narratives to a dictionary format suitable for YAML.

    Args:
        narratives: Dictionary of ChartNarrative objects

    Returns:
        Dictionary with chart_id -> {headline, insight, callout}
    """
    result = {}
    for chart_id, narrative in narratives.items():
        result[chart_id] = {
            "headline": narrative.auto_headline,
            "insight": narrative.auto_insight,
            "callout": narrative.auto_callout,
        }
    return result
