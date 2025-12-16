"""
Essay 6: What Makes Customers Different

This essay reveals the customer traits that actually matter - which characteristics
drive revenue, retention, and can be used for personalization.

Key visualizations:
- Impact Heatmap: Trait impact across revenue, retention, personalization
- Trait Bars: Top traits ranked by overall impact
- Coverage Chart: How many customers each trait applies to
"""

from src.essays.base import ChartNarrative, ChartSpec, EssaySection, ScrollyStep
from src.essays.data_queries import TraitDiscoveryData
from src.essays.narratives import (
    generate_trait_coverage_narrative,
    generate_trait_heatmap_narrative,
    generate_trait_impact_narrative,
)


def create_trait_insights_section(
    data: TraitDiscoveryData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Trait Insights essay section.

    Args:
        data: TraitDiscoveryData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the trait insights essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    heatmap_narrative = (
        narratives.get("trait_heatmap")
        or generate_trait_heatmap_narrative(data)
    )
    impact_narrative = (
        narratives.get("trait_impact")
        or generate_trait_impact_narrative(data)
    )
    coverage_narrative = (
        narratives.get("trait_coverage")
        or generate_trait_coverage_narrative(data)
    )

    # Create chart specs
    heatmap_chart = create_impact_heatmap_spec("trait_heatmap", data)
    impact_chart = create_trait_impact_spec("trait_impact", data)
    coverage_chart = create_trait_coverage_spec("trait_coverage", data)

    # Build scrollytelling steps
    total_traits = data.total_traits_found
    significant_count = data.significant_trait_count
    top_revenue = data.top_revenue_trait.replace("_", " ").title()
    top_retention = data.top_retention_trait.replace("_", " ").title()
    avg_coverage = data.avg_coverage

    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"We looked at {total_traits} things we know about customers - "
                f"like what they buy, when they shop, and how they browse. "
                f"{significant_count} of these actually predict who spends more and who stays longer."
            ),
            chart_action="showOverview",
            action_params={},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"'{top_revenue}' is your best predictor of high spenders. "
                f"For example: customers who shop certain categories spend 2-3x more "
                f"than average. Target campaigns around this."
            ),
            chart_action="highlightTrait",
            action_params={"trait": data.top_revenue_trait, "dimension": "revenue"},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"For keeping customers long-term, '{top_retention}' matters most. "
                f"Customers strong in this trait stay 40% longer. "
                f"Use this for loyalty program targeting."
            ),
            chart_action="highlightTrait",
            action_params={"trait": data.top_retention_trait, "dimension": "retention"},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"Good news: we have data on {avg_coverage:.0%} of customers for these traits. "
                f"That's enough to build reliable segments and personalized campaigns."
            ),
            chart_action="showCoverage",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"{significant_count} of {total_traits} traits significantly impact business metrics. "
        f"'{top_revenue}' drives revenue; '{top_retention}' drives retention."
    )

    marketing_narrative = f"""
**What This Means For Your Marketing**

We've identified which customer characteristics actually matter for your business.
Here's how to use them:

**🎯 For Higher Revenue**

Focus on '{top_revenue}'. Customers who index high on this trait spend significantly more.

*Example:* If this is "Product Category" - customers who buy from premium categories
spend 2-3x more. Create campaigns targeting these categories.

{_format_top_traits_simple(data, "revenue")}

**🔄 For Better Retention**

'{top_retention}' predicts who stays. Monitor customers who drop in this trait -
they're at risk of leaving.

{_format_top_traits_simple(data, "retention")}

**📧 For Personalization**

Use these traits to customize emails and offers:
{_format_trait_list(data.personalization_traits)}

*Example:* If "Device Type" matters, send mobile-optimized content to mobile shoppers.

**Your Next Steps**

1. **This week:** Create a segment of high '{top_revenue}' customers for a revenue campaign
2. **This month:** Set up alerts when '{top_retention}' scores drop for VIP customers
3. **Ongoing:** Use traits with {avg_coverage:.0%}+ coverage for your main campaigns
"""

    technical_details = f"""
**Methodology**

Trait discovery analyzes customer attributes to determine their impact on key metrics:

- **Revenue Impact**: Correlation between trait values and total customer revenue
- **Retention Impact**: Correlation between trait values and customer retention rate
- **Personalization Value**: Usefulness of trait for content/offer customization

**Impact Scoring**

Scores range from 0 to 1, where:
- 0.0-0.3: Low impact (likely noise)
- 0.3-0.6: Moderate impact (useful for some applications)
- 0.6-1.0: High impact (core segmentation driver)

**Significance Testing**

A trait is marked "significant" if:
- Overall impact score ≥0.5
- Coverage ≥20% of customer base
- Distinct values ≤100 (for categorical traits)

**Trait Impact Details**

{_format_trait_impact_table(data)}

**Trait Coverage Details**

{_format_coverage_table(data)}
"""

    # Key metrics for this section
    key_metrics = {
        "total_traits": total_traits,
        "significant_traits": significant_count,
        "top_revenue_trait": data.top_revenue_trait,
        "top_retention_trait": data.top_retention_trait,
        "avg_coverage": avg_coverage,
    }

    return EssaySection(
        section_id="trait-insights",
        title="What Makes Customers Different",
        chart=heatmap_chart,
        narrative=heatmap_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[impact_chart, coverage_chart],
        key_metrics=key_metrics,
        appendix_data={
            "top_traits": data.top_traits,
            "impact_heatmap": data.impact_heatmap,
            "trait_coverage": data.trait_coverage,
            "segmentation_traits": data.segmentation_traits,
            "personalization_traits": data.personalization_traits,
            "retention_traits": data.retention_traits,
        },
    )


# =============================================================================
# CHART SPECS
# =============================================================================


def create_impact_heatmap_spec(chart_id: str, data: TraitDiscoveryData) -> ChartSpec:
    """Create heatmap spec for trait impact across dimensions."""
    heatmap_data = [
        {
            "trait": item["trait"].replace("_", " ").title(),
            "dimension": dim,
            "value": item[dim],
        }
        for item in data.impact_heatmap
        for dim in ["revenue", "retention", "personalization"]
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="heatmap",
        data=heatmap_data,
        config={
            "x": "dimension",
            "y": "trait",
            "value": "value",
            "colorScale": ["#fee8c8", "#e34a33"],
            "title": "Trait Impact Heatmap",
            "xLabel": "Impact Dimension",
            "yLabel": "Customer Trait",
        },
    )


def create_trait_impact_spec(chart_id: str, data: TraitDiscoveryData) -> ChartSpec:
    """Create horizontal bar chart for overall trait impact."""
    impact_data = [
        {
            "trait": t["name"].replace("_", " ").title(),
            "overall_score": t["overall_score"],
            "is_significant": t["is_significant"],
        }
        for t in data.top_traits[:10]
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="horizontal_bar",
        data=impact_data,
        config={
            "x": "overall_score",
            "y": "trait",
            "colorBy": "is_significant",
            "colors": {"true": "#27ae60", "false": "#95a5a6"},
            "title": "Top Traits by Impact",
            "xLabel": "Overall Impact Score",
            "yLabel": "Trait",
            "xDomain": [0, 1],
        },
    )


def create_trait_coverage_spec(chart_id: str, data: TraitDiscoveryData) -> ChartSpec:
    """Create bar chart for trait coverage distribution."""
    coverage_data = [
        {
            "trait": item["trait"].replace("_", " ").title(),
            "coverage": item["coverage"],
            "distinct_values": item["distinct_values"],
        }
        for item in data.trait_coverage
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar",
        data=coverage_data,
        config={
            "x": "trait",
            "y": "coverage",
            "color": "#3498db",
            "title": "Trait Coverage (% of Customers)",
            "xLabel": "Trait",
            "yLabel": "Coverage (%)",
            "yDomain": [0, 100],
            "rotateLabels": True,
        },
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================


def _format_top_traits(data: TraitDiscoveryData) -> str:
    """Format top traits as a descriptive list."""
    lines = []
    for t in data.top_traits[:5]:
        name = t["name"].replace("_", " ").title()
        revenue = t["revenue_impact"]
        retention = t["retention_impact"]
        lines.append(
            f"- **{name}**: {revenue:.0%} revenue impact, {retention:.0%} retention impact"
        )
    return "\n".join(lines)


def _format_top_traits_simple(data: TraitDiscoveryData, metric: str) -> str:
    """Format top traits in plain language for marketers.

    Args:
        data: TraitDiscoveryData with trait info
        metric: 'revenue' or 'retention'

    Returns:
        Plain language description of top traits
    """
    impact_key = f"{metric}_impact"
    sorted_traits = sorted(
        data.top_traits[:5],
        key=lambda t: t.get(impact_key, 0),
        reverse=True
    )

    lines = []
    for i, t in enumerate(sorted_traits[:3], 1):
        name = t["name"].replace("_", " ").title()
        impact = t.get(impact_key, 0)

        # Translate impact score to plain language
        if impact >= 0.7:
            strength = "strongly predicts"
        elif impact >= 0.5:
            strength = "moderately predicts"
        else:
            strength = "slightly predicts"

        lines.append(f"{i}. **{name}** - {strength} higher {metric} ({impact:.0%} correlation)")

    return "\n".join(lines)


def _format_trait_list(traits: list[str]) -> str:
    """Format a list of traits as bullet points."""
    if not traits:
        return "- (None identified)"
    return "\n".join(f"- {t.replace('_', ' ').title()}" for t in traits[:5])


def _format_trait_impact_table(data: TraitDiscoveryData) -> str:
    """Format trait impacts as a text table."""
    lines = ["| Trait | Revenue | Retention | Personalization | Overall |"]
    lines.append("|-------|---------|-----------|-----------------|---------|")
    for t in data.top_traits[:10]:
        name = t["name"].replace("_", " ").title()[:15]
        lines.append(
            f"| {name} | {t['revenue_impact']:.2f} | {t['retention_impact']:.2f} | "
            f"{t['personalization_value']:.2f} | {t['overall_score']:.2f} |"
        )
    return "\n".join(lines)


def _format_coverage_table(data: TraitDiscoveryData) -> str:
    """Format coverage as a text table."""
    lines = ["| Trait | Coverage | Distinct Values |"]
    lines.append("|-------|----------|-----------------|")
    for item in data.trait_coverage:
        name = item["trait"].replace("_", " ").title()[:15]
        lines.append(f"| {name} | {item['coverage']:.1f}% | {item['distinct_values']} |")
    return "\n".join(lines)
