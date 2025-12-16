"""
Essay 5: Hidden Cross-Sell Opportunities

This essay reveals untapped revenue through whitespace analysis - finding customers
who look like buyers but haven't purchased yet in specific categories.

Key visualizations:
- Opportunity Bar: Revenue potential by category
- Similarity Scatter: Lookalike confidence distribution
- Buyer Comparison: Behavioral profiles of buyers vs lookalikes
"""

from src.essays.base import ChartNarrative, ChartSpec, EssaySection, ScrollyStep
from src.essays.data_queries import WhitespaceOpportunityData
from src.essays.narratives import (
    generate_whitespace_comparison_narrative,
    generate_whitespace_opportunity_narrative,
    generate_whitespace_similarity_narrative,
)


def create_whitespace_opportunities_section(
    data: WhitespaceOpportunityData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Whitespace Opportunities essay section.

    Args:
        data: WhitespaceOpportunityData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the whitespace opportunities essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    opportunity_narrative = (
        narratives.get("whitespace_opportunity")
        or generate_whitespace_opportunity_narrative(data)
    )
    similarity_narrative = (
        narratives.get("whitespace_similarity")
        or generate_whitespace_similarity_narrative(data)
    )
    comparison_narrative = (
        narratives.get("whitespace_comparison")
        or generate_whitespace_comparison_narrative(data)
    )

    # Create chart specs
    opportunity_chart = create_opportunity_bar_spec("whitespace_opportunity", data)
    similarity_chart = create_similarity_scatter_spec("whitespace_similarity", data)
    comparison_chart = create_buyer_comparison_spec("whitespace_comparison", data)

    # Build scrollytelling steps
    total_value = float(data.total_opportunity_value)
    total_lookalikes = data.total_lookalikes
    top_category = data.top_category
    top_value = float(data.top_category_value)
    avg_similarity = data.avg_similarity_score

    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"Hidden in your customer base is ${total_value:,.0f} in untapped potential. "
                f"We found {total_lookalikes:,} customers who look like buyers but haven't "
                f"purchased yet."
            ),
            chart_action="showOverview",
            action_params={},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"The largest opportunity is in {top_category}, representing "
                f"${top_value:,.0f} in potential revenue. These lookalikes match "
                f"buyer behavior patterns with {avg_similarity:.0%} average similarity."
            ),
            chart_action="highlightCategory",
            action_params={"category": top_category},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"Lookalikes with similarity scores above 0.8 are your highest-confidence "
                f"targets. They behave almost identically to existing buyers - they just "
                f"haven't converted yet."
            ),
            chart_action="highlightHighSimilarity",
            action_params={"threshold": 0.8},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                "Comparing buyer and lookalike profiles reveals the gap. "
                "These customers already show buyer-like engagement - "
                "a targeted incentive could bridge the conversion gap."
            ),
            chart_action="showComparison",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"${total_value:,.0f} in hidden cross-sell opportunities identified. "
        f"{total_lookalikes:,} customers look like buyers but haven't purchased. "
        f"Top opportunity: {top_category} (${top_value:,.0f})."
    )

    marketing_narrative = f"""
**Unlocking Hidden Revenue Through Lookalike Targeting**

Cross-sell opportunities aren't always obvious. Traditional approaches look at what
customers *have* bought to suggest what they *might* buy. But what about customers
who are *ready* to buy but haven't yet?

Our whitespace analysis identified {total_lookalikes:,} customers who match the
behavioral profile of buyers in specific categories - but haven't made a purchase.
Combined, they represent ${total_value:,.0f} in potential revenue.

**The Opportunity Breakdown**

{_format_opportunity_list(data)}

**How Lookalike Matching Works**

We compare customer behavior vectors using similarity scoring. A customer with a
0.9 similarity score to buyers in a category behaves almost identically to someone
who has purchased - they browse the same products, spend similar time on pages,
and show comparable engagement patterns.

**Recommended Campaign Strategy**

1. **Tier 1 (Similarity ≥0.9)**: Immediate conversion campaign with modest incentive
2. **Tier 2 (0.8-0.9)**: Educational content + offer after engagement
3. **Tier 3 (0.7-0.8)**: Brand awareness and category introduction
"""

    technical_details = f"""
**Methodology**

Whitespace analysis uses FAISS-based approximate nearest neighbor search to find
customers whose behavioral vectors are similar to category buyers but haven't
purchased in that category.

**Similarity Scoring**

- Feature vector dimensions: sessions, page views, cart additions, time on site
- Distance metric: Cosine similarity normalized to [0, 1]
- Minimum threshold: 0.5 for inclusion

**Opportunity Valuation**

Expected revenue = Buyer average CLV × Number of lookalikes × Expected conversion rate

Conversion rate estimated at 15% based on historical lookalike campaign performance.

**Category Performance**

{_format_opportunity_table(data)}

**Similarity Distribution**

{_format_similarity_distribution(data)}
"""

    # Key metrics for this section
    key_metrics = {
        "total_opportunity_value": total_value,
        "total_lookalikes": total_lookalikes,
        "top_category": top_category,
        "top_category_value": top_value,
        "avg_similarity": avg_similarity,
        "opportunity_count": data.total_opportunities,
    }

    return EssaySection(
        section_id="whitespace-opportunities",
        title="Hidden Cross-Sell Opportunities",
        chart=opportunity_chart,
        narrative=opportunity_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[similarity_chart, comparison_chart],
        key_metrics=key_metrics,
        appendix_data={
            "opportunities": data.top_opportunities,
            "similarity_distribution": data.similarity_distribution,
            "buyer_profile": data.buyer_profile,
            "lookalike_profile": data.lookalike_profile,
        },
    )


# =============================================================================
# CHART SPECS
# =============================================================================


def create_opportunity_bar_spec(chart_id: str, data: WhitespaceOpportunityData) -> ChartSpec:
    """Create bar chart spec for opportunity by category."""
    chart_data = [
        {
            "category": opp["category"],
            "value": opp["opportunity_value"],
            "lookalikes": opp["lookalikes"],
            "buyers": opp["buyers"],
        }
        for opp in data.top_opportunities[:8]
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar",
        data=chart_data,
        config={
            "x": "category",
            "y": "value",
            "color": "#2ecc71",
            "title": "Revenue Opportunity by Category",
            "xLabel": "Category",
            "yLabel": "Opportunity Value ($)",
            "sortBy": "value",
            "sortOrder": "descending",
        },
    )


def create_similarity_scatter_spec(chart_id: str, data: WhitespaceOpportunityData) -> ChartSpec:
    """Create scatter plot spec for similarity distribution."""
    # Flatten opportunities into points
    scatter_data = []
    for opp in data.top_opportunities:
        scatter_data.append({
            "category": opp["category"],
            "similarity": opp["avg_similarity"],
            "value": opp["opportunity_value"],
            "size": opp["lookalikes"],
        })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="scatter",
        data=scatter_data,
        config={
            "x": "similarity",
            "y": "value",
            "size": "size",
            "color": "category",
            "title": "Opportunity vs. Confidence",
            "xLabel": "Average Similarity Score",
            "yLabel": "Opportunity Value ($)",
            "xDomain": [0.5, 1.0],
        },
    )


def create_buyer_comparison_spec(chart_id: str, data: WhitespaceOpportunityData) -> ChartSpec:
    """Create grouped bar chart comparing buyer vs lookalike profiles."""
    comparison_data = []

    metrics = ["avg_sessions", "avg_page_views", "avg_cart_additions"]
    labels = ["Sessions", "Page Views", "Cart Additions"]

    for metric, label in zip(metrics, labels, strict=False):
        buyer_val = data.buyer_profile.get(metric, 0)
        lookalike_val = data.lookalike_profile.get(metric, 0)
        comparison_data.append({
            "metric": label,
            "Buyers": buyer_val,
            "Lookalikes": lookalike_val,
        })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="grouped_bar",
        data=comparison_data,
        config={
            "x": "metric",
            "groups": ["Buyers", "Lookalikes"],
            "colors": ["#3498db", "#e74c3c"],
            "title": "Buyer vs. Lookalike Behavior",
            "xLabel": "Metric",
            "yLabel": "Average Value",
        },
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================


def _format_opportunity_list(data: WhitespaceOpportunityData) -> str:
    """Format top opportunities as a bullet list."""
    lines = []
    for opp in data.top_opportunities[:5]:
        lines.append(
            f"- **{opp['category']}**: ${opp['opportunity_value']:,.0f} potential "
            f"({opp['lookalikes']:,} lookalikes, {opp['avg_similarity']:.0%} avg similarity)"
        )
    return "\n".join(lines)


def _format_opportunity_table(data: WhitespaceOpportunityData) -> str:
    """Format opportunities as a text table."""
    lines = ["| Category | Buyers | Lookalikes | Opportunity | Similarity |"]
    lines.append("|----------|--------|------------|-------------|------------|")
    for opp in data.top_opportunities[:8]:
        lines.append(
            f"| {opp['category'][:15]} | {opp['buyers']:,} | {opp['lookalikes']:,} | "
            f"${opp['opportunity_value']:,.0f} | {opp['avg_similarity']:.2f} |"
        )
    return "\n".join(lines)


def _format_similarity_distribution(data: WhitespaceOpportunityData) -> str:
    """Format similarity distribution as text."""
    lines = []
    for bucket in data.similarity_distribution:
        lines.append(f"- {bucket['bucket']}: {bucket['count']:,} customers")
    return "\n".join(lines)
